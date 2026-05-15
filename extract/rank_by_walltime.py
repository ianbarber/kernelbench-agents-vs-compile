"""Rank kernels by wall-time from PyTorch profiler Chrome traces.

The traces under `baselines/results/traces/` were captured with cudagraphs OFF
(`compile_default`). PyTorch profiler on this build emitted only `cpu_op`
phase-X events (no separate GPU-side events). The `dur` field on a `cpu_op`
event whose `args.kernel_backend == "triton"` is the wall-time around the
launch+execute (kernels are dispatched and the launch op encompasses it on
this backend). For aten dispatcher ops (e.g. `aten::mm`,
`aten::_efficient_attention_forward`) `dur` is similarly the dispatcher span.

Caveat: aten ops form a hierarchy (`aten::_scaled_dot_product_efficient_attention`
calls `aten::_efficient_attention_forward`). We dedupe by removing events
whose time-span is fully nested inside a parent event from a richer-info
"leaf"; concretely we keep `aten::_efficient_attention_forward` (the
backend-specific leaf) and drop its parent `aten::_scaled_dot_product_*`
when both are present. For Triton kernels we always treat them as leaves.
We also drop the umbrella `Torch-Compiled Region` / `Pregraph bytecode` /
`AOTDispatcher Runtime Wrapper Prologue` / `TorchDynamo Cache Lookup`
container events.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
TRACES = {
    "prefill_512_b1": ROOT / "baselines/results/traces/default_prefill_512_b1.json",
    "decode_ctx512_b1": ROOT / "baselines/results/traces/default_decode_ctx512_b1.json",
}
OUT_DIR = Path(__file__).resolve().parent

# Container ops that wrap others; we never want to count their duration as
# "kernel" time. They are dispatcher / profiler bookkeeping.
CONTAINER_NAMES = {
    "Torch-Compiled Region",
    "Pregraph bytecode",
    "AOTDispatcher Runtime Wrapper Prologue",
    "TorchDynamo Cache Lookup",
    # SDPA dispatcher op — the actual backend leaf is _efficient_attention_forward
    # (or _flash_attention_forward / etc.). Dropping the parent avoids double-count.
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::scaled_dot_product_attention",
}
# Some "Torch-Compiled Region: 0/1" etc. have variable suffixes; prefix-match.
CONTAINER_PREFIXES = (
    "Torch-Compiled Region",
)

# Aten ops that are "view-only" / metadata-only and never launch a kernel.
# We separately report these but don't include in the ranking-by-walltime
# top list because they aren't candidates for kernel replacement.
NO_KERNEL_ATEN = {
    "aten::transpose",
    "aten::as_strided",
    "aten::empty",
    "aten::empty_strided",
    "aten::view",
    "aten::reshape",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::expand",
    "aten::permute",
    "aten::slice",
    "aten::detach",
    "aten::contiguous",
    "aten::t",
    "aten::_unsafe_view",
}


def _is_container(name: str) -> bool:
    if name in CONTAINER_NAMES:
        return True
    for p in CONTAINER_PREFIXES:
        if name.startswith(p):
            return True
    return False


def _is_kernel_event(ev: dict) -> bool:
    """An event represents real GPU work."""
    if ev.get("ph") != "X" or ev.get("cat") != "cpu_op":
        return False
    name = ev.get("name", "")
    if _is_container(name):
        return False
    args = ev.get("args", {})
    # Inductor-emitted triton kernels carry kernel_backend/kernel_name.
    if args.get("kernel_backend") == "triton":
        return True
    # aten dispatcher ops — include unless explicitly known no-op (view).
    if name.startswith("aten::"):
        if name in NO_KERNEL_ATEN:
            return False
        return True
    # cudnn/cublas-named events would land here too (none in current traces).
    return False


def rank_trace(trace_path: Path) -> Dict:
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"]

    triton_stats: Dict[str, Dict] = defaultdict(lambda: {"total_us": 0.0, "n": 0, "kind": "triton"})
    aten_stats: Dict[str, Dict] = defaultdict(lambda: {"total_us": 0.0, "n": 0, "kind": "aten"})
    sample_args: Dict[str, dict] = {}

    for ev in events:
        if not _is_kernel_event(ev):
            continue
        name = ev["name"]
        dur = float(ev.get("dur", 0.0))
        args = ev.get("args", {})
        is_triton = args.get("kernel_backend") == "triton"
        bucket = triton_stats if is_triton else aten_stats
        bucket[name]["total_us"] += dur
        bucket[name]["n"] += 1
        if name not in sample_args:
            keep = {
                "Input Dims": args.get("Input Dims"),
                "Input Strides": args.get("Input Strides"),
                "Input type": args.get("Input type"),
            }
            if is_triton:
                keep.update({
                    "kernel_file": args.get("kernel_file"),
                    "kernel_hash": args.get("kernel_hash"),
                    "num_warps": args.get("num_warps"),
                    "num_stages": args.get("num_stages"),
                    "kernel_kwargs": args.get("kernel_kwargs"),
                })
            sample_args[name] = keep

    # Build merged ranking list.
    all_total = sum(s["total_us"] for s in triton_stats.values()) + sum(
        s["total_us"] for s in aten_stats.values()
    )

    def _entries(bucket):
        for name, st in bucket.items():
            yield {
                "name": name,
                "kind": st["kind"],
                "total_us": st["total_us"],
                "pct_total": 100.0 * st["total_us"] / all_total if all_total else 0.0,
                "invocations": st["n"],
                "mean_us": st["total_us"] / st["n"] if st["n"] else 0.0,
                "sample_inputs": sample_args.get(name, {}),
            }

    rows = list(_entries(triton_stats)) + list(_entries(aten_stats))
    rows.sort(key=lambda r: r["total_us"], reverse=True)

    triton_total = sum(s["total_us"] for s in triton_stats.values())
    aten_total = sum(s["total_us"] for s in aten_stats.values())

    return {
        "trace_path": str(trace_path),
        "total_us": all_total,
        "triton_total_us": triton_total,
        "aten_total_us": aten_total,
        "triton_pct": 100.0 * triton_total / all_total if all_total else 0.0,
        "aten_pct": 100.0 * aten_total / all_total if all_total else 0.0,
        "n_distinct_triton": len(triton_stats),
        "n_distinct_aten": len(aten_stats),
        "ranking": rows,
    }


def _md_table(ranking: List[dict], top_n: int = 20) -> List[str]:
    lines = [
        "| rank | kind | walltime_ms | %total | calls | mean_us | name |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranking[:top_n], 1):
        # Trim long fused names for readability.
        n = r["name"]
        if len(n) > 90:
            n = n[:87] + "..."
        lines.append(
            f"| {i} | {r['kind']} | {r['total_us']/1000:.3f} | "
            f"{r['pct_total']:.2f} | {r['invocations']} | {r['mean_us']:.2f} | "
            f"`{n}` |"
        )
    return lines


def _coverage(ranking: List[dict], k: int, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * sum(r["total_us"] for r in ranking[:k]) / total


def main():
    md = ["# Kernel wall-time ranking (cudagraphs OFF profiler traces)", ""]
    summary = {}
    for wl_name, path in TRACES.items():
        result = rank_trace(path)
        out_json = OUT_DIR / f"ranking_{wl_name.split('_')[0]}.json"  # ranking_prefill.json / ranking_decode.json
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        summary[wl_name] = {
            "n_distinct_triton": result["n_distinct_triton"],
            "n_distinct_aten": result["n_distinct_aten"],
            "triton_pct": result["triton_pct"],
            "aten_pct": result["aten_pct"],
            "top5_pct": _coverage(result["ranking"], 5, result["total_us"]),
            "top10_pct": _coverage(result["ranking"], 10, result["total_us"]),
            "top20_pct": _coverage(result["ranking"], 20, result["total_us"]),
        }

        md.append(f"## {wl_name}")
        md.append("")
        md.append(f"- Trace: `{path.relative_to(ROOT)}`")
        md.append(f"- Total kernel-event time: {result['total_us']/1000:.2f} ms")
        md.append(
            f"- Triton (inductor codegen): {result['triton_total_us']/1000:.2f} ms "
            f"({result['triton_pct']:.1f}%), {result['n_distinct_triton']} distinct kernels"
        )
        md.append(
            f"- Aten (cuBLAS/cuDNN/SDPA backends): {result['aten_total_us']/1000:.2f} ms "
            f"({result['aten_pct']:.1f}%), {result['n_distinct_aten']} distinct ops"
        )
        md.append(
            f"- Coverage: top-5 = {summary[wl_name]['top5_pct']:.1f}%  "
            f"top-10 = {summary[wl_name]['top10_pct']:.1f}%  "
            f"top-20 = {summary[wl_name]['top20_pct']:.1f}%"
        )
        md.append("")
        md.append("### Top 20 by walltime")
        md.append("")
        md += _md_table(result["ranking"], top_n=20)
        md.append("")

    md.append("## Summary")
    md.append("")
    md.append("| workload | triton% | aten% | top5% | top10% | top20% | #triton | #aten |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for wl, s in summary.items():
        md.append(
            f"| {wl} | {s['triton_pct']:.1f} | {s['aten_pct']:.1f} | "
            f"{s['top5_pct']:.1f} | {s['top10_pct']:.1f} | {s['top20_pct']:.1f} | "
            f"{s['n_distinct_triton']} | {s['n_distinct_aten']} |"
        )

    with open(OUT_DIR / "ranking.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # Also write a small summary json to make machine consumption easy.
    with open(OUT_DIR / "ranking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {OUT_DIR/'ranking.md'}")
    print(f"wrote {OUT_DIR/'ranking_prefill.json'}")
    print(f"wrote {OUT_DIR/'ranking_decode.json'}")
    print(f"wrote {OUT_DIR/'ranking_summary.json'}")


if __name__ == "__main__":
    main()
