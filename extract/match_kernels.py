"""Match ranked kernel names to inductor source slices.

For every fused-Triton kernel that appears in the profiler ranking, locate the
corresponding `def triton_*` in any of the per-workload `output_code.py` files
under `extract/inductor_debug/by_workload/`. Slice out the kernel definition
plus the immediately-preceding `@triton_heuristics.*(...)` / `@triton.jit`
decorators and any module-scope tiling / autotune-config metadata that
references the kernel name.

For aten ops (`aten::mm`, `aten::_efficient_attention_forward`, etc.) write
`extract/aten_calls.json` with the same stats — those are *not* candidates for
agents (they're already on cuBLAS/cuDNN/SDPA backends).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXTRACT_DIR = Path(__file__).resolve().parent
ROOT = EXTRACT_DIR.parent
BY_WL = EXTRACT_DIR / "inductor_debug" / "by_workload"
CACHE_DIR = EXTRACT_DIR / "inductor_debug" / "cache"
KERNELS_OUT = EXTRACT_DIR / "kernels"
KERNELS_OUT.mkdir(parents=True, exist_ok=True)


def _collect_output_codes() -> Dict[str, Path]:
    """Return list of all output_code.py files keyed by workload+graph."""
    out: Dict[str, Path] = {}
    for wl_dir in sorted(BY_WL.iterdir()):
        if not wl_dir.is_dir():
            continue
        for oc in wl_dir.rglob("output_code.py"):
            graph_dir = oc.parent
            # graph_dir name is e.g. "model__9_inference_9.9"
            key = f"{wl_dir.name}::{graph_dir.name}"
            out[key] = oc
    return out


def _collect_cache_kernel_files() -> Dict[str, Path]:
    """Return {kernel_name: standalone .py path in cache dir}.

    Cache files always have inductor_meta {'kernel_name': '<name>'} embedded.
    """
    out: Dict[str, Path] = {}
    if not CACHE_DIR.exists():
        return out
    for py in CACHE_DIR.rglob("*.py"):
        if py.parent.name.endswith(".debug"):
            continue
        try:
            text = py.read_text()
        except Exception:
            continue
        m = re.search(r"'kernel_name':\s*'([^']+)'", text)
        if m:
            out[m.group(1)] = py
    return out


# Regex to find a top-level kernel definition: @triton_heuristics.<kind>(...)
# possibly with multiple decorators, then @triton.jit, then `def <name>(...)`
# We slice from the top decorator line to the end of the function body
# (next top-level def/comment or EOF).
_DEF_LINE_RE = re.compile(r"^def\s+(triton_\w+)\s*\(", re.MULTILINE)


def _slice_kernel(text: str, kernel_name: str) -> Optional[str]:
    """Extract decorators + def for kernel_name from a concatenated output_code."""
    lines = text.split("\n")
    # Find the def line.
    def_idx = None
    for i, ln in enumerate(lines):
        m = re.match(rf"^def\s+{re.escape(kernel_name)}\s*\(", ln)
        if m:
            def_idx = i
            break
    if def_idx is None:
        return None
    # Walk backwards to capture decorators and any module-level metadata
    # blocks (lines starting with '@' or continuing from one).
    start = def_idx
    j = def_idx - 1
    paren_balance = 0
    while j >= 0:
        ln = lines[j]
        stripped = ln.strip()
        if not stripped:
            # blank — keep going only if we've already grabbed at least one decorator
            if start < def_idx:
                start = j
                j -= 1
                continue
            else:
                break
        if stripped.startswith("@") or paren_balance > 0 or stripped.endswith(")"):
            # decorator (possibly multi-line)
            # update balance crude check
            paren_balance += ln.count(")") - ln.count("(")
            start = j
            j -= 1
            continue
        if stripped.startswith("#") and start < def_idx:
            start = j
            j -= 1
            continue
        break

    # Walk forward to end of function body.
    end = def_idx + 1
    n = len(lines)
    while end < n:
        ln = lines[end]
        # Function body is indented (or blank). New top-level def/decorator/class/import ends body.
        if ln.startswith("def ") or ln.startswith("class ") or ln.startswith("@") or \
                (ln and not ln[0].isspace() and not ln.startswith("#")):
            break
        end += 1
    # Trim trailing blanks.
    while end > def_idx + 1 and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _find_kernel_source(kernel_name: str, output_codes: Dict[str, Path],
                       cache_kernels: Dict[str, Path]) -> Tuple[Optional[str], List[str]]:
    """Return (source_text, list_of_workload_graphs_that_emitted_it)."""
    sources_in: List[str] = []
    source_text: Optional[str] = None

    # 1) Try the standalone cache file (cleanest single-kernel source).
    if kernel_name in cache_kernels:
        try:
            source_text = cache_kernels[kernel_name].read_text()
        except Exception:
            source_text = None

    # 2) Identify which workload/graph output_code.py contained this kernel.
    for key, oc_path in output_codes.items():
        try:
            text = oc_path.read_text()
        except Exception:
            continue
        if re.search(rf"^def\s+{re.escape(kernel_name)}\s*\(", text, re.MULTILINE):
            sources_in.append(key)
            if source_text is None:
                sliced = _slice_kernel(text, kernel_name)
                if sliced:
                    source_text = sliced

    return source_text, sources_in


def _stub_microbench(kernel_name: str, sample_inputs: dict) -> str:
    """Generate a runnable-shaped microbenchmark stub (not actually runnable yet)."""
    return f'''"""Microbenchmark stub for {kernel_name}.

This is a *template* — input construction is not wired up yet. The next stage
of the experiment will fill in the launch grid + buffer allocation.
"""
import torch
import triton

SAMPLE_INPUTS = {json.dumps(sample_inputs, indent=2)}

# TODO: import kernel from kernel.py once the launcher is generated.
# from kernel import {kernel_name}

def make_inputs():
    raise NotImplementedError("Stage 3 will generate input tensors here.")

def reference(*args):
    raise NotImplementedError("Reference: load inductor's output_code.py and call its launcher.")

def candidate(*args):
    raise NotImplementedError("Candidate: agent-generated replacement.")
'''


def main():
    output_codes = _collect_output_codes()
    cache_kernels = _collect_cache_kernel_files()
    print(f"[match] found {len(output_codes)} output_code.py files")
    print(f"[match] found {len(cache_kernels)} cached kernel .py files")

    # Load rankings.
    with open(EXTRACT_DIR / "ranking_prefill.json") as f:
        rank_prefill = json.load(f)
    with open(EXTRACT_DIR / "ranking_decode.json") as f:
        rank_decode = json.load(f)

    # Map kernel name -> ranking entry per workload.
    def _to_map(rk):
        return {r["name"]: r for r in rk["ranking"]}

    rmap = {
        "prefill_512_b1": _to_map(rank_prefill),
        "decode_ctx512_b1": _to_map(rank_decode),
    }

    # All triton kernel names that appear in either ranking.
    triton_names = set()
    for m in rmap.values():
        for r in m.values():
            if r["kind"] == "triton":
                triton_names.add(r["name"])

    # All aten ops appearing in either ranking.
    aten_summary = {}
    for wl, m in rmap.items():
        for r in m.values():
            if r["kind"] == "aten":
                aten_summary.setdefault(r["name"], {})[wl] = {
                    "total_us": r["total_us"],
                    "pct_total": r["pct_total"],
                    "invocations": r["invocations"],
                    "mean_us": r["mean_us"],
                    "sample_inputs": r.get("sample_inputs", {}),
                }

    # Write aten_calls.json.
    with open(EXTRACT_DIR / "aten_calls.json", "w") as f:
        json.dump({
            "note": (
                "These ops were dispatched by inductor to cuBLAS/cuDNN/SDPA "
                "backends (NOT codegen'd by inductor itself). They are not "
                "candidates for agent replacement."
            ),
            "ops": aten_summary,
        }, f, indent=2)
    print(f"[match] wrote {EXTRACT_DIR/'aten_calls.json'} ({len(aten_summary)} ops)")

    # For every triton kernel we know about: try to locate its source and
    # write it under extract/kernels/{name}/.
    index = []
    missing: List[str] = []
    for name in sorted(triton_names):
        source, sources_in = _find_kernel_source(name, output_codes, cache_kernels)
        if source is None:
            missing.append(name)
            continue

        out_dir = KERNELS_OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kernel.py").write_text(source)

        meta = {
            "name": name,
            "found_in": sources_in,  # list of "workload::graph_id"
            "stats": {
                wl: rmap[wl][name] for wl in rmap if name in rmap[wl]
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        sample = next((r.get("sample_inputs", {}) for r in meta["stats"].values()), {})
        (out_dir / "microbench.py").write_text(_stub_microbench(name, sample))

        index.append({
            "name": name,
            "kernel_path": str((out_dir / "kernel.py").relative_to(EXTRACT_DIR)),
            "found_in": sources_in,
            "walltime_pct": {wl: rmap[wl][name]["pct_total"] for wl in rmap if name in rmap[wl]},
            "invocations": {wl: rmap[wl][name]["invocations"] for wl in rmap if name in rmap[wl]},
        })

    # Also enumerate all distinct fused-Triton kernel definitions across the 6
    # workloads (not just those that appear in the two profiled traces).
    all_emitted = set()
    per_workload_emitted: Dict[str, set] = {}
    for key, oc_path in output_codes.items():
        wl = key.split("::", 1)[0]
        try:
            text = oc_path.read_text()
        except Exception:
            continue
        per_workload_emitted.setdefault(wl, set())
        for m in _DEF_LINE_RE.finditer(text):
            all_emitted.add(m.group(1))
            per_workload_emitted[wl].add(m.group(1))

    summary = {
        "n_distinct_kernels_in_traces": len(triton_names),
        "n_distinct_kernels_emitted_across_6_workloads": len(all_emitted),
        "per_workload_emitted_counts": {wl: len(s) for wl, s in per_workload_emitted.items()},
        "per_workload_emitted_kernels": {wl: sorted(s) for wl, s in per_workload_emitted.items()},
        "all_emitted_kernels": sorted(all_emitted),
        "n_kernels_with_source": len(index),
        "n_kernels_missing_source": len(missing),
        "missing": missing,
        "index": index,
    }
    with open(EXTRACT_DIR / "kernels_index.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[match] wrote {EXTRACT_DIR/'kernels_index.json'} "
        f"({len(index)} kernel sources extracted, {len(missing)} missing)"
    )
    print(
        f"[match] inductor emitted {len(all_emitted)} distinct fused Triton kernel "
        f"definitions across the 6 workloads"
    )
    for wl, s in sorted(per_workload_emitted.items()):
        print(f"  {wl}: {len(s)} kernels")


if __name__ == "__main__":
    main()
