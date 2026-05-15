"""Generate e2e/summary.md from results/*.json.

Reads per-config JSONs from e2e/results/ plus the existing
baselines/results/compile_default.json, builds a per-workload comparison
table with latency, speedup vs eager, correctness, and a narrative.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E2E_RESULTS = ROOT / "e2e" / "results"
BASELINE_RESULTS = ROOT / "baselines" / "results"

CONFIGS = [
    ("eager", "eager"),
    ("eager_swiglu_kimi", "+swiglu (kimi)"),
    ("eager_rmsnorm_pure", "+rmsnorm-pure (claude)"),
    ("eager_both_pure", "+swiglu +rmsnorm-pure"),
    ("eager_swiglu_rmsnorm_fused", "+swiglu +rmsnorm-fused"),
    ("compile_default", "torch.compile (default)"),
]

WORKLOADS = [
    "prefill_512_b1",
    "prefill_2048_b1",
    "decode_ctx512_b1",
    "decode_ctx512_b8",
    "decode_ctx2048_b1",
    "decode_ctx2048_b8",
]


def _load(cfg: str) -> dict:
    if cfg == "compile_default":
        p = BASELINE_RESULTS / "compile_default.json"
    elif cfg == "eager":
        # Prefer e2e's eager (run on same code path) but fall back to baseline.
        p = E2E_RESULTS / "eager.json"
        if not p.exists():
            p = BASELINE_RESULTS / "eager.json"
    else:
        p = E2E_RESULTS / f"{cfg}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _fmt_ms(v):
    if v is None:
        return "—"
    return f"{v:.2f}"


def _fmt_speedup(base, val):
    if base is None or val is None or val == 0:
        return "—"
    return f"{base / val:.3f}×"


def _corr(entry):
    if not entry or "correctness" not in entry:
        return ""
    c = entry["correctness"]
    if c.get("pass"):
        if c.get("strict_pass"):
            return "PASS (strict)"
        return f"PASS (std, cos={c['cos_sim']:.5f})"
    return f"FAIL: {c.get('reasons', ['?'])[0]}"


def main():
    all_data = {cfg: _load(cfg) for cfg, _ in CONFIGS}
    lines = []
    lines.append("# End-to-end Qwen3-1.7B with Agent Kernels")
    lines.append("")
    lines.append(
        "Comparing eager / patched (swiglu-kimi, rmsnorm-claude pure & fused) / "
        "`torch.compile` on the canonical 6 workloads. Bench: "
        "`triton.testing.do_bench` warmup 25, rep 100, median ms. "
        "Correctness: `task=standard` (cos_sim ≥ 0.95, l1_rel ≤ 0.05, rmse ≤ 0.10) "
        "vs eager reference logits."
    )
    lines.append("")

    # Per-workload tables.
    for wl in WORKLOADS:
        lines.append(f"## {wl}")
        lines.append("")
        lines.append("| Config | median (ms) | p10 | p90 | tok/s | peak MiB | speedup vs eager | correctness |")
        lines.append("|---|---|---|---|---|---|---|---|")
        eager_med = (all_data.get("eager", {}).get(wl) or {}).get("median_ms")
        for cfg, label in CONFIGS:
            entry = (all_data.get(cfg) or {}).get(wl) or {}
            med = entry.get("median_ms")
            p10 = entry.get("p10_ms")
            p90 = entry.get("p90_ms")
            tps = entry.get("tokens_per_sec")
            mem = entry.get("peak_mem_bytes")
            speedup = _fmt_speedup(eager_med, med)
            corr = _corr(entry)
            tps_s = f"{tps:.1f}" if isinstance(tps, (int, float)) else "—"
            mem_s = f"{mem/2**20:.0f}" if isinstance(mem, (int, float)) else "—"
            lines.append(
                f"| {label} | {_fmt_ms(med)} | {_fmt_ms(p10)} | {_fmt_ms(p90)} | "
                f"{tps_s} | {mem_s} | {speedup} | {corr} |"
            )
        lines.append("")

    # Headline summary across workloads.
    lines.append("## Headline: speedup vs eager (median × across all workloads)")
    lines.append("")
    lines.append("| Config | geomean speedup | min | max |")
    lines.append("|---|---|---|---|")
    import math
    for cfg, label in CONFIGS:
        ratios = []
        for wl in WORKLOADS:
            entry = (all_data.get(cfg) or {}).get(wl) or {}
            med = entry.get("median_ms")
            eager_med = (all_data.get("eager", {}).get(wl) or {}).get("median_ms")
            if eager_med and med:
                ratios.append(eager_med / med)
        if ratios:
            gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            lines.append(f"| {label} | {gm:.3f}× | {min(ratios):.3f}× | {max(ratios):.3f}× |")
        else:
            lines.append(f"| {label} | — | — | — |")
    lines.append("")

    # Narrative.
    lines.append("## Narrative")
    lines.append("")

    def per_workload_diff(cfg_a, cfg_b):
        out = {}
        for wl in WORKLOADS:
            ma = (all_data.get(cfg_a, {}).get(wl) or {}).get("median_ms")
            mb = (all_data.get(cfg_b, {}).get(wl) or {}).get("median_ms")
            if ma and mb:
                out[wl] = (ma, mb, ma / mb)
        return out

    lines.append("### SwiGLU (1.06× standalone microbench)")
    diffs = per_workload_diff("eager", "eager_swiglu_kimi")
    if diffs:
        for wl, (ma, mb, r) in diffs.items():
            lines.append(f"- {wl}: eager {ma:.2f} → patched {mb:.2f} ms  ({r:.3f}× faster)")
    lines.append("")
    lines.append("### RMSNorm pure (1.17× standalone, but pure variant has zero residual = wasted fusion)")
    diffs = per_workload_diff("eager", "eager_rmsnorm_pure")
    if diffs:
        for wl, (ma, mb, r) in diffs.items():
            lines.append(f"- {wl}: eager {ma:.2f} → patched {mb:.2f} ms  ({r:.3f}× faster)")
    lines.append("")
    lines.append("### Both pure")
    diffs = per_workload_diff("eager", "eager_both_pure")
    if diffs:
        for wl, (ma, mb, r) in diffs.items():
            lines.append(f"- {wl}: eager {ma:.2f} → patched {mb:.2f} ms  ({r:.3f}× faster)")
    lines.append("")
    lines.append("### SwiGLU + RMSNorm fused (proper residual fusion in DecoderLayer)")
    diffs = per_workload_diff("eager", "eager_swiglu_rmsnorm_fused")
    if diffs:
        for wl, (ma, mb, r) in diffs.items():
            lines.append(f"- {wl}: eager {ma:.2f} → patched {mb:.2f} ms  ({r:.3f}× faster)")
    lines.append("")
    lines.append("### torch.compile (reference for what \"smart fusion\" gives you)")
    diffs = per_workload_diff("eager", "compile_default")
    if diffs:
        for wl, (ma, mb, r) in diffs.items():
            lines.append(f"- {wl}: eager {ma:.2f} → compile {mb:.2f} ms  ({r:.3f}× faster)")
    lines.append("")

    summary_path = ROOT / "e2e" / "summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
