"""Mechanical stats for the 10 kernels in the code-review stage.

Computes LoC (excluding blank + comment-only), bytes, cyclomatic complexity
(via radon), maintainability index, and rough counts of Triton ops.

Outputs:
    review/stats.json
    review/stats.md
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/ianbarber/Projects/KernelBench")
RADON = ROOT / ".venv" / "bin" / "radon"
REVIEW = ROOT / "review"

# (task, author, path)
KERNELS: list[tuple[str, str, Path]] = [
    ("swiglu", "claude",   ROOT / "agent_loop/sandbox/swiglu_claude_strict/candidate.py"),
    ("swiglu", "codex",    ROOT / "agent_loop/sandbox/swiglu_codex_strict/candidate.py"),
    ("swiglu", "kimi",     ROOT / "agent_loop/sandbox/swiglu_kimi_strict/candidate.py"),
    ("swiglu", "inductor", ROOT / "extract/kernels/triton_poi_fused__unsafe_view_mul_silu_6/kernel.py"),

    ("rmsnorm", "claude",   ROOT / "agent_loop/sandbox/rmsnorm_claude_v1/candidate.py"),
    ("rmsnorm", "codex",    ROOT / "agent_loop/sandbox/rmsnorm_codex_v1/candidate.py"),
    ("rmsnorm", "kimi",     ROOT / "agent_loop/sandbox/rmsnorm_kimi_v1/candidate.py"),
    ("rmsnorm", "inductor", ROOT / "extract/kernels/triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9/kernel.py"),

    ("sdpa_prelude", "claude",   ROOT / "agent_loop/sandbox/sdpa_prelude_claude_v1/candidate.py"),
    ("sdpa_prelude", "codex",    ROOT / "agent_loop/sandbox/sdpa_prelude_codex_v1/candidate.py"),
    ("sdpa_prelude", "kimi",     ROOT / "agent_loop/sandbox/sdpa_prelude_kimi_v1/candidate.py"),
    # SDPA prelude: representative dominant kernel (where_3 = GQA expand, 25% of prefill walltime).
    # Inductor's full SDPA prelude is split across many kernels -- we note this in the writeup.
    ("sdpa_prelude", "inductor",
     ROOT / "extract/kernels/triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_transpose_unsqueeze_view_where_3/kernel.py"),
]

# Measured candidate_us values (from agent_loop/runs/{run_id}/result.json).
# We extract these for the summary table but compute fresh here for self-containment.
RUN_IDS = {
    ("swiglu", "claude"):  "swiglu_claude_strict",
    ("swiglu", "codex"):   "swiglu_codex_strict",
    ("swiglu", "kimi"):    "swiglu_kimi_strict",
    ("rmsnorm", "claude"): "rmsnorm_claude_v1",
    ("rmsnorm", "codex"):  "rmsnorm_codex_v1",
    ("rmsnorm", "kimi"):   "rmsnorm_kimi_v1",
    ("sdpa_prelude", "claude"): "sdpa_prelude_claude_v1",
    ("sdpa_prelude", "codex"):  "sdpa_prelude_codex_v1",
    ("sdpa_prelude", "kimi"):   "sdpa_prelude_kimi_v1",
}

# Inductor baselines (from sandbox/*/inductor_baseline_us.json -- same values throughout)
INDUCTOR_US = {
    "swiglu": 109.58,
    "rmsnorm": 35.84,
    "sdpa_prelude": 4045.73,  # full prelude, not just where_3
}


def count_loc(text: str) -> int:
    """Count non-blank, non-comment-only lines."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        n += 1
    return n


def count_triton_ops(text: str) -> dict[str, int]:
    return {
        "triton_jit_fns": len(re.findall(r"@triton\.jit", text)),
        "tl_load": len(re.findall(r"\btl\.load\b", text)),
        "tl_store": len(re.findall(r"\btl\.store\b", text)),
        "tl_where": len(re.findall(r"\btl\.where\b", text)),
        "tl_sum": len(re.findall(r"\btl\.sum\b", text)),
        "tl_dot": len(re.findall(r"\btl\.dot\b", text)),
    }


def radon_cc(path: Path) -> dict:
    """Run `radon cc -j` and return total + max complexity for the file."""
    try:
        out = subprocess.check_output([str(RADON), "cc", "-j", str(path)], text=True)
        data = json.loads(out)
        items = data.get(str(path), [])
        if not items:
            return {"total": 0, "max": 0, "n_blocks": 0}
        comps = [i["complexity"] for i in items]
        return {"total": sum(comps), "max": max(comps), "n_blocks": len(comps)}
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "total": None, "max": None, "n_blocks": 0}


def radon_mi(path: Path) -> float | None:
    """Run `radon mi -j` and return the maintainability index (0-100)."""
    try:
        out = subprocess.check_output([str(RADON), "mi", "-j", "-s", str(path)], text=True)
        data = json.loads(out)
        v = data.get(str(path), {})
        return v.get("mi")
    except subprocess.CalledProcessError:
        return None


def get_candidate_us(task: str, author: str) -> float | None:
    if author == "inductor":
        return INDUCTOR_US.get(task)
    run_id = RUN_IDS.get((task, author))
    if run_id is None:
        return None
    p = ROOT / "agent_loop/runs" / run_id / "result.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("harness_result", {}).get("candidate_us")


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for task, author, path in KERNELS:
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            continue
        text = path.read_text()
        key = f"{task}/{author}"
        cc = radon_cc(path)
        mi = radon_mi(path)
        ops = count_triton_ops(text)
        candidate_us = get_candidate_us(task, author)
        inductor_us = INDUCTOR_US[task]
        speedup = (inductor_us / candidate_us) if candidate_us else None
        results[key] = {
            "task": task,
            "author": author,
            "path": str(path),
            "loc": count_loc(text),
            "bytes": len(text.encode("utf-8")),
            "raw_lines": len(text.splitlines()),
            "cc": cc,
            "mi": mi,
            "ops": ops,
            "candidate_us": candidate_us,
            "speedup_vs_inductor": speedup,
        }

    (REVIEW / "stats.json").write_text(json.dumps(results, indent=2))

    # ------- Markdown report -------
    md: list[str] = ["# Mechanical stats: 10 kernels\n",
                     "LoC = non-blank, non-comment-only. CC = cyclomatic complexity (sum / max). ",
                     "MI = maintainability index (0-100, higher better). bench is candidate_us median.\n"]
    for task in ("swiglu", "rmsnorm", "sdpa_prelude"):
        md.append(f"\n## {task}\n")
        md.append("| author | LoC | bytes | CC (sum/max/blocks) | MI | jit fns | loads | stores | where | sum | dot | bench (us) | speedup vs inductor |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for author in ("claude", "codex", "kimi", "inductor"):
            key = f"{task}/{author}"
            if key not in results:
                continue
            r = results[key]
            cc = r["cc"]
            o = r["ops"]
            us = r["candidate_us"]
            sp = r["speedup_vs_inductor"]
            us_s = f"{us:.1f}" if us is not None else "-"
            sp_s = f"{sp:.2f}x" if sp is not None else "-"
            mi_s = f"{r['mi']:.1f}" if r["mi"] is not None else "-"
            md.append(
                f"| {author} | {r['loc']} | {r['bytes']} | "
                f"{cc.get('total','-')}/{cc.get('max','-')}/{cc.get('n_blocks','-')} | "
                f"{mi_s} | {o['triton_jit_fns']} | {o['tl_load']} | {o['tl_store']} | "
                f"{o['tl_where']} | {o['tl_sum']} | {o['tl_dot']} | {us_s} | {sp_s} |"
            )

    md.append("\n---\n")
    md.append("Notes:")
    md.append("- Inductor SDPA prelude is split across ~6 kernels (3 mm + 2 norm/RoPE + 2 GQA-expand + 1 mask). ")
    md.append("  We show the dominant single kernel here (`_where_3`, the GQA expand, 25% of prefill walltime), ")
    md.append("  but the speedup column compares against the *full* prelude microbench (4045.73 us).")
    md.append("- Agent SDPA candidates are single-file modules: they include host setup + multiple `@triton.jit` ")
    md.append("  functions, so direct LoC/CC comparison to a single inductor kernel is unfair on both sides. ")
    md.append("  Read the agent vs inductor SDPA rows as a documented mismatch, not apples-to-apples.")
    md.append("- SwiGLU / RMSNorm comparisons are fair: 1 kernel per file, similar boilerplate.")

    (REVIEW / "stats.md").write_text("\n".join(md) + "\n")
    print(f"wrote {REVIEW/'stats.json'}")
    print(f"wrote {REVIEW/'stats.md'}")


if __name__ == "__main__":
    main()
