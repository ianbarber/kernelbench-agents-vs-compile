"""Blinded code review of the 12 kernels (3 agent + 1 inductor x 3 tasks).

For each task:
    1. Read task.md + reference.py.
    2. Shuffle the 4 candidates and label A/B/C/D.
    3. Save the blinding map to review/blinding_map_{task}.json.
    4. Build a prompt and call `claude -p`.
    5. Save raw response to review/reviews/{task}.md.
    6. Write review/reviews/{task}_unblinded.md mapping labels -> authors.

The reviewer is the same model family as one of the writers (claude). This is
a known limitation we call out in the writeup.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/ianbarber/Projects/KernelBench")
REVIEW = ROOT / "review"
REVIEWS = REVIEW / "reviews"

TASKS = ["swiglu", "rmsnorm", "sdpa_prelude"]

# Map task -> (task_dir, sandbox_suffix, inductor_kernel_dir)
TASK_CONFIG = {
    "swiglu": {
        "task_dir": ROOT / "agent_loop/tasks/swiglu",
        "sandboxes": {
            "claude": ROOT / "agent_loop/sandbox/swiglu_claude_strict/candidate.py",
            "codex":  ROOT / "agent_loop/sandbox/swiglu_codex_strict/candidate.py",
            "kimi":   ROOT / "agent_loop/sandbox/swiglu_kimi_strict/candidate.py",
        },
        "inductor": ROOT / "extract/kernels/triton_poi_fused__unsafe_view_mul_silu_6/kernel.py",
        "inductor_note": "single emitted Triton kernel for SwiGLU (silu(x_gate) * x_up)",
    },
    "rmsnorm": {
        "task_dir": ROOT / "agent_loop/tasks/rmsnorm",
        "sandboxes": {
            "claude": ROOT / "agent_loop/sandbox/rmsnorm_claude_v1/candidate.py",
            "codex":  ROOT / "agent_loop/sandbox/rmsnorm_codex_v1/candidate.py",
            "kimi":   ROOT / "agent_loop/sandbox/rmsnorm_kimi_v1/candidate.py",
        },
        "inductor": ROOT / "extract/kernels/triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9/kernel.py",
        "inductor_note": "single emitted Triton kernel for residual-add + RMSNorm",
    },
    "sdpa_prelude": {
        "task_dir": ROOT / "agent_loop/tasks/sdpa_prelude",
        "sandboxes": {
            "claude": ROOT / "agent_loop/sandbox/sdpa_prelude_claude_v1/candidate.py",
            "codex":  ROOT / "agent_loop/sandbox/sdpa_prelude_codex_v1/candidate.py",
            "kimi":   ROOT / "agent_loop/sandbox/sdpa_prelude_kimi_v1/candidate.py",
        },
        # Use the dominant kernel (where_3, GQA expand, 25% of prefill walltime).
        "inductor": ROOT / "extract/kernels/triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_transpose_unsqueeze_view_where_3/kernel.py",
        "inductor_note": (
            "NOTE: inductor splits the SDPA prelude across ~6 kernels (3x cuBLAS mm, "
            "2x per-head RMSNorm+RoPE, 2x GQA-expand, 1x causal mask). The file shown "
            "is the single dominant Triton kernel (GQA expand, where_3), which is "
            "~25% of prefill walltime. The agent candidates are full single-file "
            "modules that fuse the whole prelude, so they will *look* much bigger -- "
            "this is structural, not bloat. Judge each on its own terms."
        ),
    },
}

# Seed for blinding -- deterministic across reruns, but not from built-in hash().
# (See feedback_cross_process_determinism note.)
def stable_seed(task: str) -> int:
    h = hashlib.sha256(f"kernelbench-review::{task}".encode()).digest()
    return int.from_bytes(h[:8], "big")


RUBRIC = """
**Rubric (1-5 scale, where 1=poor, 5=excellent):**
1. **Correctness reasoning** -- does the code make it easy to convince yourself the math is right? Are there obvious tolerance hazards (e.g. bf16 intermediate where fp32 is needed, missing fp32 promotion of accumulators)?
2. **Performance reasoning** -- block sizing, load coalescing, register pressure, choice of fusion boundaries. Can you predict if it's fast just from reading?
3. **Readability** -- variable naming, structure, comments where they earn their keep.
4. **Code length** -- right-sized for the job, or under/over-engineered?
5. **Risk** -- would you ship this to production? What's the failure mode you'd worry about?
""".strip()


def build_prompt(task: str, mapping: dict[str, str], kernels: dict[str, str]) -> str:
    cfg = TASK_CONFIG[task]
    task_md = (cfg["task_dir"] / "task.md").read_text()
    reference_py = (cfg["task_dir"] / "reference.py").read_text()
    inductor_note = cfg["inductor_note"]

    parts = [
        f"# Code review: {task} Triton kernels",
        "",
        "You are reviewing four candidate Triton kernel implementations of the same task. "
        "Three were written by AI coding agents; one was emitted by torch.inductor. "
        "**You do not know which is which.** Review them all on their merits.",
        "",
        f"Inductor context: {inductor_note}",
        "",
        "## Task description",
        "",
        task_md.strip(),
        "",
        "## Eager reference implementation",
        "",
        "```python",
        reference_py.strip(),
        "```",
        "",
        "## Candidate kernels",
        "",
    ]
    for label in ["A", "B", "C", "D"]:
        parts += [
            f"### KERNEL_{label}",
            "",
            "```python",
            kernels[label].rstrip(),
            "```",
            "",
        ]

    parts += [
        "## Your task",
        "",
        RUBRIC,
        "",
        "For each of KERNEL_A, KERNEL_B, KERNEL_C, KERNEL_D, give a score on each of the 5 rubric dimensions plus a 2-4 sentence rationale.",
        "",
        "Then answer one **forced-rank** question:",
        "> If you had to ship exactly ONE of A/B/C/D to production, which would it be, and why? Limit yourself to ~4 sentences.",
        "",
        "Format your response as markdown with sections `## KERNEL_A`, `## KERNEL_B`, `## KERNEL_C`, `## KERNEL_D`, then `## Forced rank: ship one`.",
        "Under each kernel, present the 5 rubric scores as a small markdown table (dimension | score | one-line note).",
    ]
    return "\n".join(parts)


def review_one_task(task: str) -> dict:
    cfg = TASK_CONFIG[task]
    rng = random.Random(stable_seed(task))

    authors = ["claude", "codex", "kimi", "inductor"]
    rng.shuffle(authors)
    labels = ["A", "B", "C", "D"]
    label_to_author = dict(zip(labels, authors))

    # Load the source for each
    kernels: dict[str, str] = {}
    for label, author in label_to_author.items():
        if author == "inductor":
            kernels[label] = cfg["inductor"].read_text()
        else:
            kernels[label] = cfg["sandboxes"][author].read_text()

    # Save blinding map
    (REVIEW / f"blinding_map_{task}.json").write_text(
        json.dumps({"task": task, "label_to_author": label_to_author}, indent=2)
    )

    prompt = build_prompt(task, label_to_author, kernels)
    prompt_path = REVIEWS / f"{task}_prompt.txt"
    prompt_path.write_text(prompt)

    print(f"[{task}] prompt: {len(prompt)} chars, calling claude -p ...", flush=True)
    proc = subprocess.run(
        ["claude", "-p", "--permission-mode", "bypassPermissions"],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        print(f"[{task}] claude FAILED rc={proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"claude review failed for {task}")
    response = proc.stdout

    # Save raw (blinded) review
    raw_path = REVIEWS / f"{task}.md"
    header = (
        f"# Blinded review: {task}\n\n"
        f"Reviewer: claude (CLI) -- same model family as one writer (known limitation).\n"
        f"Prompt: `review/reviews/{task}_prompt.txt`\n\n"
        "---\n\n"
    )
    raw_path.write_text(header + response.strip() + "\n")

    # Unblinded version: just append the mapping table
    unblinded = REVIEWS / f"{task}_unblinded.md"
    map_table = ["# Unblinded review key: " + task, "",
                 "| label | author |", "|---|---|"]
    for label, author in label_to_author.items():
        map_table.append(f"| KERNEL_{label} | {author} |")
    map_table.append("")
    map_table.append("---")
    map_table.append("")
    map_table.append("See `" + task + ".md` for the blinded review text. ")
    map_table.append("Use the mapping above to translate KERNEL_X -> author when reading.")
    unblinded.write_text("\n".join(map_table) + "\n")

    return {"task": task, "label_to_author": label_to_author, "response_chars": len(response)}


def main() -> None:
    REVIEWS.mkdir(parents=True, exist_ok=True)
    out = []
    for task in TASKS:
        out.append(review_one_task(task))
    (REVIEW / "reviewer_log.json").write_text(json.dumps(out, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
