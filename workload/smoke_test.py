"""Smoke test: imports + canonical inputs + a synthetic correctness check.

Does NOT load the Qwen3 weights — that's the next agent's job.
"""

from __future__ import annotations

import os
import sys

# Make sibling-module imports work when run as `python3 workload/smoke_test.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from workload import correctness, inputs, model  # noqa: E402,F401


def main() -> int:
    print(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}")
    print(f"available workloads: {inputs.list_workloads()}")

    w = inputs.get_workload("prefill_512_b1")
    print(
        f"prefill_512_b1: input_ids={tuple(w['input_ids'].shape)} "
        f"dtype={w['input_ids'].dtype} "
        f"attention_mask={tuple(w['attention_mask'].shape)} "
        f"mode={w['mode']} seq_len={w['seq_len']} batch_size={w['batch_size']}"
    )

    # Sanity check a decode workload's shapes (without actually building the cache).
    dw = inputs.get_workload("decode_ctx512_b8")
    print(
        f"decode_ctx512_b8: input_ids={tuple(dw['input_ids'].shape)} "
        f"prompt_ids={tuple(dw['prompt_ids'].shape)} "
        f"attention_mask={tuple(dw['attention_mask'].shape)} "
        f"mode={dw['mode']} seq_len={dw['seq_len']} batch_size={dw['batch_size']} "
        f"has_kv_cache_builder={dw['kv_cache_builder'] is not None}"
    )

    # Synthetic correctness check: two near-identical bf16 tensors should pass.
    torch.manual_seed(0)
    ref = torch.randn(4, 8, dtype=torch.float32)
    cand = ref + 0.001 * torch.randn_like(ref)
    result = correctness.check_outputs(ref, cand, dtype="bf16", task="standard")
    print(f"correctness (near-identical): {result}")

    # And a clearly-failing pair.
    bad = torch.randn_like(ref)
    result_bad = correctness.check_outputs(ref, bad, dtype="bf16", task="standard")
    print(f"correctness (random vs random): pass={result_bad['pass']} "
          f"cos_sim={result_bad['cos_sim']:.4f} reasons={result_bad['reasons']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
