"""Regenerate the eager-derived pinned `last_token_ids` artifacts.

For every decode workload, load eager Qwen3-1.7B, run prefill on the workload's
prompt, take `argmax(last_position_logits)`, and save the tensor under
`baselines/results/eager_last_token_ids/<workload>.pt`.

The e2e orchestrator reads these so every config (eager / compile / patched)
decodes from THE SAME starting token — without this pinning, bf16 prefill
drift through patched kernels would flip the last-position argmax and surface
as a spurious `decode_ctx512_b1` correctness failure.

Usage:
    python -m tools.regenerate_pinned_tokens
    python -m tools.regenerate_pinned_tokens --workloads decode_ctx512_b1

This script does not run baselines/benchmarks — it ONLY regenerates the
pinned-token artifacts. Safe to run any time the workload registry or eager
model change.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workload.model import load_model, build_kv_cache  # noqa: E402
from workload.inputs import get_workload, list_workloads  # noqa: E402

PINNED_TOKEN_DIR = ROOT / "baselines" / "results" / "eager_last_token_ids"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", default=None,
                    help="Subset of decode workloads to regenerate.")
    args = ap.parse_args()

    PINNED_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    all_wls = args.workloads or list_workloads()
    decode_wls = [w for w in all_wls
                  if get_workload(w, pin_last_token=False)["mode"] == "decode"]
    if not decode_wls:
        print("[regen] no decode workloads selected; nothing to do.")
        return

    print(f"[regen] loading eager Qwen3-1.7B bf16 ...")
    t0 = time.time()
    model, _ = load_model(dtype=torch.bfloat16, device="cuda")
    print(f"[regen] loaded in {time.time() - t0:.1f}s")

    for name in decode_wls:
        print(f"[regen] {name} ...")
        wl = get_workload(name, pin_last_token=False)
        # Recreate the prompt-side prefill that build_kv_cache uses, but skip
        # the KV-cache return — we only want last_token_ids.
        prompt_ids = wl["prompt_ids"].to("cuda")
        _, last_token_ids, _ = build_kv_cache(model, prompt_ids)
        out_path = PINNED_TOKEN_DIR / f"{name}.pt"
        torch.save(last_token_ids.detach().to("cpu"), out_path)
        print(f"[regen]   saved {out_path}  shape={tuple(last_token_ids.shape)}")

    print("[regen] done.")


if __name__ == "__main__":
    main()
