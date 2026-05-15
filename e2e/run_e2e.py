"""End-to-end Qwen3-1.7B bench with agent-kernel patches.

Compares eager / patched / patched-fused / torch.compile across the canonical
workload set. For each config:
  - fresh model load
  - install patches
  - correctness vs baselines/results/reference_outputs/<workload>.pt
  - triton.testing.do_bench (warmup 25 × rep 100, median)
  - peak GPU mem
  - tokens/sec
Writes one JSON per config under e2e/results/.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import triton.testing

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workload.model import load_model, prefill_fn, decode_fn  # noqa: E402
from workload.inputs import get_workload, list_workloads  # noqa: E402
from workload.correctness import check_outputs  # noqa: E402

from e2e import patches as P  # noqa: E402

RESULTS_DIR = ROOT / "e2e" / "results"
REF_DIR = ROOT / "baselines" / "results" / "reference_outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 25
REP = 100


# Reuse build_runner pattern from baselines/run_eager.py.
def _build_runner(model, workload):
    device = next(model.parameters()).device
    mode = workload["mode"]
    if mode == "prefill":
        input_ids = workload["input_ids"].to(device)
        attn = workload["attention_mask"].to(device)

        def run():
            return prefill_fn(model, input_ids, attn)
        return run

    elif mode == "decode":
        kv_state = workload["kv_cache_builder"](model)
        past = kv_state["past_key_values"]
        last_token_ids = kv_state["last_token_ids"]
        attn = kv_state["attention_mask"]

        def snapshot(cache):
            if hasattr(cache, "layers") and cache.layers is not None:
                return ("layers", [(l.keys.clone() if l.keys is not None else None,
                                     l.values.clone() if l.values is not None else None)
                                    for l in cache.layers])
            elif hasattr(cache, "key_cache"):
                return ("legacy", [(k.clone(), v.clone()) for k, v in zip(cache.key_cache, cache.value_cache)])
            else:
                raise RuntimeError("unknown cache layout")

        def restore(cache, snap):
            kind, data = snap
            if kind == "layers":
                for layer, (k, v) in zip(cache.layers, data):
                    layer.keys = None if k is None else k.clone()
                    layer.values = None if v is None else v.clone()
            else:
                cache.key_cache = [k.clone() for k, _ in data]
                cache.value_cache = [v.clone() for _, v in data]

        snap = snapshot(past)

        def run():
            restore(past, snap)
            logits, _ = decode_fn(model, past, last_token_ids, attn)
            return logits
        return run
    else:
        raise ValueError(mode)


def _measure(run_fn):
    torch.cuda.reset_peak_memory_stats()
    median = triton.testing.do_bench(run_fn, warmup=WARMUP, rep=REP, return_mode="median")
    try:
        all_ms = triton.testing.do_bench(run_fn, warmup=WARMUP, rep=REP, return_mode="all")
        if hasattr(all_ms, "tolist"):
            all_ms = all_ms.tolist()
        xs = sorted(all_ms)
        p10 = xs[max(0, int(0.10 * len(xs)) - 1)]
        p90 = xs[min(len(xs) - 1, int(0.90 * len(xs)) - 1)]
    except Exception:
        p10 = p90 = float(median)
    return {
        "median_ms": float(median),
        "p10_ms": float(p10),
        "p90_ms": float(p90),
        "peak_mem_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _tokens_per_sec(wl, median_ms):
    toks = wl["seq_len"] * wl["batch_size"] if wl["mode"] == "prefill" else wl["batch_size"]
    return toks / (median_ms / 1000.0)


def _install_for_config(model, cfg: str):
    """Install patches per named config. Always start clean (uninstall first)."""
    P.uninstall(model)
    if cfg == "eager":
        pass
    elif cfg == "eager_swiglu_kimi":
        P.install_swiglu_kimi(model)
    elif cfg == "eager_rmsnorm_pure":
        P.install_rmsnorm_claude_pure(model)
    elif cfg == "eager_both_pure":
        P.install_swiglu_kimi(model)
        P.install_rmsnorm_claude_pure(model)
    elif cfg == "eager_swiglu_rmsnorm_fused":
        P.install_swiglu_kimi(model)
        P.install_rmsnorm_claude_fused(model)
    elif cfg == "eager_sdpa_prelude_kimi":
        P.install_sdpa_prelude_kimi(model)
    elif cfg == "eager_all_winners":
        # All three best agent kernels stacked.
        P.install_swiglu_kimi(model)
        P.install_rmsnorm_claude_pure(model)
        P.install_sdpa_prelude_kimi(model)
    else:
        raise ValueError(cfg)


def _run_config(cfg: str, workloads):
    print(f"\n========== Config: {cfg} ==========")
    # Fresh model.
    t0 = time.time()
    model, _ = load_model(dtype=torch.bfloat16, device="cuda")
    print(f"[{cfg}] loaded model in {time.time() - t0:.1f}s")
    _install_for_config(model, cfg)

    results = {}
    for name in workloads:
        print(f"\n[{cfg}] === {name} ===")
        try:
            wl = get_workload(name)
            run = _build_runner(model, wl)
            # Correctness vs reference.
            ref = torch.load(REF_DIR / f"{name}.pt", map_location="cuda", weights_only=False)
            with torch.no_grad():
                out = run()
            # Use "standard" tolerance (matches baselines/run_compile.py) — strict
            # is unrealistic for end-to-end accumulation across 28 layers × ~3
            # RMSNorm per layer (and through the prefill->KV-cache pipe).
            corr = check_outputs(ref, out, dtype="bf16", task="standard")
            corr_strict = check_outputs(ref, out, dtype="bf16", task="strict")
            if not corr["pass"]:
                print(f"[{cfg}]   CORRECTNESS FAIL (standard): {corr['reasons']}")
            else:
                print(f"[{cfg}]   correctness ok (standard) cos={corr['cos_sim']:.5f} l1={corr['l1_rel']:.4f} rmse={corr['rmse']:.4f}  strict_pass={corr_strict['pass']}")

            stats = _measure(run)
            stats["tokens_per_sec"] = _tokens_per_sec(wl, stats["median_ms"])
            stats["mode"] = wl["mode"]
            stats["batch_size"] = wl["batch_size"]
            stats["seq_len"] = wl["seq_len"]
            stats["correctness"] = {
                "pass": corr["pass"],
                "cos_sim": corr["cos_sim"],
                "l1_rel": corr["l1_rel"],
                "rmse": corr["rmse"],
                "reasons": corr["reasons"],
                "strict_pass": corr_strict["pass"],
                "strict_reasons": corr_strict["reasons"],
            }
            results[name] = stats
            print(f"[{cfg}]   median={stats['median_ms']:.3f}ms p10={stats['p10_ms']:.3f} p90={stats['p90_ms']:.3f} peak={stats['peak_mem_bytes']/2**20:.1f}MiB tok/s={stats['tokens_per_sec']:.1f}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{cfg}] FAILURE on {name}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e), "traceback": traceback.format_exc()}

    out_path = RESULTS_DIR / f"{cfg}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{cfg}] wrote {out_path}")

    # Free model.
    P.uninstall(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Subset of configs to run.")
    ap.add_argument("--workloads", nargs="*", default=None,
                    help="Subset of workloads.")
    args = ap.parse_args()

    workloads = args.workloads or list_workloads()
    all_configs = [
        "eager",
        "eager_swiglu_kimi",
        "eager_rmsnorm_pure",
        "eager_both_pure",
        "eager_swiglu_rmsnorm_fused",
        "eager_sdpa_prelude_kimi",
        "eager_all_winners",
    ]
    configs = args.configs or all_configs

    for cfg in configs:
        _run_config(cfg, workloads)


if __name__ == "__main__":
    main()
