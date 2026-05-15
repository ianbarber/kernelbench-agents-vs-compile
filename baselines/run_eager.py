"""Eager-mode baseline benchmarks for Qwen3-1.7B on all canonical workloads.

For each workload:
  - load model once (bf16)
  - prepare inputs (prefill) or build KV cache (decode)
  - warmup 25 + measure 100 with `triton.testing.do_bench` (median latency)
  - capture peak GPU memory, p10/p90, tokens/sec
  - save reference logits for later correctness checks
  - capture a profiler trace for one prefill + one decode workload

Outputs:
  baselines/results/eager.json
  baselines/results/reference_outputs/{workload_name}.pt
  baselines/results/traces/eager_{workload}.json

Notes:
  - nvidia-smi memory metrics are broken on GB10; use torch.cuda.* metrics.
  - The cached forward (decode) requires the full attention mask covering
    past_seen + 1 positions; we follow the convention in workload/model.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import triton.testing

# Make project root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workload.model import load_model, prefill_fn, decode_fn  # noqa: E402
from workload.inputs import get_workload, list_workloads  # noqa: E402

RESULTS_DIR = ROOT / "baselines" / "results"
REF_DIR = RESULTS_DIR / "reference_outputs"
TRACE_DIR = RESULTS_DIR / "traces"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 25
REP = 100

# Workloads we profile (one prefill + one decode).
PROFILE_PREFILL = "prefill_512_b1"
PROFILE_DECODE = "decode_ctx512_b1"


def _build_runner(model, workload):
    """Return (runner_callable, get_output_callable).

    `runner_callable()` runs one full forward pass and returns nothing (for
    do_bench). `get_output_callable()` runs once and returns the logits we
    want to save as reference. Both share the prepared inputs/state.
    """
    device = next(model.parameters()).device
    mode = workload["mode"]

    if mode == "prefill":
        input_ids = workload["input_ids"].to(device)
        attn = workload["attention_mask"].to(device)

        def run():
            return prefill_fn(model, input_ids, attn)

        return run, run

    elif mode == "decode":
        # Build KV cache fresh.
        kv_state = workload["kv_cache_builder"](model)
        past = kv_state["past_key_values"]
        last_token_ids = kv_state["last_token_ids"]
        attn = kv_state["attention_mask"]

        # IMPORTANT: a real do_bench loop will call this many times. Each call
        # appends to the cache, which (a) blows up the cache and (b) changes
        # the attention mask shape. We need to keep the cache *fixed* at the
        # post-prefill state across timed iterations. We do that by snapshotting
        # the cache's tensors once and restoring before each call.
        from transformers import DynamicCache

        # Snapshot. DynamicCache stores key/value caches in `.layers` (newer)
        # or `.key_cache` / `.value_cache` (older). Be robust.
        def snapshot(cache):
            if hasattr(cache, "layers") and cache.layers is not None:
                # Newer API: list of CacheLayer with .keys, .values
                snaps = []
                for layer in cache.layers:
                    snaps.append((
                        layer.keys.clone() if layer.keys is not None else None,
                        layer.values.clone() if layer.values is not None else None,
                    ))
                return ("layers", snaps)
            elif hasattr(cache, "key_cache"):
                return (
                    "legacy",
                    [
                        (k.clone(), v.clone())
                        for k, v in zip(cache.key_cache, cache.value_cache)
                    ],
                )
            else:
                raise RuntimeError("Unknown DynamicCache layout")

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

        return run, run
    else:
        raise ValueError(f"unknown mode {mode}")


def _measure(run_fn):
    """Time `run_fn` with do_bench. Returns dict of stats in ms."""
    # do_bench reports in ms; quantiles for p10/p90, median for headline.
    # Reset peak memory before measuring.
    torch.cuda.reset_peak_memory_stats()
    median_ms = triton.testing.do_bench(
        run_fn, warmup=WARMUP, rep=REP, return_mode="median"
    )
    # Also gather p10/p90 from a second sweep — do_bench's quantile return is
    # available via return_mode="all" in newer triton; if not, fall back to
    # repeated medians. We'll do a single pass with all=True; older versions
    # return a tensor of per-iter timings.
    try:
        all_ms = triton.testing.do_bench(
            run_fn, warmup=WARMUP, rep=REP, return_mode="all"
        )
        if hasattr(all_ms, "tolist"):
            all_ms = all_ms.tolist()
        if isinstance(all_ms, (list, tuple)) and len(all_ms) >= 2:
            xs = sorted(all_ms)
            p10 = xs[max(0, int(0.10 * len(xs)) - 1)]
            p90 = xs[min(len(xs) - 1, int(0.90 * len(xs)) - 1)]
        else:
            p10 = p90 = float(median_ms)
    except Exception:
        p10 = p90 = float(median_ms)

    peak_bytes = torch.cuda.max_memory_allocated()
    return {
        "median_ms": float(median_ms),
        "p10_ms": float(p10),
        "p90_ms": float(p90),
        "peak_mem_bytes": int(peak_bytes),
    }


def _tokens_per_sec(workload, median_ms):
    if workload["mode"] == "prefill":
        toks = workload["seq_len"] * workload["batch_size"]
    else:
        toks = workload["batch_size"]
    return toks / (median_ms / 1000.0)


def _profile_one(run_fn, out_path: Path):
    """Run a profiler trace for ~10 iters of run_fn, save Chrome JSON."""
    from torch.profiler import profile, ProfilerActivity

    # Warm up first to avoid capturing lazy init.
    for _ in range(5):
        run_fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        with_flops=False,
    ) as prof:
        for _ in range(10):
            run_fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_path))


def main():
    print(f"[eager] Loading Qwen3-1.7B bf16 on cuda...")
    t_load = time.time()
    model, tokenizer = load_model(dtype=torch.bfloat16, device="cuda")
    print(f"[eager] Loaded in {time.time() - t_load:.1f}s")

    results = {}
    workloads = list_workloads()
    for name in workloads:
        print(f"\n[eager] === Workload: {name} ===")
        try:
            wl = get_workload(name)
            run, ref_run = _build_runner(model, wl)

            # 1) reference output (single deterministic call).
            with torch.no_grad():
                ref_logits = ref_run()
                # For decode, save next-token logits (full output is [B,1,V]).
                # For prefill, save full logits [B,S,V]. (decode_fn already
                # returns [B,1,V] so just save as-is.)
                ref_to_save = ref_logits.detach().to("cpu")
            torch.save(ref_to_save, REF_DIR / f"{name}.pt")
            print(f"[eager]   saved reference {tuple(ref_to_save.shape)}")

            # 2) measure.
            stats = _measure(run)
            stats["tokens_per_sec"] = _tokens_per_sec(wl, stats["median_ms"])
            stats["mode"] = wl["mode"]
            stats["batch_size"] = wl["batch_size"]
            stats["seq_len"] = wl["seq_len"]
            results[name] = stats
            print(
                f"[eager]   median={stats['median_ms']:.3f} ms  "
                f"p10={stats['p10_ms']:.3f}  p90={stats['p90_ms']:.3f}  "
                f"peak_mem={stats['peak_mem_bytes']/2**20:.1f} MiB  "
                f"tok/s={stats['tokens_per_sec']:.1f}"
            )

            # 3) profile selected workloads.
            if name in (PROFILE_PREFILL, PROFILE_DECODE):
                print(f"[eager]   capturing profiler trace ...")
                _profile_one(run, TRACE_DIR / f"eager_{name}.json")

            # Free decode KV cache between workloads.
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[eager] FAILURE on {name}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e), "traceback": traceback.format_exc()}

    with open(RESULTS_DIR / "eager.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eager] wrote {RESULTS_DIR / 'eager.json'}")


if __name__ == "__main__":
    main()
