"""torch.compile baseline for Qwen3-1.7B on all canonical workloads.

Three modes:
  A) "default"        — inductor default, cudagraphs OFF  (primary)
  B) "reduce-overhead"— inductor + cudagraphs ON
  C) "max-autotune"   — autotuned codegen

For each mode x each workload:
  - load model fresh (re-instantiate per mode)
  - wrap with torch.compile(..., dynamic=False)
  - time the FIRST forward separately as compile-time (seconds)
  - check the compiled output vs eager reference (correctness.check_outputs)
  - do_bench: 25 warmup, 100 measure
  - capture profiler trace for one prefill + one decode per mode

Outputs:
  baselines/results/compile_{mode}.json
  baselines/results/traces/{mode}_{workload}.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple

import torch
import triton.testing

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workload.model import load_model, prefill_fn, decode_fn  # noqa: E402
from workload.inputs import get_workload, list_workloads  # noqa: E402
from workload.correctness import check_outputs  # noqa: E402

RESULTS_DIR = ROOT / "baselines" / "results"
REF_DIR = RESULTS_DIR / "reference_outputs"
TRACE_DIR = RESULTS_DIR / "traces"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 25
REP = 100

PROFILE_PREFILL = "prefill_512_b1"
PROFILE_DECODE = "decode_ctx512_b1"

# Mode key -> (label-for-files, torch.compile kwargs, pre-compile hook fn).
def _set_cudagraphs(on: bool):
    import torch._inductor.config as ic
    ic.triton.cudagraphs = on

MODES = {
    "default": {
        "label": "default",
        "kwargs": {"mode": "default", "dynamic": False},
        "pre": lambda: _set_cudagraphs(False),
    },
    "cudagraphs": {
        "label": "cudagraphs",
        "kwargs": {"mode": "reduce-overhead", "dynamic": False},
        "pre": lambda: _set_cudagraphs(True),
    },
    "max_autotune": {
        "label": "max_autotune",
        "kwargs": {"mode": "max-autotune", "dynamic": False},
        "pre": lambda: _set_cudagraphs(True),  # max-autotune enables cgraphs
    },
}


def _snapshot_restore_factory(cache):
    """Return (snapshot, restore) bound to a specific DynamicCache instance."""

    def snapshot():
        if hasattr(cache, "layers") and cache.layers is not None:
            return ("layers", [
                (
                    layer.keys.clone() if layer.keys is not None else None,
                    layer.values.clone() if layer.values is not None else None,
                )
                for layer in cache.layers
            ])
        elif hasattr(cache, "key_cache"):
            return ("legacy", [
                (k.clone(), v.clone())
                for k, v in zip(cache.key_cache, cache.value_cache)
            ])
        raise RuntimeError("Unknown DynamicCache layout")

    def restore(snap):
        kind, data = snap
        if kind == "layers":
            for layer, (k, v) in zip(cache.layers, data):
                layer.keys = None if k is None else k.clone()
                layer.values = None if v is None else v.clone()
        else:
            cache.key_cache = [k.clone() for k, _ in data]
            cache.value_cache = [v.clone() for _, v in data]

    return snapshot, restore


def _build_runner(model, workload):
    """Mirrors run_eager._build_runner but uses the *compiled* model wrapper.

    We call model(...) directly so torch.compile sees a stable graph (HF's
    forward signature). prefill_fn/decode_fn are thin wrappers around model().
    """
    device = next(model.parameters()).device
    mode = workload["mode"]

    if mode == "prefill":
        input_ids = workload["input_ids"].to(device)
        attn = workload["attention_mask"].to(device)

        def run():
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            return out.logits

        return run

    elif mode == "decode":
        kv_state = workload["kv_cache_builder"](model)
        past = kv_state["past_key_values"]
        last_token_ids = kv_state["last_token_ids"]
        attn = kv_state["attention_mask"]

        snapshot, restore = _snapshot_restore_factory(past)
        snap = snapshot()

        def run():
            restore(snap)
            with torch.no_grad():
                out = model(
                    input_ids=last_token_ids,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
            return out.logits

        return run
    else:
        raise ValueError(f"unknown mode {mode}")


def _measure(run_fn):
    torch.cuda.reset_peak_memory_stats()
    median_ms = triton.testing.do_bench(
        run_fn, warmup=WARMUP, rep=REP, return_mode="median"
    )
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
    from torch.profiler import profile, ProfilerActivity

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


def _load_reference(name: str) -> torch.Tensor:
    return torch.load(REF_DIR / f"{name}.pt", map_location="cpu")


def run_mode(mode_key: str) -> dict:
    mode = MODES[mode_key]
    label = mode["label"]
    print(f"\n{'=' * 60}\n[compile/{label}] starting mode\n{'=' * 60}")

    # Apply inductor config BEFORE loading & compiling.
    mode["pre"]()

    # Default dynamo recompile_limit is 8; we have 6 workloads × multiple
    # shape variants, so we'd hit the cap and silently fall back to eager.
    # Bump it generously.
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64

    print(f"[compile/{label}] loading model fresh ...")
    t0 = time.time()
    model, _ = load_model(dtype=torch.bfloat16, device="cuda")
    print(f"[compile/{label}] loaded in {time.time() - t0:.1f}s")

    results = {}
    workloads = list_workloads()
    for name in workloads:
        print(f"\n[compile/{label}] --- Workload: {name} ---")
        rec: dict = {}
        try:
            # Reset dynamo state between workloads so each shape gets a
            # clean compile (rather than living off accumulated specializations
            # from earlier workloads).
            torch._dynamo.reset()
            compiled = torch.compile(model, **mode["kwargs"])
            wl = get_workload(name)
            run = _build_runner(compiled, wl)

            # Compile-time = the first forward latency.
            t_c = time.time()
            with torch.no_grad():
                first_out = run()
            torch.cuda.synchronize()
            compile_s = time.time() - t_c
            rec["compile_time_s"] = compile_s
            print(f"[compile/{label}]   compile/first-fwd: {compile_s:.2f} s")

            # Correctness: compare to eager reference.
            try:
                ref = _load_reference(name)
                cand = first_out.detach().to("cpu")
                # Save candidate so we can re-check correctness without re-running.
                cand_dir = RESULTS_DIR / "candidate_outputs" / label
                cand_dir.mkdir(parents=True, exist_ok=True)
                torch.save(cand, cand_dir / f"{name}.pt")
                # Reduce_overhead/cudagraphs sometimes returns a tensor backed
                # by a static buffer that gets overwritten on next call. We've
                # already synchronized and copied to cpu, so this is safe.
                check = check_outputs(ref, cand, dtype="bf16", task="standard")
                rec["correctness"] = check
                if not check["pass"]:
                    print(
                        f"[compile/{label}]   !!! CORRECTNESS FAIL on {name}: "
                        f"{check['reasons']}"
                    )
                else:
                    print(
                        f"[compile/{label}]   correctness OK "
                        f"cos={check['cos_sim']:.4f} l1_rel={check['l1_rel']:.4f} "
                        f"rmse={check['rmse']:.4f}"
                    )
            except FileNotFoundError:
                rec["correctness"] = {"error": "reference not found; run eager first"}
                print(f"[compile/{label}]   correctness skipped: ref missing")

            # Measure latency.
            stats = _measure(run)
            stats["tokens_per_sec"] = _tokens_per_sec(wl, stats["median_ms"])
            stats["mode"] = wl["mode"]
            stats["batch_size"] = wl["batch_size"]
            stats["seq_len"] = wl["seq_len"]
            rec.update(stats)
            print(
                f"[compile/{label}]   median={stats['median_ms']:.3f} ms  "
                f"peak_mem={stats['peak_mem_bytes']/2**20:.1f} MiB  "
                f"tok/s={stats['tokens_per_sec']:.1f}"
            )

            # Profile.
            if name in (PROFILE_PREFILL, PROFILE_DECODE):
                print(f"[compile/{label}]   capturing profiler trace ...")
                try:
                    _profile_one(run, TRACE_DIR / f"{label}_{name}.json")
                except Exception as pe:
                    print(f"[compile/{label}]   trace failed: {pe}")
                    rec["trace_error"] = str(pe)

            results[name] = rec
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[compile/{label}] FAILURE on {name}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e), "traceback": traceback.format_exc()}

    out_path = RESULTS_DIR / f"compile_{label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[compile/{label}] wrote {out_path}")

    # Drop the model + compile cache for this mode.
    del compiled
    del model
    torch.cuda.empty_cache()
    return results


def main():
    for key in ("default", "cudagraphs", "max_autotune"):
        try:
            run_mode(key)
        except Exception as e:
            print(f"[compile] mode {key} crashed entirely: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
