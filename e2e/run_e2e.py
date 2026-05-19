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

Replicate mode (--trials N > 1):
  The orchestrator process spawns N fresh subprocesses per config x workload
  set. Each subprocess loads the model from scratch and writes its results to
  e2e/results/<config>_trial<i>.json. After all trials, the orchestrator
  aggregates to e2e/results/<config>.json with median + MAD (median absolute
  deviation) per workload. Default --trials 1 preserves the legacy in-process
  behaviour.

Inductor cache hygiene:
  At the START of every (config x trial) run we delete the inductor on-disk
  cache (~/.cache/torch_inductor or $TORCHINDUCTOR_CACHE_DIR). This prevents
  the Triton autotuner-state pollution we hit once where `eager_both_pure`'s
  median bounced 32.9 -> 64.96 -> 29.93 ms across consecutive runs. We do it
  per-trial-subprocess in replicate mode (because each subprocess is a clean
  CUDA context) and per-config in N=1 mode (since the process is shared).
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
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
PINNED_TOKEN_DIR = ROOT / "baselines" / "results" / "eager_last_token_ids"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 25
REP = 100


def _inductor_cache_dir() -> Path:
    """Return the directory torch.inductor uses for its on-disk cache.

    `import torch._inductor` sets `TORCHINDUCTOR_CACHE_DIR` to
    `/tmp/torchinductor_<user>` on this nightly even when the env var was
    initially unset, so prefer the env var (which torch itself respects)
    and only fall back to `~/.cache/torch_inductor` for completeness.
    """
    env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "torch_inductor"


def _clear_inductor_cache(cfg: str) -> None:
    """Wipe the inductor cache before a config (or trial) starts.

    Guarantees fresh Triton autotuner runs and eliminates the cross-config
    autotuner-state pollution that produced the historical `eager_both_pure`
    median wobble. Also clear any in-process inductor caches that survive
    inside the same Python process.
    """
    d = _inductor_cache_dir()
    if d.exists():
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"[{cfg}] WARN: failed to rmtree {d}: {e}")
    # Best-effort: also reset any in-process Triton/Inductor caches so a
    # single-process N=1 invocation doesn't carry over autotune state from a
    # previous config that ran in the same process.
    try:
        import torch._inductor.codecache as _cc
        if hasattr(_cc, "FxGraphCache") and hasattr(_cc.FxGraphCache, "clear"):
            _cc.FxGraphCache.clear()
    except Exception:
        pass
    try:
        torch._dynamo.reset()  # type: ignore[attr-defined]
    except Exception:
        pass
    print(f"[{cfg}] cleared inductor cache")


def _is_decode_workload(name: str) -> bool:
    return name.startswith("decode_")


def _pinned_token_available(workload_name: str) -> bool:
    return (PINNED_TOKEN_DIR / f"{workload_name}.pt").exists()


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
    """Install patches per named config. Always start clean (uninstall first).

    Returns a dict of side-state the caller may need to keep alive (e.g.
    compiled model handles, cudagraph capture state). For most configs this
    is empty; cudagraphs configs return the compiled / captured callables.
    """
    P.uninstall(model)
    state = {"forward_override": None, "compiled_model": None}

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
    elif cfg == "compile_default":
        # `torch.compile` is applied per-workload below (so we can pick the
        # right `dynamic` setting per mode). Just mark the intent.
        state["compile_mode"] = "default"
        state["compile_cudagraphs"] = False
    elif cfg == "compile_default_cgraphs":
        state["compile_mode"] = "default"
        state["compile_cudagraphs"] = True
    elif cfg == "eager_all_winners_cgraphs":
        # Patch eager with all winners, then wrap each per-workload runner in
        # a raw `torch.cuda.CUDAGraph()` capture. We chose raw CUDAGraph over
        # `make_graphed_callables` because:
        #   1. our patched forward path uses an HF DynamicCache (mutable
        #      Python object). `make_graphed_callables` requires a callable
        #      with static tensor I/O; passing a DynamicCache would force us
        #      to expose every cache tensor in the signature, fight HF's
        #      class layout, and would still need a separate graph per
        #      workload (different ctx_len → different cache shapes).
        #   2. raw capture lets us reuse the existing `_build_runner` runner
        #      closure unchanged — we just capture it and replay.
        # The penalty is per-run capture overhead, but capture happens once
        # per (config, workload) and the replay loop is what do_bench
        # measures. This is the same pattern HF uses internally for
        # `torch.cuda.graphs.make_graphed_callables` and what the PyTorch
        # team recommends for "irregular" callables.
        P.install_swiglu_kimi(model)
        P.install_rmsnorm_claude_pure(model)
        P.install_sdpa_prelude_kimi(model)
        state["wrap_cudagraph"] = True
    else:
        raise ValueError(cfg)
    return state


def _wrap_with_cudagraph(run_fn):
    """Capture `run_fn` into a CUDAGraph and return a replay-only callable.

    Contract: `run_fn` must be a zero-arg callable that does its own input
    prep (e.g. the closure from `_build_runner`). The closure is run a few
    times for warmup (so any lazy kernel JITs / autotuners can settle),
    then captured, and the returned callable just replays the graph.
    """
    # Warmup outside the capture so any lazy autotune / JIT happens now.
    for _ in range(3):
        run_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # One more dry run on the capture stream so any side-effects (e.g.
        # KV cache restore allocations) are settled before capture.
        run_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        run_fn()

    def replay():
        g.replay()
    return replay


def _maybe_compile(model, state, workload):
    """Apply torch.compile if the config requested it. Returns a callable run."""
    mode = state.get("compile_mode")
    if mode is None:
        # No compile requested.
        return _build_runner(model, workload)

    use_cgraphs = bool(state.get("compile_cudagraphs", False))
    # Toggle inductor's cudagraphs flag at compile time.
    try:
        import torch._inductor.config as _icfg
        _icfg.triton.cudagraphs = use_cgraphs  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[compile] WARN: could not toggle triton.cudagraphs: {e}")

    compiled = torch.compile(model, mode=mode, dynamic=False)
    # Build the runner against the compiled model.
    return _build_runner(compiled, workload)


def _run_config(cfg: str, workloads, trial_output: Path | None = None):
    """Run a single (config, workloads) pass in this process.

    If `trial_output` is given, that path is written instead of
    e2e/results/{cfg}.json. Used by the replicate-mode orchestrator to drop
    per-trial JSON into e2e/results/{cfg}_trial<i>.json.
    """
    print(f"\n========== Config: {cfg} ==========")
    _clear_inductor_cache(cfg)
    # Fresh model.
    t0 = time.time()
    model, _ = load_model(dtype=torch.bfloat16, device="cuda")
    print(f"[{cfg}] loaded model in {time.time() - t0:.1f}s")
    state = _install_for_config(model, cfg)

    results = {}
    for name in workloads:
        print(f"\n[{cfg}] === {name} ===")
        try:
            # Pin last_token_ids on decode workloads when the eager artifact
            # is available — guarantees every config decodes from the same
            # starting token. Falls back transparently if the file is
            # missing.
            pin = _is_decode_workload(name) and _pinned_token_available(name)
            wl = get_workload(name, pin_last_token=pin)
            run = _maybe_compile(model, state, wl)
            if state.get("wrap_cudagraph"):
                run = _wrap_with_cudagraph(run)
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
                print(f"[{cfg}]   correctness ok (standard) cos={corr['cos_sim']:.5f} l1={corr['l1_rel']:.4f} rmse={corr['rmse']:.4f}  strict_pass={corr_strict['pass']}  pinned={wl.get('pinned_last_token', False)}")

            stats = _measure(run)
            stats["tokens_per_sec"] = _tokens_per_sec(wl, stats["median_ms"])
            stats["mode"] = wl["mode"]
            stats["batch_size"] = wl["batch_size"]
            stats["seq_len"] = wl["seq_len"]
            stats["pinned_last_token"] = bool(wl.get("pinned_last_token", False))
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

    out_path = trial_output if trial_output is not None else (RESULTS_DIR / f"{cfg}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{cfg}] wrote {out_path}")

    # Free model.
    P.uninstall(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mad(xs, med=None):
    """Median absolute deviation."""
    if not xs:
        return float("nan")
    if med is None:
        med = _median(xs)
    return _median([abs(x - med) for x in xs])


def _aggregate_trials(cfg: str, workloads, n_trials: int):
    """Read e2e/results/{cfg}_trial{i}.json for i in [0..N) and aggregate."""
    per_wl_medians: dict[str, list[float]] = {w: [] for w in workloads}
    per_trial = []
    for i in range(n_trials):
        p = RESULTS_DIR / f"{cfg}_trial{i}.json"
        if not p.exists():
            print(f"[aggregate] WARN: missing {p}")
            continue
        with open(p) as f:
            data = json.load(f)
        per_trial.append(data)
        for w in workloads:
            entry = data.get(w)
            if entry and isinstance(entry, dict) and "median_ms" in entry:
                per_wl_medians[w].append(float(entry["median_ms"]))

    agg = {}
    for w in workloads:
        ms = per_wl_medians[w]
        if not ms:
            # Pass through the last (or first) trial's entry verbatim so the
            # error/traceback is preserved.
            for d in per_trial:
                if w in d:
                    agg[w] = d[w]
                    break
            continue
        med = _median(ms)
        agg[w] = {
            "median_ms": med,
            "mad_ms": _mad(ms, med),
            "trials_ms": ms,
            "n_trials": len(ms),
        }
        # Carry forward metadata from trial 0 if available.
        proto = None
        for d in per_trial:
            if w in d and isinstance(d[w], dict) and "median_ms" in d[w]:
                proto = d[w]
                break
        if proto is not None:
            for k in ("mode", "batch_size", "seq_len", "tokens_per_sec",
                      "peak_mem_bytes", "correctness", "pinned_last_token"):
                if k in proto:
                    agg[w][k] = proto[k]
            # Recompute tok/s from the aggregated median for honesty.
            try:
                wl_seq = agg[w]["seq_len"]
                wl_bs = agg[w]["batch_size"]
                toks = wl_seq * wl_bs if agg[w]["mode"] == "prefill" else wl_bs
                agg[w]["tokens_per_sec"] = toks / (med / 1000.0)
            except Exception:
                pass

    out_path = RESULTS_DIR / f"{cfg}.json"
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[aggregate] {cfg}: wrote {out_path} (n_trials={n_trials})")


def _spawn_trial(cfg: str, workloads, trial_index: int) -> int:
    """Spawn a fresh subprocess for one (config, trial) pair.

    Each subprocess loads model from scratch and gets a clean inductor cache.
    Returns the subprocess exit code.
    """
    trial_path = RESULTS_DIR / f"{cfg}_trial{trial_index}.json"
    cmd = [
        sys.executable, "-m", "e2e.run_e2e",
        "--configs", cfg,
        "--workloads", *workloads,
        "--trial-index", str(trial_index),
        "--trial-output", str(trial_path),
    ]
    print(f"[orchestrator] spawning trial {trial_index} for {cfg}: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Subset of configs to run.")
    ap.add_argument("--workloads", nargs="*", default=None,
                    help="Subset of workloads.")
    ap.add_argument("--trials", type=int, default=1,
                    help="Replicate count. When >1, each (config, trial) "
                         "runs in a fresh subprocess with a cold inductor "
                         "cache; results aggregated as median +/- MAD.")
    ap.add_argument("--trial-index", type=int, default=None,
                    help="Internal: index of the trial subprocess to write. "
                         "Set by the orchestrator when --trials > 1.")
    ap.add_argument("--trial-output", type=str, default=None,
                    help="Internal: where the trial subprocess should "
                         "write its JSON output.")
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
        "compile_default_cgraphs",
        "eager_all_winners_cgraphs",
    ]
    configs = args.configs or all_configs

    # Subprocess trial mode: just run the one config and write to the
    # designated trial output path.
    if args.trial_index is not None:
        assert args.trial_output is not None
        assert len(configs) == 1, "trial subprocess must run exactly one config"
        _run_config(configs[0], workloads,
                    trial_output=Path(args.trial_output))
        return

    if args.trials <= 1:
        for cfg in configs:
            _run_config(cfg, workloads)
        return

    # Replicate mode: orchestrator spawns N subprocesses per config and
    # aggregates afterwards.
    for cfg in configs:
        for i in range(args.trials):
            rc = _spawn_trial(cfg, workloads, i)
            if rc != 0:
                print(f"[orchestrator] WARN: trial {i} for {cfg} returned rc={rc}")
        _aggregate_trials(cfg, workloads, args.trials)


if __name__ == "__main__":
    main()
