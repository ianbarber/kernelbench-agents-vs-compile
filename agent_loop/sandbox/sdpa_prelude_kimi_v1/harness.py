"""Harness for the sdpa_prelude (Q/K/V + RoPE + mask) fused-prelude task.

Run from the task's sandbox directory:
    python harness.py

Exit codes:
  0 = candidate passes correctness; benchmark printed.
  2 = candidate fails correctness (FAIL_CORRECTNESS).
  3 = candidate mutates its inputs (FAIL_MUTATION).
  4 = candidate is non-deterministic (FAIL_NONDETERMINISTIC).
  1 = other error (import failure, exception during run, etc.).

Verdict ladder (highest -> lowest):
  PASS_STRICT  : passes strict tolerance on all outputs + non-mutating + deterministic.
  PASS         : passes standard tolerance on all outputs + non-mutating + deterministic.
  FAIL_MUTATION
  FAIL_NONDETERMINISTIC
  FAIL_CORRECTNESS
  ERROR
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import torch

# Make the project root importable so `workload.correctness` resolves regardless
# of where the sandbox lives.
_PROJECT_ROOT = Path("/home/ianbarber/Projects/KernelBench")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workload.correctness import check_outputs  # noqa: E402

HERE = Path(__file__).parent.resolve()

# Names of the four output tensors, in order.
_OUTPUT_NAMES = ("q", "k", "v", "mask")

# Hardcoded fallback if extract/microbench_inductor.json doesn't yet have
# sdpa_prelude_us. Set to the profiler-aggregate-derived ballpark; the
# microbench task should overwrite this in microbench_inductor.json.
_HARDCODED_FALLBACK_US = 200.0


def _load_inductor_baseline_us() -> float:
    candidates = [
        _PROJECT_ROOT / "extract" / "microbench_inductor.json",
    ]
    for p in candidates:
        try:
            data = json.loads(p.read_text())
            v = data.get("sdpa_prelude_us")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        except Exception:
            continue
    print(
        f"[harness] WARNING: could not load sdpa_prelude_us from "
        f"extract/microbench_inductor.json; falling back to "
        f"hardcoded {_HARDCODED_FALLBACK_US} us. This number is a rough "
        f"placeholder, NOT a standalone microbench.",
        file=sys.stderr,
    )
    return _HARDCODED_FALLBACK_US


INDUCTOR_BASELINE_US = _load_inductor_baseline_us()

# Canonical shapes for the prefill_512_b1 case in Qwen3-1.7B.
BATCH = 1
SEQ = 512
HIDDEN = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16
EPS = 1e-6
SEED = 0xC0FFEE

# Determinism RMSE floor: two runs of the candidate on identical inputs must
# match to better than this in fp32. Pure-fp32 ops should hit 0.0; small slack
# covers asynchronous reductions.
DETERMINISM_RMSE_TOL = 1e-6

# Qwen3's RoPE uses theta=1e6 (per HF config). We bake an inv_freq that matches
# that recipe so the reference produces the same numerical behaviour as the
# inductor kernels would. Agents see this as a fixed input tensor.
_ROPE_THETA = 1_000_000.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs():
    """Build canonical inputs. Tensors are sampled to look like trained-model
    statistics: hidden_states ~ N(0, 1), weights ~ N(0, 0.02) (LLM scale),
    norm scales ~ 1.0 + 0.1 * N(0, 1)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(SEED)

    hidden_states = torch.randn(
        (BATCH, SEQ, HIDDEN), device="cuda", dtype=DTYPE, generator=g
    )
    # Projection weights (LLM initialisation scale).
    w_q = (torch.randn((NUM_Q_HEADS * HEAD_DIM, HIDDEN), device="cuda",
                       dtype=torch.float32, generator=g) * 0.02).to(DTYPE)
    w_k = (torch.randn((NUM_KV_HEADS * HEAD_DIM, HIDDEN), device="cuda",
                       dtype=torch.float32, generator=g) * 0.02).to(DTYPE)
    w_v = (torch.randn((NUM_KV_HEADS * HEAD_DIM, HIDDEN), device="cuda",
                       dtype=torch.float32, generator=g) * 0.02).to(DTYPE)
    # Per-head RMSNorm scales (near 1.0).
    w_q_norm = (torch.randn((HEAD_DIM,), device="cuda", dtype=torch.float32,
                            generator=g) * 0.1 + 1.0).to(DTYPE)
    w_k_norm = (torch.randn((HEAD_DIM,), device="cuda", dtype=torch.float32,
                            generator=g) * 0.1 + 1.0).to(DTYPE)
    # Inverse-frequency table (Qwen3 base=1e6, head_dim=128 -> 64 inv_freqs).
    inv_freq = (1.0 / (_ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32)
        / HEAD_DIM
    ))).contiguous()
    # Position ids and attention mask (no padding for the canonical bench).
    position_ids = torch.arange(SEQ, device="cuda", dtype=torch.int64).view(1, SEQ)
    attention_mask = torch.ones((BATCH, SEQ), device="cuda", dtype=torch.int64)

    return (
        hidden_states, w_q, w_k, w_v, w_q_norm, w_k_norm,
        inv_freq, position_ids, attention_mask,
    )


def _scan_candidate_for_forbidden(path: Path) -> list[str]:
    """Cheap textual check for forbidden APIs in candidate.py."""
    try:
        src = path.read_text()
    except Exception:
        return []
    bad = []
    forbidden = (
        "torch.compile",
        "@torch.compile",
        "torch.jit.script",
        "torch.jit.trace",
        # SDPA backends defeat the comparison: agents may NOT use them.
        "torch.nn.functional.scaled_dot_product_attention",
        "F.scaled_dot_product_attention",
        "torch.ops.aten._scaled_dot_product_efficient_attention",
        "torch.ops.aten._scaled_dot_product_flash_attention",
        "torch.ops.aten.scaled_dot_product_attention",
    )
    for token in forbidden:
        if token in src:
            bad.append(token)
    return bad


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32).flatten()
    b32 = b.detach().to(torch.float32).flatten()
    # The mask can contain -inf; treat those positions as matching only if
    # both are -inf, else as a finite mismatch via clamp.
    diff = a32 - b32
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    # -inf - -inf yields nan; the where above handles that. -inf - 0 = -inf:
    # that's a real mismatch, which we mark as a large but finite penalty so
    # rmse stays comparable.
    diff = torch.where(torch.isinf(diff), torch.full_like(diff, 1e6), diff)
    return float(torch.sqrt((diff ** 2).mean()))


def _emit(result: dict) -> None:
    print(json.dumps(result, indent=2))


def _check_correctness_all(refs, cands, task: str):
    """Run check_outputs on each (ref, cand) pair, return combined dict."""
    per_output = {}
    all_pass = True
    for name, ref, cand in zip(_OUTPUT_NAMES, refs, cands):
        # The mask contains -inf; check_outputs uses fp32 ops which keep
        # -inf positions stable. cos_sim over a tensor with -inf would be nan.
        # We replace -inf with a large negative finite number for the
        # correctness check (still distinguishes "masked" from "unmasked").
        if name == "mask":
            ref_check = ref.clone().to(torch.float32)
            cand_check = cand.clone().to(torch.float32)
            ref_check[torch.isinf(ref_check)] = -1.0e4
            cand_check[torch.isinf(cand_check)] = -1.0e4
            res = check_outputs(ref_check.to(torch.bfloat16),
                                cand_check.to(torch.bfloat16),
                                dtype="bf16", task=task)
        else:
            res = check_outputs(ref, cand, dtype="bf16", task=task)
        per_output[name] = res
        all_pass = all_pass and bool(res["pass"])
    combined = {
        "pass": all_pass,
        "per_output": per_output,
        # Surface aggregate reasons for convenience.
        "reasons": [
            f"{name}: {r}" for name, res in per_output.items()
            for r in res["reasons"]
        ],
    }
    return combined


def main() -> int:
    candidate_path = HERE / "candidate.py"
    reference_path = HERE / "reference.py"

    if not candidate_path.exists():
        print(json.dumps({
            "error": "candidate.py not found",
            "path": str(candidate_path),
        }))
        return 1

    if candidate_path.stat().st_size == 0:
        print(json.dumps({
            "error": "candidate.py is empty",
        }))
        return 1

    forbidden = _scan_candidate_for_forbidden(candidate_path)
    if forbidden:
        print(json.dumps({
            "error": "candidate.py contains forbidden tokens",
            "forbidden": forbidden,
        }))
        return 1

    try:
        candidate = _load_module("candidate_mod", candidate_path)
    except Exception:
        print("Failed to import candidate.py:", file=sys.stderr)
        traceback.print_exc()
        return 1

    try:
        reference = _load_module("reference_mod", reference_path)
    except Exception:
        print("Failed to import reference.py:", file=sys.stderr)
        traceback.print_exc()
        return 1

    if not hasattr(candidate, "run"):
        print(json.dumps({"error": "candidate.py missing run()"}))
        return 1

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available"}))
        return 1

    torch.cuda.synchronize()

    # Build canonical reference inputs and run reference once.
    ref_inputs = _make_inputs()
    try:
        ref_out = reference.run(*ref_inputs, EPS)
    except Exception:
        print("reference.run failed:", file=sys.stderr)
        traceback.print_exc()
        return 1
    torch.cuda.synchronize()

    if not isinstance(ref_out, tuple) or len(ref_out) != len(_OUTPUT_NAMES):
        print(json.dumps({
            "error": "reference.run did not return a 4-tuple",
            "verdict": "ERROR",
        }))
        return 1

    # === Mutation check ===
    cand_inputs = _make_inputs()
    originals = [t.clone() for t in cand_inputs]

    try:
        cand_out = candidate.run(*cand_inputs, EPS)
    except Exception:
        print("candidate.run raised:", file=sys.stderr)
        traceback.print_exc()
        print(json.dumps({"error": "candidate.run raised exception",
                          "verdict": "ERROR"}))
        return 1

    if not isinstance(cand_out, tuple) or len(cand_out) != len(_OUTPUT_NAMES):
        print(json.dumps({
            "error": f"candidate.run must return a 4-tuple "
                     f"(q, k, v, mask); got {type(cand_out).__name__} "
                     f"with len={len(cand_out) if hasattr(cand_out, '__len__') else 'n/a'}",
            "verdict": "ERROR",
        }))
        return 1
    for i, t in enumerate(cand_out):
        if not isinstance(t, torch.Tensor):
            print(json.dumps({
                "error": f"candidate.run output[{i}] ({_OUTPUT_NAMES[i]}) "
                         f"is not a torch.Tensor (got {type(t).__name__})",
                "verdict": "ERROR",
            }))
            return 1

    torch.cuda.synchronize()

    input_names = [
        "hidden_states", "w_q", "w_k", "w_v", "w_q_norm", "w_k_norm",
        "inv_freq", "position_ids", "attention_mask",
    ]
    mutations = {
        name: not torch.equal(t, orig)
        for name, t, orig in zip(input_names, cand_inputs, originals)
    }
    any_mutation = any(mutations.values())

    # Snapshot the candidate's output for correctness BEFORE we run again
    # (in case the second call clobbers shared storage).
    cand_out_snapshot = tuple(t.detach().clone() for t in cand_out)

    correctness = _check_correctness_all(ref_out, cand_out_snapshot, "standard")
    correctness_strict = _check_correctness_all(
        ref_out, cand_out_snapshot, "strict"
    )

    if any_mutation:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutations": mutations,
            "deterministic": None,
            "candidate_us": None,
            "reference_us": None,
            "inductor_baseline_us": INDUCTOR_BASELINE_US,
            "speedup_vs_inductor": None,
            "verdict": "FAIL_MUTATION",
        }
        _emit(result)
        return 3

    # === Determinism check ===
    cand_inputs2 = _make_inputs()
    try:
        cand_out2 = candidate.run(*cand_inputs2, EPS)
    except Exception:
        print("candidate.run raised on 2nd call:", file=sys.stderr)
        traceback.print_exc()
        print(json.dumps({"error": "candidate.run raised on 2nd call",
                          "verdict": "ERROR"}))
        return 1
    torch.cuda.synchronize()

    det_rmses = {
        name: _rmse(a, b)
        for name, a, b in zip(_OUTPUT_NAMES, cand_out_snapshot, cand_out2)
    }
    deterministic = all(v < DETERMINISM_RMSE_TOL or
                        torch.equal(a, b)
                        for v, (name, a, b) in
                        zip(det_rmses.values(),
                            zip(_OUTPUT_NAMES, cand_out_snapshot, cand_out2)))

    if not deterministic:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutations": mutations,
            "deterministic": False,
            "determinism_rmse": det_rmses,
            "candidate_us": None,
            "reference_us": None,
            "inductor_baseline_us": INDUCTOR_BASELINE_US,
            "speedup_vs_inductor": None,
            "verdict": "FAIL_NONDETERMINISTIC",
        }
        _emit(result)
        return 4

    # === Correctness gate (standard) ===
    if not correctness["pass"]:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutations": mutations,
            "deterministic": True,
            "determinism_rmse": det_rmses,
            "candidate_us": None,
            "reference_us": None,
            "inductor_baseline_us": INDUCTOR_BASELINE_US,
            "speedup_vs_inductor": None,
            "verdict": "FAIL_CORRECTNESS",
        }
        _emit(result)
        return 2

    # === Benchmark ===
    import triton.testing

    bench_inputs = _make_inputs()

    try:
        cand_us = triton.testing.do_bench(
            lambda: candidate.run(*bench_inputs, EPS),
            warmup=25,
            rep=100,
            return_mode="median",
        ) * 1000.0  # do_bench returns ms; we want us
    except Exception:
        print("candidate do_bench failed:", file=sys.stderr)
        traceback.print_exc()
        print(json.dumps({"error": "candidate do_bench raised",
                          "verdict": "ERROR"}))
        return 1

    try:
        ref_us = triton.testing.do_bench(
            lambda: reference.run(*bench_inputs, EPS),
            warmup=25,
            rep=100,
            return_mode="median",
        ) * 1000.0
    except Exception:
        print("reference do_bench failed:", file=sys.stderr)
        traceback.print_exc()
        ref_us = None

    speedup = None
    if cand_us and cand_us > 0:
        speedup = INDUCTOR_BASELINE_US / cand_us

    pass_strict = bool(correctness_strict["pass"])
    verdict = "PASS_STRICT" if pass_strict else "PASS"

    result = {
        "correctness": correctness,
        "correctness_strict": correctness_strict,
        "mutations": mutations,
        "deterministic": True,
        "determinism_rmse": det_rmses,
        "candidate_us": cand_us,
        "reference_us": ref_us,
        "inductor_baseline_us": INDUCTOR_BASELINE_US,
        "speedup_vs_inductor": speedup,
        "verdict": verdict,
    }
    _emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
