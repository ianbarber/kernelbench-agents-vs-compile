"""Harness for the swiglu kernel task.

Run from the task's sandbox directory:
    python harness.py

Exit codes:
  0 = candidate passes correctness; benchmark printed.
  2 = candidate fails correctness (FAIL_CORRECTNESS).
  3 = candidate mutates its inputs (FAIL_MUTATION).
  4 = candidate is non-deterministic (FAIL_NONDETERMINISTIC).
  1 = other error (import failure, exception during run, etc.).

Verdict ladder (highest -> lowest):
  PASS_STRICT  : passes strict tolerance + non-mutating + deterministic.
  PASS         : passes standard tolerance + non-mutating + deterministic.
  FAIL_MUTATION
  FAIL_NONDETERMINISTIC
  FAIL_CORRECTNESS
  ERROR
"""
from __future__ import annotations

import importlib.util
import json
import os
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
INDUCTOR_BASELINE_US = 361.3154750000003

SHAPE = (1, 512, 6144)
DTYPE = torch.bfloat16
SEED = 0xC0FFEE

# Determinism RMSE floor: two runs of the candidate on identical inputs must
# match to better than this in fp32. Pure-fp32 ops should hit 0.0; the tiny
# slack covers any innocuous asynchronous reduction nondeterminism.
DETERMINISM_RMSE_TOL = 1e-6


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs():
    g = torch.Generator(device="cuda")
    g.manual_seed(SEED)
    x = torch.randn(SHAPE, device="cuda", dtype=DTYPE, generator=g)
    y = torch.randn(SHAPE, device="cuda", dtype=DTYPE, generator=g)
    return x, y


def _scan_candidate_for_forbidden(path: Path) -> list[str]:
    """Cheap textual check for forbidden APIs in candidate.py."""
    try:
        src = path.read_text()
    except Exception:
        return []
    bad = []
    for token in ("torch.compile", "@torch.compile", "torch.jit.script",
                  "torch.jit.trace"):
        if token in src:
            bad.append(token)
    return bad


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32).flatten()
    b32 = b.detach().to(torch.float32).flatten()
    return float(torch.sqrt(((a32 - b32) ** 2).mean()))


def _emit(result: dict) -> None:
    print(json.dumps(result, indent=2))


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
    x_ref, y_ref = _make_inputs()
    try:
        ref_out = reference.run(x_ref, y_ref)
    except Exception:
        print("reference.run failed:", file=sys.stderr)
        traceback.print_exc()
        return 1
    torch.cuda.synchronize()

    # === Mutation check ===
    # Build fresh candidate inputs and keep pristine clones for comparison.
    x_cand, y_cand = _make_inputs()
    x_orig = x_cand.clone()
    y_orig = y_cand.clone()

    try:
        cand_out = candidate.run(x_cand, y_cand)
    except Exception:
        print("candidate.run raised:", file=sys.stderr)
        traceback.print_exc()
        print(json.dumps({"error": "candidate.run raised exception",
                          "verdict": "ERROR"}))
        return 1

    if not isinstance(cand_out, torch.Tensor):
        print(json.dumps({
            "error": "candidate.run did not return a torch.Tensor",
            "got_type": type(cand_out).__name__,
            "verdict": "ERROR",
        }))
        return 1

    torch.cuda.synchronize()

    mutates_x = not torch.equal(x_cand, x_orig)
    mutates_y = not torch.equal(y_cand, y_orig)

    # Snapshot the candidate's output for correctness BEFORE we run again
    # (since some candidates aliased into y, a second call would clobber it).
    cand_out_snapshot = cand_out.detach().clone()

    correctness = check_outputs(ref_out, cand_out_snapshot,
                                dtype="bf16", task="standard")
    correctness_strict = check_outputs(ref_out, cand_out_snapshot,
                                       dtype="bf16", task="strict")

    if mutates_x or mutates_y:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutates_x": bool(mutates_x),
            "mutates_y": bool(mutates_y),
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
    # Restore inputs (they should already be unchanged but be defensive),
    # run again, compare.
    x_cand2 = x_orig.clone()
    y_cand2 = y_orig.clone()
    try:
        cand_out2 = candidate.run(x_cand2, y_cand2)
    except Exception:
        print("candidate.run raised on 2nd call:", file=sys.stderr)
        traceback.print_exc()
        print(json.dumps({"error": "candidate.run raised on 2nd call",
                          "verdict": "ERROR"}))
        return 1
    torch.cuda.synchronize()

    det_rmse = _rmse(cand_out_snapshot, cand_out2)
    deterministic = bool(
        torch.equal(cand_out_snapshot, cand_out2) or det_rmse < DETERMINISM_RMSE_TOL
    )

    if not deterministic:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutates_x": False,
            "mutates_y": False,
            "deterministic": False,
            "determinism_rmse": det_rmse,
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
            "mutates_x": False,
            "mutates_y": False,
            "deterministic": True,
            "determinism_rmse": det_rmse,
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

    # Always use fresh inputs each bench call to defeat warm-cache asymmetries
    # and avoid reusing tensors a candidate might (incorrectly) cache against.
    x_b, y_b = _make_inputs()

    # By the time we get here the candidate has been proven non-mutating, so
    # we can safely reuse x_b/y_b across iterations — matching the original
    # smoke harness and giving comparable wall-clock numbers.
    try:
        cand_us = triton.testing.do_bench(
            lambda: candidate.run(x_b, y_b),
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
            lambda: reference.run(x_b, y_b),
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
        "mutates_x": False,
        "mutates_y": False,
        "deterministic": True,
        "determinism_rmse": det_rmse,
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
