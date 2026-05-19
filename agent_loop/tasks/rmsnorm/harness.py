"""Harness for the rmsnorm (residual-fused) kernel task.

Run from the task's sandbox directory:
    python harness.py

Exit codes:
  0 = candidate passes correctness; benchmark printed.
  2 = candidate fails correctness (FAIL_CORRECTNESS).
  3 = candidate mutates its inputs (FAIL_MUTATION).
  4 = candidate is non-deterministic (FAIL_NONDETERMINISTIC).
  5 = candidate fails correctness / crashes on a non-canonical shape
       (FAIL_SHAPE_GENERALIZATION).
  1 = other error (import failure, exception during run, etc.).

Verdict ladder (highest -> lowest):
  PASS_STRICT
  PASS
  FAIL_SHAPE_GENERALIZATION
  FAIL_MUTATION
  FAIL_NONDETERMINISTIC
  FAIL_CORRECTNESS
  ERROR

The candidate is tested at THREE shapes (the canonical one used for
benchmarking, plus two additional shapes Qwen3-1.7B exercises end-to-end).
Benchmark + speedup_vs_inductor only run at the canonical shape. The extra
shapes exist to catch kernels that hardcode the canonical hidden_size (the
kimi-v1 rmsnorm hardfaulted at head_dim=128 on a 3090 because of exactly this
class of bug).

Extra shapes:
  - (B=8, S=1, hidden=2048)   : decode-step input/post-attn RMSNorm
  - (B=1, S=512, head_dim=128): q_norm / k_norm per-head normalization
"""
from __future__ import annotations

import importlib.util
import json
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

# Load the standalone inductor microbench baseline (codegen-vs-codegen, do_bench
# median) from extract/microbench_inductor.json. Hard fallback only if missing.
_HARDCODED_FALLBACK_US = 259.0  # close to the profiler-aggregate mean (258.8 us)


def _load_inductor_baseline_us() -> float:
    candidates = [
        _PROJECT_ROOT / "extract" / "microbench_inductor.json",
    ]
    for p in candidates:
        try:
            data = json.loads(p.read_text())
            v = data.get("rmsnorm_us")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        except Exception:
            continue
    print(
        f"[harness] WARNING: could not load rmsnorm_us from "
        f"extract/microbench_inductor.json; falling back to "
        f"hardcoded {_HARDCODED_FALLBACK_US} us (profiler aggregate proxy, NOT "
        f"the standalone microbench).",
        file=sys.stderr,
    )
    return _HARDCODED_FALLBACK_US


INDUCTOR_BASELINE_US = _load_inductor_baseline_us()

# Canonical shapes for kernel _9 (prefill, residual-fused RMSNorm):
#   x:        (1, 512, 2048)  bf16
#   residual: (512, 2048)     bf16   (broadcast over leading singleton)
#   weight:   (2048,)         bf16
SHAPE_X = (1, 512, 2048)
SHAPE_RESIDUAL = (512, 2048)
SHAPE_WEIGHT = (2048,)
DTYPE = torch.bfloat16
EPS = 1e-6
SEED = 0xC0FFEE

# Extra shapes the model actually hits end-to-end. Each entry is
# (label, x_shape, residual_shape, weight_shape).
EXTRA_SHAPES = [
    ("decode_b8_s1_h2048", (8, 1, 2048), (8, 1, 2048), (2048,)),
    ("qk_norm_b1_s512_d128", (1, 512, 128), (1, 512, 128), (128,)),
]

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


def _make_inputs(
    shape_x=SHAPE_X,
    shape_residual=SHAPE_RESIDUAL,
    shape_weight=SHAPE_WEIGHT,
):
    g = torch.Generator(device="cuda")
    g.manual_seed(SEED)
    x = torch.randn(shape_x, device="cuda", dtype=DTYPE, generator=g)
    residual = torch.randn(shape_residual, device="cuda", dtype=DTYPE, generator=g)
    # Weight is a learned scale, conventionally near 1.0. Multiplying randn by
    # a small factor keeps RMSNorm output magnitudes representative of what
    # the model actually sees, and avoids extreme bf16 outputs that would
    # inflate the rmse threshold artificially.
    weight = (torch.randn(shape_weight, device="cuda", dtype=torch.float32,
                          generator=g) * 0.1 + 1.0).to(DTYPE)
    return x, residual, weight


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


def _check_extra_shape(candidate, reference, label, shape_x, shape_residual,
                       shape_weight):
    """Run candidate + reference at a non-canonical shape.

    Returns (ok: bool, info: dict). `ok` is True only if the candidate runs to
    completion AND passes standard correctness against the eager reference.
    """
    info = {"shape_x": list(shape_x), "shape_residual": list(shape_residual),
            "shape_weight": list(shape_weight)}
    try:
        x, r, w = _make_inputs(shape_x, shape_residual, shape_weight)
    except Exception as e:
        info["error"] = f"input construction failed: {e}"
        return False, info

    try:
        ref_out = reference.run(x, r, w, EPS)
        torch.cuda.synchronize()
    except Exception:
        info["error"] = "reference.run raised"
        info["traceback"] = traceback.format_exc()
        return False, info

    try:
        cand_out = candidate.run(x, r, w, EPS)
        torch.cuda.synchronize()
    except Exception:
        info["error"] = "candidate.run raised"
        info["traceback"] = traceback.format_exc()
        return False, info

    if not isinstance(cand_out, torch.Tensor):
        info["error"] = f"candidate.run returned non-tensor ({type(cand_out).__name__})"
        return False, info

    if cand_out.shape != ref_out.shape:
        info["error"] = (
            f"shape mismatch: cand={tuple(cand_out.shape)} "
            f"ref={tuple(ref_out.shape)}"
        )
        return False, info

    corr = check_outputs(ref_out, cand_out, dtype="bf16", task="standard")
    info["correctness"] = corr
    return bool(corr["pass"]), info


def main() -> int:
    # Honor --help; everything else runs the full check.
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0

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
    x_ref, res_ref, w_ref = _make_inputs()
    try:
        ref_out = reference.run(x_ref, res_ref, w_ref, EPS)
    except Exception:
        print("reference.run failed:", file=sys.stderr)
        traceback.print_exc()
        return 1
    torch.cuda.synchronize()

    # === Mutation check ===
    # Build fresh candidate inputs and keep pristine clones for comparison.
    x_cand, res_cand, w_cand = _make_inputs()
    x_orig = x_cand.clone()
    res_orig = res_cand.clone()
    w_orig = w_cand.clone()

    try:
        cand_out = candidate.run(x_cand, res_cand, w_cand, EPS)
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
    mutates_residual = not torch.equal(res_cand, res_orig)
    mutates_weight = not torch.equal(w_cand, w_orig)

    # Snapshot the candidate's output for correctness BEFORE we run again
    # (in case the second call clobbers shared storage).
    cand_out_snapshot = cand_out.detach().clone()

    correctness = check_outputs(ref_out, cand_out_snapshot,
                                dtype="bf16", task="standard")
    correctness_strict = check_outputs(ref_out, cand_out_snapshot,
                                       dtype="bf16", task="strict")

    if mutates_x or mutates_residual or mutates_weight:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutates_x": bool(mutates_x),
            "mutates_residual": bool(mutates_residual),
            "mutates_weight": bool(mutates_weight),
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
    x_cand2 = x_orig.clone()
    res_cand2 = res_orig.clone()
    w_cand2 = w_orig.clone()
    try:
        cand_out2 = candidate.run(x_cand2, res_cand2, w_cand2, EPS)
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
            "mutates_residual": False,
            "mutates_weight": False,
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

    # === Correctness gate (standard, canonical shape) ===
    if not correctness["pass"]:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutates_x": False,
            "mutates_residual": False,
            "mutates_weight": False,
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

    # === Shape-generalization gate ===
    # Run candidate + reference at every extra shape. Hard-fault or correctness
    # failure on ANY shape demotes the verdict to FAIL_SHAPE_GENERALIZATION.
    extra_shape_results = {}
    extra_shape_ok = True
    for label, sx, sr, sw in EXTRA_SHAPES:
        ok, info = _check_extra_shape(candidate, reference, label, sx, sr, sw)
        extra_shape_results[label] = {"pass": bool(ok), **info}
        if not ok:
            extra_shape_ok = False

    if not extra_shape_ok:
        result = {
            "correctness": correctness,
            "correctness_strict": correctness_strict,
            "mutates_x": False,
            "mutates_residual": False,
            "mutates_weight": False,
            "deterministic": True,
            "determinism_rmse": det_rmse,
            "extra_shapes": extra_shape_results,
            "candidate_us": None,
            "reference_us": None,
            "inductor_baseline_us": INDUCTOR_BASELINE_US,
            "speedup_vs_inductor": None,
            "verdict": "FAIL_SHAPE_GENERALIZATION",
        }
        _emit(result)
        return 5

    # === Benchmark (canonical shape only) ===
    import triton.testing

    x_b, res_b, w_b = _make_inputs()

    try:
        cand_us = triton.testing.do_bench(
            lambda: candidate.run(x_b, res_b, w_b, EPS),
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
            lambda: reference.run(x_b, res_b, w_b, EPS),
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
        "mutates_residual": False,
        "mutates_weight": False,
        "deterministic": True,
        "determinism_rmse": det_rmse,
        "extra_shapes": extra_shape_results,
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
