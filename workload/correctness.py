"""Numerical correctness checks for candidate kernels.

Two thresholds matching the KernelBenchX paper:

  standard: cos_sim >= 0.95, l1_rel <= 0.05, rmse <= 0.10
  strict:   cos_sim >= 0.99, l1_rel <= 0.01, rmse <= 0.01

And a two-mode outlier protocol where the candidate must pass on both
normal N(0,1) inputs and on a copy with rare large-magnitude entries
injected (probability 0.1%, scale 50x).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch

_THRESHOLDS = {
    "standard": {"cos_sim": 0.95, "l1_rel": 0.05, "rmse": 0.10},
    "strict": {"cos_sim": 0.99, "l1_rel": 0.01, "rmse": 0.01},
}

# Tolerance floor used for relative L1 to avoid divide-by-near-zero. bf16 has
# ~3e-3 unit roundoff, so we treat anything smaller than that as effectively
# zero when computing the denominator.
_DTYPE_EPS = {
    "bf16": 1e-3,
    "fp16": 1e-3,
    "fp32": 1e-6,
}


def _to_float32_flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).flatten()


def _to_float64_flat(t: torch.Tensor) -> torch.Tensor:
    # Cosine similarity over hundreds of millions of elements (e.g. logits of
    # shape (1, 2048, 151936) ≈ 311M values) loses too much precision when the
    # dot product is accumulated in fp32 — observed cos_sim values > 1 and < 0
    # on outputs that are actually within tolerance. Use fp64 for the reduce.
    return t.detach().to(torch.float64).flatten()


def check_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    dtype: str = "bf16",
    task: str = "standard",
) -> Dict:
    """Compare candidate vs reference. Returns metrics dict + pass/fail."""
    if task not in _THRESHOLDS:
        raise ValueError(
            f"Unknown task '{task}'. Expected one of {list(_THRESHOLDS)}"
        )
    thresholds = _THRESHOLDS[task]
    eps = _DTYPE_EPS.get(dtype, 1e-6)

    reasons: List[str] = []

    if reference.shape != candidate.shape:
        return {
            "pass": False,
            "cos_sim": float("nan"),
            "l1_rel": float("nan"),
            "rmse": float("nan"),
            "reasons": [
                f"shape mismatch: reference={tuple(reference.shape)} "
                f"vs candidate={tuple(candidate.shape)}"
            ],
        }

    # fp32 for the abs-difference metrics (no cancellation, accurate).
    r32 = _to_float32_flat(reference)
    c32 = _to_float32_flat(candidate)

    if not torch.isfinite(c32).all():
        reasons.append("candidate contains non-finite values")

    # fp64 for cosine (large tensors + signed accumulation hits fp32 limits).
    r64 = _to_float64_flat(reference)
    c64 = _to_float64_flat(candidate)
    nr = torch.linalg.norm(r64)
    nc = torch.linalg.norm(c64)
    denom = nr * nc
    if denom.item() == 0.0:
        cos_sim = 1.0 if (nr.item() == 0.0 and nc.item() == 0.0) else 0.0
    else:
        cos_sim = float(torch.dot(r64, c64) / denom)
        # Clamp tiny fp64 overshoots — true cosine is in [-1, 1].
        cos_sim = max(-1.0, min(1.0, cos_sim))

    # Relative L1: mean(|r - c|) / (mean(|r|) + eps).
    abs_diff = (r32 - c32).abs()
    l1_rel = float(abs_diff.mean() / (r32.abs().mean() + eps))

    # Root mean square error.
    rmse = float(torch.sqrt((abs_diff ** 2).mean()))

    if cos_sim < thresholds["cos_sim"]:
        reasons.append(
            f"cos_sim {cos_sim:.4f} < {thresholds['cos_sim']}"
        )
    if l1_rel > thresholds["l1_rel"]:
        reasons.append(
            f"l1_rel {l1_rel:.4f} > {thresholds['l1_rel']}"
        )
    if rmse > thresholds["rmse"]:
        reasons.append(
            f"rmse {rmse:.4f} > {thresholds['rmse']}"
        )

    return {
        "pass": len(reasons) == 0,
        "cos_sim": cos_sim,
        "l1_rel": l1_rel,
        "rmse": rmse,
        "reasons": reasons,
    }


def _inject_outliers(
    x: torch.Tensor,
    prob: float = 1e-3,
    scale: float = 50.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a copy of `x` where ~`prob` of entries are multiplied by `scale`."""
    mask = torch.rand(x.shape, device=x.device, generator=generator) < prob
    out = x.clone()
    out[mask] = out[mask] * scale
    return out


def check_outputs_outlier_mode(
    reference: torch.Tensor,
    candidate_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_fn: Callable[[], torch.Tensor],
    dtype: str = "bf16",
    task: str = "standard",
    outlier_prob: float = 1e-3,
    outlier_scale: float = 50.0,
    reference_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0,
) -> Dict:
    """Two-mode protocol: run candidate on normal and outlier-injected inputs.

    Arguments:
      reference: a reference output for the *normal* sample produced by
        `sample_fn` — used only when `reference_fn` is None (legacy path).
      candidate_fn(x) -> tensor: candidate under test.
      sample_fn() -> tensor: draws an N(0,1) input batch.
      reference_fn(x) -> tensor: if provided, used to compute the reference
        for both the normal and outlier-injected inputs (recommended).
      seed: controls the outlier mask RNG.

    Returns a dict with sub-results under "normal" and "outlier" plus a
    top-level "pass" that requires both to pass.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Normal pass.
    x_normal = sample_fn()
    cand_normal = candidate_fn(x_normal)
    ref_normal = reference_fn(x_normal) if reference_fn is not None else reference
    res_normal = check_outputs(ref_normal, cand_normal, dtype=dtype, task=task)

    # Outlier pass. Inject on the CPU sample then move to the candidate's
    # device implicitly via candidate_fn.
    x_outlier = _inject_outliers(
        x_normal, prob=outlier_prob, scale=outlier_scale, generator=g
    )
    cand_outlier = candidate_fn(x_outlier)
    if reference_fn is not None:
        ref_outlier = reference_fn(x_outlier)
    else:
        # Without a reference_fn we can't honestly evaluate the outlier branch.
        ref_outlier = reference
    res_outlier = check_outputs(
        ref_outlier, cand_outlier, dtype=dtype, task=task
    )

    return {
        "pass": bool(res_normal["pass"] and res_outlier["pass"]),
        "normal": res_normal,
        "outlier": res_outlier,
    }
