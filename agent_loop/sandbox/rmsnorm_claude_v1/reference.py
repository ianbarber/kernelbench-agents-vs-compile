"""Eager PyTorch reference for the residual-add + RMSNorm fused op.

This matches inductor's `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9`,
which fuses a single residual-add into the RMSNorm preamble:

    s = (x + residual).float()          # bf16 -> fp32 for the reduction
    var = s.pow(2).mean(-1, keepdim=True)
    out = (s * rsqrt(var + eps)) * weight.float()
    return out.to(x.dtype)              # bf16

Notes on the contract:
- The residual-add result is consumed by the rsqrt branch only -- inductor does
  NOT write the post-add residual back to memory (it lives in registers).
  So `run` returns a single tensor: the RMSNorm output.
- `x`, `residual`, and `weight` are read-only inputs; we MUST NOT mutate them.
- All arithmetic is done in fp32 internally to match inductor's `to(fp32)`
  conversion (the .to(fp32) -> .to(fp32) -> mul -> store is what the kernel
  emits), then cast back to bf16 on the store.
"""
from __future__ import annotations

import torch

EPS_DEFAULT = 1e-6


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS_DEFAULT,
) -> torch.Tensor:
    """Fused residual-add + RMSNorm.

    Args:
        x: bf16 CUDA tensor, shape (..., H).
        residual: bf16 CUDA tensor, broadcastable to x along the leading
            dims (in the canonical inductor case, x is (1, 512, H) and
            residual is (512, H)).
        weight: bf16 CUDA tensor, shape (H,).
        eps: small float for numerical stability inside rsqrt.

    Returns:
        bf16 CUDA tensor of the same shape as x.
    """
    out_dtype = x.dtype
    # Promote to fp32 for the reduction and the rsqrt -- matches inductor's
    # .to(tl.float32) on every load.
    s = x.to(torch.float32) + residual.to(torch.float32)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    out = s * inv * weight.to(torch.float32)
    return out.to(out_dtype)
