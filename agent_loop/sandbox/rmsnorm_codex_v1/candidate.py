from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_residual_2d(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * BLOCK_N + cols

    s = tl.load(x_ptr + offsets, eviction_policy="evict_last").to(tl.float32)
    s += tl.load(residual_ptr + offsets, eviction_policy="evict_last").to(tl.float32)

    ss = tl.sum(s * s, 1)[:, None]
    inv = tl.rsqrt(ss * (1.0 / 2048.0) + eps)
    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    tl.store(out_ptr + offsets, s * inv * weight)


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _rmsnorm_residual_2d[(128,)](
        x,
        residual,
        weight,
        out,
        eps,
        BLOCK_M=4,
        BLOCK_N=2048,
        num_warps=8,
        num_stages=1,
    )
    return out
