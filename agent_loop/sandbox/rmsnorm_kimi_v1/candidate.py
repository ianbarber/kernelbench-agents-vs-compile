"""Fast Triton kernel for fused residual-add + RMSNorm.

Mimics inductor's 2D layout with a single tile (R0_BLOCK=2048)
so the loop over the reduction dimension runs exactly once.
Uses eviction-policy hints matching inductor's codegen.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    row_stride,
    eps: tl.constexpr,
    N: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < num_rows

    r0_base = tl.arange(0, R0_BLOCK)[None, :]

    # --- first pass: accumulate variance ---
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, N, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask_tile = r0_index < N
        tmp0 = tl.load(
            x_ptr + (r0_index + row_stride * xindex),
            r0_mask_tile & xmask,
            eviction_policy='evict_last',
            other=0.0,
        ).to(tl.float32)
        tmp1 = tl.load(
            res_ptr + (r0_index + row_stride * xindex),
            r0_mask_tile & xmask,
            eviction_policy='evict_last',
            other=0.0,
        ).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp2
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask_tile & xmask, tmp7, _tmp6)

    tmp6 = tl.sum(_tmp6, 1)[:, None]

    # --- second pass: write output ---
    for r0_offset in tl.range(0, N, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask_tile = r0_index < N
        tmp8 = tl.load(
            w_ptr + r0_index,
            r0_mask_tile,
            eviction_policy='evict_last',
            other=0.0,
        ).to(tl.float32)
        tmp9 = tl.load(
            x_ptr + (r0_index + row_stride * xindex),
            r0_mask_tile & xmask,
            eviction_policy='evict_first',
            other=0.0,
        ).to(tl.float32)
        tmp10 = tl.load(
            res_ptr + (r0_index + row_stride * xindex),
            r0_mask_tile & xmask,
            eviction_policy='evict_first',
            other=0.0,
        ).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp14 = tmp6 / float(N)
        tmp16 = tmp14 + eps
        tmp17 = tl.math.rsqrt(tmp16)
        tmp18 = tmp11 * tmp17
        tmp20 = tmp8 * tmp18
        tl.store(
            out_ptr + (r0_index + row_stride * xindex),
            tmp20.to(tl.bfloat16),
            r0_mask_tile & xmask,
        )


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    res_2d = residual.reshape(-1, residual.shape[-1])
    out = torch.empty_like(x_2d)

    num_rows, N = x_2d.shape
    row_stride = x_2d.stride(0)

    XBLOCK = 2
    R0_BLOCK = 2048
    grid = (triton.cdiv(num_rows, XBLOCK),)

    _rmsnorm_kernel[grid](
        x_2d,
        res_2d,
        weight,
        out,
        num_rows=num_rows,
        row_stride=row_stride,
        eps=eps,
        N=N,
        XBLOCK=XBLOCK,
        R0_BLOCK=R0_BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out.reshape(x.shape)
