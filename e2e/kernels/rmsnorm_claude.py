import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, R_ptr, W_ptr, OUT_ptr,
    xnumel, r0_numel,
    eps,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex

    acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        m = r0_mask & xmask
        off = r0_index + r0_numel * x0
        a = tl.load(X_ptr + off, m, eviction_policy='evict_last', other=0.0).to(tl.float32)
        b = tl.load(R_ptr + off, m, eviction_policy='evict_last', other=0.0).to(tl.float32)
        s = a + b
        sq = s * s
        acc = tl.where(m, acc + sq, acc)
    sum_sq = tl.sum(acc, 1)[:, None]
    inv = tl.rsqrt(sum_sq * (1.0 / r0_numel) + eps)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        m = r0_mask & xmask
        off = r0_index + r0_numel * x0
        w = tl.load(W_ptr + r0_index, r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        a = tl.load(X_ptr + off, m, eviction_policy='evict_first', other=0.0).to(tl.float32)
        b = tl.load(R_ptr + off, m, eviction_policy='evict_first', other=0.0).to(tl.float32)
        s = a + b
        out = s * inv * w
        tl.store(OUT_ptr + off, out.to(OUT_ptr.dtype.element_ty), m)


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.shape[-1]
    M = x.numel() // N
    grid = (M,)
    _rmsnorm_kernel[grid](
        x, residual, weight, out,
        M, N, eps,
        XBLOCK=1, R0_BLOCK=1024,
        num_warps=2, num_stages=1,
    )
    return out
