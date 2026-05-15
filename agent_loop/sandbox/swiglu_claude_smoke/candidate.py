import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    X_ptr, Y_ptr, OUT_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    x_f = x.to(tl.float32)
    silu = x_f * tl.sigmoid(x_f)
    out = silu * y.to(tl.float32)
    tl.store(OUT_ptr + offs, out.to(tl.bfloat16), mask=mask)


_BLOCK = 1024


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    _swiglu_kernel[grid](
        x, y, out, n,
        BLOCK=_BLOCK,
        num_warps=16,
        num_stages=2,
    )
    return out
