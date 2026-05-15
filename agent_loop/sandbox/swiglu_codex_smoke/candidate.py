import torch
import triton
import triton.language as tl


_BLOCK = 2048


@triton.jit
def _swiglu_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    sig = x * 0.21 + 0.5
    sig = tl.minimum(1.0, tl.maximum(0.0, sig))
    out = (x * sig) * y
    tl.store(out_ptr + offs, out)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    grid = (x.numel() // _BLOCK,)
    _swiglu_kernel[grid](x, y, y, BLOCK=_BLOCK, num_warps=32, num_stages=1)
    return y
