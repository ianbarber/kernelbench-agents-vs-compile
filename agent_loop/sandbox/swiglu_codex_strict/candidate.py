import torch
import triton
import triton.language as tl


_BLOCK = 64
_GRID = (1 * 512 * 6144 // _BLOCK,)


@triton.jit
def _swiglu_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(out_ptr + offsets, out, eviction_policy="evict_last")


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _swiglu_kernel[_GRID](x, y, out, BLOCK=_BLOCK, num_warps=4, num_stages=1)
    return out
