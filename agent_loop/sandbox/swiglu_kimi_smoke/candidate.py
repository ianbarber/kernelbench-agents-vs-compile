import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), eviction_policy="evict_last")

def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 64
    grid = (n_elements // BLOCK_SIZE,)
    swiglu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out
