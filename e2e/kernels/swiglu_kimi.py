import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets, eviction_policy='evict_first').to(tl.float32)
    y = tl.load(y_ptr + offsets, eviction_policy='evict_first').to(tl.float32)

    # silu(x) = x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)
    out = silu_x * y

    tl.store(out_ptr + offsets, out.to(tl.bfloat16), eviction_policy='evict_first')


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16
    assert x.shape == y.shape

    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = (n_elements // BLOCK_SIZE,)
    swiglu_kernel[grid](
        x, y, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    return out
