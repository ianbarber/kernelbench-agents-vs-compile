import torch
import triton
import triton.language as tl

SHAPE = (1, 512, 6144)
DTYPE = torch.bfloat16
SEED = 0xC0FFEE

g = torch.Generator(device="cuda")
g.manual_seed(SEED)
x = torch.randn(SHAPE, device="cuda", dtype=DTYPE, generator=g)
y = torch.randn(SHAPE, device="cuda", dtype=DTYPE, generator=g)
n_elements = x.numel()

@triton.jit
def swiglu_kernel_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy="evict_first").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask, eviction_policy="evict_last")

@triton.jit
def swiglu_kernel_nomask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), eviction_policy="evict_last")

out = torch.empty_like(x)

for name, kernel in [("mask", swiglu_kernel_mask), ("nomask", swiglu_kernel_nomask)]:
    grid = (triton.cdiv(n_elements, 64),)
    for _ in range(10):
        kernel[grid](x, y, out, n_elements, BLOCK_SIZE=64, num_warps=4)
    torch.cuda.synchronize()
    t = triton.testing.do_bench(
        lambda: kernel[grid](x, y, out, n_elements, BLOCK_SIZE=64, num_warps=4),
        warmup=25,
        rep=100,
        return_mode="median",
    ) * 1000.0
    print(f"{name}: {t:.2f} us")
