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
def swiglu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)

out = torch.empty_like(x)

configs = []
for BLOCK_SIZE in [1024, 2048, 4096, 8192]:
    for num_warps in [4, 8, 16]:
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        try:
            # warmup
            for _ in range(10):
                swiglu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
            torch.cuda.synchronize()
            t = triton.testing.do_bench(
                lambda: swiglu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps),
                warmup=25,
                rep=100,
                return_mode="median",
            ) * 1000.0
            configs.append((BLOCK_SIZE, num_warps, t))
            print(f"BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}: {t:.2f} us")
        except Exception as e:
            print(f"BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}: FAILED {e}")

best = min(configs, key=lambda x: x[2])
print(f"Best: BLOCK_SIZE={best[0]}, num_warps={best[1]}, {best[2]:.2f} us")
