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
def swiglu_kernel_nomask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    y = tl.load(y_ptr + offsets).to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16))

@triton.jit
def swiglu_kernel_cg(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    y = tl.load(y_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), cache_modifier=".cg")

@triton.jit
def swiglu_kernel_cs(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    y = tl.load(y_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), cache_modifier=".cs")

out = torch.empty_like(x)

configs = []
for kernel, name in [(swiglu_kernel_nomask, "nomask"), (swiglu_kernel_cg, "cg"), (swiglu_kernel_cs, "cs")]:
    for BLOCK_SIZE in [1024, 2048, 4096, 8192]:
        for num_warps in [4, 8, 16]:
            if n_elements % BLOCK_SIZE != 0:
                continue
            grid = (n_elements // BLOCK_SIZE,)
            try:
                for _ in range(10):
                    kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
                torch.cuda.synchronize()
                t = triton.testing.do_bench(
                    lambda: kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps),
                    warmup=25,
                    rep=100,
                    return_mode="median",
                ) * 1000.0
                configs.append((name, BLOCK_SIZE, num_warps, t))
                print(f"{name} BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}: {t:.2f} us")
            except Exception as e:
                print(f"{name} BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}: FAILED {e}")

best = min(configs, key=lambda x: x[3])
print(f"Best: {best[0]} BLOCK_SIZE={best[1]}, num_warps={best[2]}, {best[3]:.2f} us")
