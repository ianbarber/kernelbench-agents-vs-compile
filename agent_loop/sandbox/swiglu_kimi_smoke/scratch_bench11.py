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
def swiglu_kernel_2d(x_ptr, y_ptr, out_ptr, n_rows, VEC: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, VEC)
    x = tl.load(x_ptr + offs_m[:, None] * VEC + offs_n[None, :], eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offs_m[:, None] * VEC + offs_n[None, :], eviction_policy="evict_first").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offs_m[:, None] * VEC + offs_n[None, :], out.to(tl.bfloat16), eviction_policy="evict_last")

out = torch.empty_like(x)

configs = []
for VEC in [2, 4, 8]:
    for BLOCK_M in [64, 128, 256, 512]:
        for num_warps in [2, 4, 8]:
            n_rows = n_elements // VEC
            if n_elements % VEC != 0:
                continue
            grid = (triton.cdiv(n_rows, BLOCK_M),)
            try:
                for _ in range(10):
                    swiglu_kernel_2d[grid](x, y, out, n_rows, VEC=VEC, BLOCK_M=BLOCK_M, num_warps=num_warps)
                torch.cuda.synchronize()
                t = triton.testing.do_bench(
                    lambda: swiglu_kernel_2d[grid](x, y, out, n_rows, VEC=VEC, BLOCK_M=BLOCK_M, num_warps=num_warps),
                    warmup=25,
                    rep=100,
                    return_mode="median",
                ) * 1000.0
                configs.append((VEC, BLOCK_M, num_warps, t))
                print(f"VEC={VEC}, BLOCK_M={BLOCK_M}, warps={num_warps}: {t:.2f} us")
            except Exception as e:
                print(f"VEC={VEC}, BLOCK_M={BLOCK_M}, warps={num_warps}: FAILED {e}")

best = min(configs, key=lambda x: x[3])
print(f"Best: VEC={best[0]}, BLOCK_M={best[1]}, warps={best[2]}, {best[3]:.2f} us")
