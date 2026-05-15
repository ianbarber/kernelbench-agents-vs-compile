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
def swiglu_kernel_vec(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC
    for v in range(VEC):
        offsets = block_start + v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        silu = x * tl.sigmoid(x)
        out = silu * y
        tl.store(out_ptr + offsets, out.to(tl.bfloat16), eviction_policy="evict_last")

out = torch.empty_like(x)

configs = []
for BLOCK_SIZE in [64, 128, 256]:
    for VEC in [2, 4, 8, 16]:
        for num_warps in [1, 2, 4, 8, 16]:
            total_per_block = BLOCK_SIZE * VEC
            if n_elements % total_per_block != 0:
                continue
            grid = (n_elements // total_per_block,)
            try:
                for _ in range(10):
                    swiglu_kernel_vec[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, num_warps=num_warps)
                torch.cuda.synchronize()
                t = triton.testing.do_bench(
                    lambda: swiglu_kernel_vec[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, num_warps=num_warps),
                    warmup=25,
                    rep=100,
                    return_mode="median",
                ) * 1000.0
                configs.append((BLOCK_SIZE, VEC, num_warps, t))
                print(f"BS={BLOCK_SIZE}, VEC={VEC}, warps={num_warps}: {t:.2f} us")
            except Exception as e:
                print(f"BS={BLOCK_SIZE}, VEC={VEC}, warps={num_warps}: FAILED {e}")

best = min(configs, key=lambda x: x[3])
print(f"Best: BS={best[0]}, VEC={best[1]}, warps={best[2]}, {best[3]:.2f} us")
