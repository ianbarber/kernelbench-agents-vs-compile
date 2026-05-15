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
def swiglu_kernel_evict(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, EVICT_X: tl.constexpr, EVICT_Y: tl.constexpr, EVICT_O: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, eviction_policy=EVICT_X).to(tl.float32)
    y = tl.load(y_ptr + offsets, eviction_policy=EVICT_Y).to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), eviction_policy=EVICT_O)

out = torch.empty_like(x)

configs = []
for BLOCK_SIZE in [512, 1024, 2048]:
    for num_warps in [4, 8, 16]:
        for ex in ["evict_first", "normal"]:
            for ey in ["evict_first", "normal"]:
                for eo in ["evict_last", "normal"]:
                    if n_elements % BLOCK_SIZE != 0:
                        continue
                    grid = (n_elements // BLOCK_SIZE,)
                    try:
                        for _ in range(10):
                            swiglu_kernel_evict[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
                                                       EVICT_X=ex, EVICT_Y=ey, EVICT_O=eo)
                        torch.cuda.synchronize()
                        t = triton.testing.do_bench(
                            lambda: swiglu_kernel_evict[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
                                                               EVICT_X=ex, EVICT_Y=ey, EVICT_O=eo),
                            warmup=25,
                            rep=100,
                            return_mode="median",
                        ) * 1000.0
                        configs.append((BLOCK_SIZE, num_warps, ex, ey, eo, t))
                        print(f"BS={BLOCK_SIZE}, warps={num_warps}, ex={ex}, ey={ey}, eo={eo}: {t:.2f} us")
                    except Exception as e:
                        print(f"BS={BLOCK_SIZE}, warps={num_warps}, ex={ex}, ey={ey}, eo={eo}: FAILED {e}")

best = min(configs, key=lambda x: x[5])
print(f"Best: BS={best[0]}, warps={best[1]}, ex={best[2]}, ey={best[3]}, eo={best[4]}, {best[5]:.2f} us")
