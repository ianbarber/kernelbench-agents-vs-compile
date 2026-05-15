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
def swiglu_kernel_fp32(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy="evict_first").to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask, eviction_policy="evict_last")

out = torch.empty_like(x)
grid = (triton.cdiv(n_elements, 64),)
swiglu_kernel_fp32[grid](x, y, out, n_elements, BLOCK_SIZE=64, num_warps=4)
ref = torch.nn.functional.silu(x) * y

out_f = out.float()
ref_f = ref.float()
cos_sim = (out_f * ref_f).sum() / (out_f.norm() * ref_f.norm())
print("cos_sim (float):", cos_sim.item())
print("cos_sim (bf16):", torch.nn.functional.cosine_similarity(out.view(-1), ref.view(-1), dim=0).item())
print("max diff:", (out - ref).abs().max().item())
