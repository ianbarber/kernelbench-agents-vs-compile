import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, res_ptr, w_ptr, out_ptr, num_rows, row_stride, eps: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < num_rows
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, N, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask_tile = r0_index < N
        tmp0 = tl.load(x_ptr + (r0_index + row_stride * xindex), r0_mask_tile & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(res_ptr + (r0_index + row_stride * xindex), r0_mask_tile & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp2
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask_tile & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, N, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask_tile = r0_index < N
        tmp8 = tl.load(w_ptr + r0_index, r0_mask_tile, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(x_ptr + (r0_index + row_stride * xindex), r0_mask_tile & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(res_ptr + (r0_index + row_stride * xindex), r0_mask_tile & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp14 = tmp6 / float(N)
        tmp16 = tmp14 + eps
        tmp17 = tl.rsqrt(tmp16)
        tmp18 = tmp11 * tmp17
        tmp20 = tmp8 * tmp18
        tl.store(out_ptr + (r0_index + row_stride * xindex), tmp20.to(tl.bfloat16), r0_mask_tile & xmask)

seed = 0xC0FFEE
g = torch.Generator(device='cuda')
g.manual_seed(seed)
x = torch.randn((1, 512, 2048), device='cuda', dtype=torch.bfloat16, generator=g)
res = torch.randn((512, 2048), device='cuda', dtype=torch.bfloat16, generator=g)
w = (torch.randn((2048,), device='cuda', dtype=torch.float32, generator=g) * 0.1 + 1.0).to(torch.bfloat16)

x_2d = x.reshape(-1, x.shape[-1])
res_2d = res.reshape(-1, res.shape[-1])
out = torch.empty_like(x_2d)
num_rows, N = x_2d.shape
row_stride = x_2d.stride(0)

configs = [
    (2, 1024, 8),
    (2, 2048, 8),
    (4, 1024, 8),
    (4, 2048, 8),
    (8, 1024, 8),
    (8, 2048, 8),
    (16, 1024, 8),
    (16, 2048, 8),
    (8, 2048, 4),
    (8, 2048, 16),
    (8, 2048, 32),
    (16, 2048, 4),
    (16, 2048, 16),
]

for xblock, r0_block, num_warps in configs:
    grid = (triton.cdiv(num_rows, xblock),)
    kernel[grid](x_2d, res_2d, w, out, num_rows, row_stride, eps=1e-6, N=N, XBLOCK=xblock, R0_BLOCK=r0_block, num_warps=num_warps, num_stages=1)
    
    ref = torch.rms_norm(x_2d.to(torch.float32) + res_2d.to(torch.float32), (2048,), w.to(torch.float32), 1e-6).to(torch.bfloat16)
    diff = (out - ref).abs().max().item()
    
    us = triton.testing.do_bench(lambda: kernel[grid](x_2d, res_2d, w, out, num_rows, row_stride, eps=1e-6, N=N, XBLOCK=xblock, R0_BLOCK=r0_block, num_warps=num_warps, num_stages=1), warmup=10, rep=100, return_mode='median') * 1000
    print(f'XBLOCK={xblock} R0_BLOCK={r0_block} warps={num_warps}: {us:.2f} us, max_diff={diff}')
