import torch
import triton
import triton.language as tl

@triton.jit
def sp_kernel(x_ptr, res_ptr, w_ptr, out_ptr, row_stride, eps: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x_row = x_ptr + row * row_stride
    res_row = res_ptr + row * row_stride
    out_row = out_ptr + row * row_stride
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + res
    sq = s * s
    var = tl.sum(sq, axis=0) / float(N)
    inv = tl.rsqrt(var + eps)
    out = s * inv * w
    tl.store(out_row + cols, out.to(tl.bfloat16), mask=mask)

@triton.jit
def sp2_kernel(x_ptr, res_ptr, w_ptr, out_ptr, num_rows, row_stride, eps: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row_start = tl.program_id(0) * 2
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    for r in range(2):
        row = row_start + r
        x_row = x_ptr + row * row_stride
        res_row = res_ptr + row * row_stride
        out_row = out_ptr + row * row_stride
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + res
        sq = s * s
        var = tl.sum(sq, axis=0) / float(N)
        inv = tl.rsqrt(var + eps)
        out = s * inv * w
        tl.store(out_row + cols, out.to(tl.bfloat16), mask=mask)

@triton.jit
def tp_kernel(x_ptr, res_ptr, w_ptr, out_ptr, row_stride, eps: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * row_stride
    res_row = res_ptr + row * row_stride
    out_row = out_ptr + row * row_stride
    acc = 0.0
    for off in range(0, N, BLOCK_N):
        c = cols + off
        m = c < N
        x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
        res = tl.load(res_row + c, mask=m, other=0.0).to(tl.float32)
        s = x + res
        acc += tl.sum(s * s, axis=0)
    var = acc / float(N)
    inv = tl.rsqrt(var + eps)
    for off in range(0, N, BLOCK_N):
        c = cols + off
        m = c < N
        x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
        res = tl.load(res_row + c, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
        s = x + res
        tl.store(out_row + c, (s * inv * w).to(tl.bfloat16), mask=m)

@triton.jit
def tp2_kernel(x_ptr, res_ptr, w_ptr, out_ptr, num_rows, row_stride, eps: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row_start = tl.program_id(0) * 2
    cols = tl.arange(0, BLOCK_N)
    for r in range(2):
        row = row_start + r
        x_row = x_ptr + row * row_stride
        res_row = res_ptr + row * row_stride
        out_row = out_ptr + row * row_stride
        acc = 0.0
        for off in range(0, N, BLOCK_N):
            c = cols + off
            m = c < N
            x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
            res = tl.load(res_row + c, mask=m, other=0.0).to(tl.float32)
            s = x + res
            acc += tl.sum(s * s, axis=0)
        var = acc / float(N)
        inv = tl.rsqrt(var + eps)
        for off in range(0, N, BLOCK_N):
            c = cols + off
            m = c < N
            x = tl.load(x_row + c, mask=m, other=0.0).to(tl.float32)
            res = tl.load(res_row + c, mask=m, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
            s = x + res
            tl.store(out_row + c, (s * inv * w).to(tl.bfloat16), mask=m)

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
    ('sp', 2048, 1, 8, 1),
    ('sp', 2048, 1, 4, 1),
    ('sp', 2048, 1, 16, 1),
    ('sp', 2048, 1, 8, 2),
    ('sp2', 2048, 2, 8, 1),
    ('sp2', 2048, 2, 16, 1),
    ('tp', 1024, 1, 8, 1),
    ('tp', 1024, 1, 4, 1),
    ('tp', 1024, 1, 16, 1),
    ('tp', 1024, 1, 8, 2),
    ('tp', 512, 1, 4, 1),
    ('tp', 512, 1, 8, 1),
    ('tp2', 1024, 2, 8, 1),
    ('tp2', 1024, 2, 16, 1),
    ('tp2', 512, 2, 8, 1),
]

for name, block_n, rows_per_prog, num_warps, num_stages in configs:
    extra_args = ()
    if name == 'sp':
        fn = sp_kernel
        grid = (num_rows,)
    elif name == 'sp2':
        fn = sp2_kernel
        grid = (triton.cdiv(num_rows, rows_per_prog),)
        extra_args = (num_rows,)
    elif name == 'tp':
        fn = tp_kernel
        grid = (num_rows,)
    elif name == 'tp2':
        fn = tp2_kernel
        grid = (triton.cdiv(num_rows, rows_per_prog),)
        extra_args = (num_rows,)

    fn[grid](x_2d, res_2d, w, out, *extra_args, row_stride=row_stride, eps=1e-6, N=N, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages)

    # quick correctness
    s = x_2d.to(torch.float32) + res_2d.to(torch.float32)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + 1e-6)
    ref = (s * inv * w.to(torch.float32)).to(torch.bfloat16)
    diff = (out - ref).abs().max().item()

    us = triton.testing.do_bench(lambda: fn[grid](x_2d, res_2d, w, out, *extra_args, row_stride=row_stride, eps=1e-6, N=N, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages), warmup=10, rep=100, return_mode='median') * 1000
    print(f'{name} BLOCK={block_n} rows={rows_per_prog} warps={num_warps} stages={num_stages}: {us:.2f} us, max_diff={diff}')
