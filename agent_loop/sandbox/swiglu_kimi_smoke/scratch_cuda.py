import torch
from torch.utils.cpp_extension import load_inline

SHAPE = (1, 512, 6144)
DTYPE = torch.bfloat16
SEED = 0xC0FFEE

g = torch.Generator(device="cuda")
g.manual_seed(SEED)
x = torch.randn(SHAPE, device="cuda", dtype=DTYPE, generator=g)
y = torch.randn(SHAPE, device="cuda", dtype=DTYPE, generator=g)
n_elements = x.numel()

cpp_source = """
#include <torch/extension.h>
torch::Tensor run(torch::Tensor x, torch::Tensor y);
"""

cuda_source = """
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ y,
                              __nv_bfloat16* __restrict__ out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int VEC = 8;
    int n_vec = n / VEC;
    for (int i = idx; i < n_vec; i += gridDim.x * blockDim.x) {
        int base = i * VEC;
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
        const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y);
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

        __nv_bfloat162 xv0 = x2[base/2 + 0];
        __nv_bfloat162 xv1 = x2[base/2 + 1];
        __nv_bfloat162 xv2 = x2[base/2 + 2];
        __nv_bfloat162 xv3 = x2[base/2 + 3];

        __nv_bfloat162 yv0 = y2[base/2 + 0];
        __nv_bfloat162 yv1 = y2[base/2 + 1];
        __nv_bfloat162 yv2 = y2[base/2 + 2];
        __nv_bfloat162 yv3 = y2[base/2 + 3];

        float2 fx0 = __bfloat1622float2(xv0);
        float2 fx1 = __bfloat1622float2(xv1);
        float2 fx2 = __bfloat1622float2(xv2);
        float2 fx3 = __bfloat1622float2(xv3);

        float2 fy0 = __bfloat1622float2(yv0);
        float2 fy1 = __bfloat1622float2(yv1);
        float2 fy2 = __bfloat1622float2(yv2);
        float2 fy3 = __bfloat1622float2(yv3);

        float2 fo0, fo1, fo2, fo3;
        fo0.x = silu(fx0.x) * fy0.x; fo0.y = silu(fx0.y) * fy0.y;
        fo1.x = silu(fx1.x) * fy1.x; fo1.y = silu(fx1.y) * fy1.y;
        fo2.x = silu(fx2.x) * fy2.x; fo2.y = silu(fx2.y) * fy2.y;
        fo3.x = silu(fx3.x) * fy3.x; fo3.y = silu(fx3.y) * fy3.y;

        out2[base/2 + 0] = __float22bfloat162_rn(fo0);
        out2[base/2 + 1] = __float22bfloat162_rn(fo1);
        out2[base/2 + 2] = __float22bfloat162_rn(fo2);
        out2[base/2 + 3] = __float22bfloat162_rn(fo3);
    }
}

torch::Tensor run(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    const int VEC = 8;
    int threads = 256;
    int blocks = (n / VEC + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n);
    return out;
}
"""

mod = load_inline(
    name="swiglu_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["run"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

out = mod.run(x, y)
ref = torch.nn.functional.silu(x) * y
print("max diff:", (out - ref).abs().max().item())

import triton.testing
for _ in range(10):
    _ = mod.run(x, y)
torch.cuda.synchronize()
t = triton.testing.do_bench(
    lambda: mod.run(x, y),
    warmup=25,
    rep=100,
    return_mode="median",
) * 1000.0
print(f"CUDA kernel median: {t:.2f} us")
