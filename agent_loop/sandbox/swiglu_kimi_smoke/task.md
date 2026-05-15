# Task: Beat inductor's SwiGLU kernel

## What

Write a fast GPU kernel that computes the SwiGLU element-wise op on bf16
tensors:

```
out = silu(x) * y         # silu(x) = x / (1 + exp(-x))
```

The canonical input shape is `(1, 512, 6144)` bf16 (the prefill_512_b1 shape
from the Qwen3-1.7B workload). Total elements: 25,165,824. Memory footprint:
2 inputs * 25.2M * 2 bytes ~= 100 MB in, 50 MB out. This op is bandwidth-bound.

The eager PyTorch reference is provided in `reference.py`:

```python
import torch
def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) * y
```

## Where to put your code

You must produce a Python module at `candidate.py` in this directory exposing:

```python
def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...
```

Same signature as `reference.run`. `x` and `y` are bf16 CUDA tensors of shape
`(1, 512, 6144)`. Return a bf16 CUDA tensor of the same shape.

## How to evaluate

Run `python harness.py` from this directory. It will:

1. Build canonical inputs (deterministic seed).
2. Import your `candidate.run`.
3. Check correctness vs `reference.run` (KernelBenchX `standard` thresholds:
   cos_sim >= 0.95, l1_rel <= 0.05, rmse <= 0.10).
4. If correct, run `triton.testing.do_bench(..., return_mode="median")` and
   print median latency in microseconds.
5. Print a JSON summary at the end.

The harness exits 0 on correctness pass, 2 on correctness fail, 1 on other
errors.

## Target

Inductor's fused kernel (`triton_poi_fused__unsafe_view_mul_silu_6`) measured
~361 microseconds mean from profiler aggregation on this hardware. Your goal is
to **beat that latency** while passing correctness. The actual do_bench median
may differ — the harness prints both your latency and the eager reference's
latency for context.

## Hardware

NVIDIA GB10 (Blackwell, sm_121), 48 SMs, unified LPDDR5X. PyTorch 2.12 nightly
cu128, Triton 3.7. This op is **memory-bandwidth-bound**; the theoretical
lower bound is set by reading 100 MB and writing 50 MB.

## Allowed approaches

- **Triton** kernels via `triton.jit` — recommended path, matches what inductor
  produced.
- **Raw CUDA** via `torch.utils.cpp_extension.load_inline` or similar.
- Any other approach that produces a working Python `run(x, y)` function.

## Forbidden

- `torch.compile`, `@torch.compile`, `torch.jit.script`, `torch.jit.trace`. We
  are measuring **your** codegen, not the inductor compiler.
- Calling out to `reference.run` (defeats the purpose).

## Profiling tools available

- `triton.testing.do_bench` — preferred microbenchmark.
- `nsys`, `ncu` — system / kernel profilers (slower; use sparingly).
- You may write your own scratch scripts; just make sure `candidate.py` is the
  final artifact.

## Iteration budget

You have up to 5 attempts. After each `python harness.py` run, look at the
correctness reasons (if any) and the latency, and try to improve. Things to
think about:

- Block size (XBLOCK in inductor parlance). Inductor used 512 threads, 8 warps,
  1 stage.
- Whether to do the silu in fp32 then cast back (inductor does) or fully in
  bf16 (less precise; might fail correctness).
- Vectorized loads (e.g. load 4 or 8 bf16 elements per thread via int4 / int8
  loads) to saturate memory bandwidth.
- Number of warps and number of stages.

Good luck.
