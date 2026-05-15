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

`x` and `y` are read-only inputs. **Do not mutate them.** Your `run` must
return a new tensor (or a tensor that does not alias `x` or `y`).

## How to evaluate

Run `python harness.py` from this directory. It will:

1. Build canonical inputs (deterministic seed).
2. Import your `candidate.run`.
3. Run mutation check: confirms `x` and `y` are bytewise unchanged after
   your call. **Mutating either input => `FAIL_MUTATION`, run aborts.**
4. Run determinism check: calls `run(x, y)` twice on identical inputs and
   requires the outputs to match in fp32 to better than 1e-6 RMSE. Any
   non-deterministic kernel => `FAIL_NONDETERMINISTIC`, run aborts.
5. Check correctness vs `reference.run` against **two** tolerance tiers:
   - `standard`: cos_sim >= 0.95, l1_rel <= 0.05, rmse <= 0.10.
   - `strict`  : cos_sim >= 0.99, l1_rel <= 0.01, rmse <= 0.01.
6. If the standard tier passes, benchmark with
   `triton.testing.do_bench(..., return_mode="median")` and print median
   latency in microseconds.
7. Print a JSON summary at the end.

Verdict ladder (highest -> lowest):

- `PASS_STRICT` — passes strict tolerance + non-mutating + deterministic.
- `PASS`        — passes standard tolerance + non-mutating + deterministic.
- `FAIL_MUTATION`
- `FAIL_NONDETERMINISTIC`
- `FAIL_CORRECTNESS`
- `ERROR`

**Aim for `PASS_STRICT`.** A `PASS` (standard-but-not-strict) implies your
output drifts further from eager than necessary; it is reported but is not
the goal. Approximation tricks (e.g. sigmoid clamping, low-order polynomial
silu approximations, hard-coded sigmoid lookup tables) will typically blow
through strict tolerance — don't use them. Standard silu in fp32 then cast
to bf16 (what inductor does) is well within strict tolerance.

## Target

Inductor's fused kernel (`triton_poi_fused__unsafe_view_mul_silu_6`) on this
exact shape/dtype, measured standalone with `triton.testing.do_bench(...,
return_mode="median")`, runs in **~109.6 microseconds** on this hardware.
That is the number to beat. The harness reports `speedup_vs_inductor =
109.6 / candidate_us`.

For context: the eager PyTorch reference (`silu(x) * y`) on this hardware
measures ~140 microseconds. So inductor's fused kernel is already ~1.3x
over eager. **The remaining headroom over inductor is small** — single-
digit microseconds, not 3x. Don't expect to halve the time; expect to
shave 5-15%.

(Earlier task descriptions cited 361 us as the baseline. That number came
from a profiler aggregate over multiple call sites and is **not** the
codegen-vs-codegen baseline. The harness now uses the standalone
microbench number, ~109.6 us, as the ground truth.)

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
- Mutating `x` or `y`.
- Non-deterministic kernels (e.g. atomic adds in non-associative order).

## Profiling tools available

- `triton.testing.do_bench` — preferred microbenchmark.
- `nsys`, `ncu` — system / kernel profilers (slower; use sparingly).
- You may write your own scratch scripts; just make sure `candidate.py` is the
  final artifact.

## Iteration budget

You have up to 5 attempts. After each `python harness.py` run, look at the
`verdict`, `correctness.reasons`, `correctness_strict.reasons`, and the
latency, then try to improve. Things to think about:

- Block size (XBLOCK in inductor parlance). Inductor used 512 threads, 8 warps,
  1 stage.
- Compute silu in fp32 then cast back (inductor does this; passes strict).
  Doing it fully in bf16 saves nothing on bandwidth-bound ops and risks
  blowing strict tolerance.
- Vectorized loads (e.g. load 4 or 8 bf16 elements per thread via int4 / int8
  loads) to saturate memory bandwidth.
- Number of warps and number of stages.

Good luck.
