# Task: Beat inductor's residual-fused RMSNorm kernel

## What

Write a fast GPU kernel that computes **residual-add + RMSNorm** on bf16
tensors, matching the canonical Qwen3-1.7B attention/MLP-block prelude:

```
s    = x + residual
var  = mean(s^2, axis=-1, keepdim=True)
inv  = rsqrt(var + eps)
out  = s * inv * weight
```

All math is done in fp32 internally (loads promote bf16 -> fp32, the
single store demotes fp32 -> bf16). Inductor does exactly this. The
post-add residual `s` lives in registers — it is **not** written back to
memory.

The canonical shapes (from the inductor sample at the
prefill_512_b1 call site) are:

```
x        : (1, 512, 2048)  bf16
residual : (512, 2048)     bf16   (broadcast over the leading singleton)
weight   : (2048,)         bf16
eps      : float scalar, 1e-6
out      : (1, 512, 2048)  bf16
```

Total memory: 4 MiB read + 2 MiB write per call. This op is
**memory-bandwidth-bound**, but unlike SwiGLU it also needs a per-row
reduction over 2048 elements, so it is *less* bandwidth-bound than a pure
pointwise op (the reduction adds compute + cross-lane work).

The eager PyTorch reference is in `reference.py`:

```python
def run(x, residual, weight, eps=1e-6):
    s = x.to(torch.float32) + residual.to(torch.float32)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    return (s * inv * weight.to(torch.float32)).to(x.dtype)
```

## Where to put your code

You must produce a Python module at `candidate.py` in this directory exposing:

```python
def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...
```

Same signature as `reference.run`. Return a bf16 CUDA tensor of the same
shape as `x`.

`x`, `residual`, and `weight` are read-only inputs. **Do not mutate them.**
Your `run` must return a new tensor (or one that doesn't alias any input).

## How to evaluate

Run `python harness.py` from this directory. It will:

1. Build canonical inputs (deterministic seed). `weight` is sampled as
   `0.1 * randn + 1.0` to keep RMSNorm output magnitudes representative
   of trained-model weight statistics.
2. Import your `candidate.run`.
3. Run mutation check: confirms `x`, `residual`, `weight` are bytewise
   unchanged after your call. **Mutation => `FAIL_MUTATION`, run aborts.**
4. Run determinism check: calls `run(...)` twice on identical inputs and
   requires the outputs to match in fp32 to better than 1e-6 RMSE. Any
   non-deterministic kernel => `FAIL_NONDETERMINISTIC`, run aborts.
5. Check correctness vs `reference.run` against **two** tolerance tiers:
   - `standard`: cos_sim >= 0.95, l1_rel <= 0.05, rmse <= 0.10.
   - `strict`  : cos_sim >= 0.99, l1_rel <= 0.01, rmse <= 0.01.
6. If standard passes, benchmark with
   `triton.testing.do_bench(..., return_mode="median")` (25 warmup × 100
   reps) and print median latency in microseconds.
7. Print a JSON summary at the end.

Verdict ladder (highest -> lowest):

- `PASS_STRICT` — passes strict tolerance + non-mutating + deterministic.
- `PASS`        — passes standard tolerance + non-mutating + deterministic.
- `FAIL_MUTATION`
- `FAIL_NONDETERMINISTIC`
- `FAIL_CORRECTNESS`
- `ERROR`

**Aim for `PASS_STRICT`.** Doing the reduction and rsqrt in fp32 (what
inductor does) is well within strict tolerance. Pure-bf16 reductions on
2048 elements drift outside strict — don't do that.

## Target

Inductor's fused kernel
(`triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9`) on
this exact shape/dtype, measured standalone with
`triton.testing.do_bench(..., return_mode="median")`, runs in
**~35.8 microseconds** on this hardware. That is the number to beat.
The harness reports `speedup_vs_inductor = 35.8 / candidate_us`.

For context: the eager PyTorch reference on this hardware measures around
~80-100 us (multiple kernel launches: add, pow, mean, rsqrt, mul, mul,
cast). Inductor's fused kernel is ~2-3x over eager. **Remaining headroom
over inductor is modest** — single-digit microseconds, not 2x.

(Earlier ranking tables cited 258.8 us as the baseline. That number came
from a profiler aggregate over 130 call sites and is **not** the
codegen-vs-codegen baseline. The harness uses the standalone microbench
number, ~35.8 us, as the ground truth.)

## Hardware

NVIDIA GB10 (Blackwell, sm_121), 48 SMs, unified LPDDR5X (~273 GB/s peak).
PyTorch 2.12 nightly cu128, Triton 3.7. This op is bandwidth-bound but
also has a per-row reduction across the hidden dim (2048).

## Allowed approaches

- **Triton** kernels via `triton.jit` — recommended path, matches inductor.
- **Raw CUDA** via `torch.utils.cpp_extension.load_inline` or similar.
- Any other approach that produces a working Python `run(x, residual, weight, eps)`.

## Forbidden

- `torch.compile`, `@torch.compile`, `torch.jit.script`, `torch.jit.trace`.
  We are measuring **your** codegen, not the inductor compiler.
- Calling out to `reference.run` (defeats the purpose).
- Mutating `x`, `residual`, or `weight`.
- Non-deterministic kernels (e.g. atomic adds in non-associative order).

## Profiling tools available

- `triton.testing.do_bench` — preferred microbenchmark.
- `nsys`, `ncu` — system / kernel profilers (slower; use sparingly).
- You may write your own scratch scripts; just make sure `candidate.py` is
  the final artifact.

## Iteration budget

You have up to 5 attempts. After each `python harness.py` run, look at the
`verdict`, `correctness.reasons`, `correctness_strict.reasons`, and the
latency, then try to improve.

Things to think about:

- Block size. Inductor picked `XBLOCK=2, R0_BLOCK=1024, num_warps=8,
  num_stages=1` (so 256 grid programs, each handling 2 rows × 2048 cols).
  That's not necessarily optimal — try wider XBLOCK (more rows per
  program), or a single-pass (no two-pass over R0) layout if you can fit
  the full row in registers (R0=2048 fits comfortably in 8 warps).
- **Single-pass vs two-pass.** Inductor's `_9` is a two-pass reduction
  (load row, accumulate `s^2` sum, then reload row, normalize, store).
  At hidden=2048 the whole row fits in registers — a one-pass layout
  (load row once into a register tile, compute sum cross-lane, broadcast,
  multiply, store) saves ~half the memory traffic on `x` and `residual`.
- fp32 accumulation, bf16 storage. Don't skip the fp32 cast or strict
  tolerance will fail.
- Vectorized loads (load 2 bf16 = 1 int32 = 4 bytes, or wider).
- Number of warps. 8 warps × 32 threads = 256 threads / row of 2048
  means each thread owns 8 elements — fits nicely.

Good luck.
