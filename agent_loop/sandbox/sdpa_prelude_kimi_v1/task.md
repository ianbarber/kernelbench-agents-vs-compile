# Task: Beat inductor's SDPA-prelude (Q/K/V + RoPE + GQA-expand + mask)

## What

Write a fast GPU kernel (or composition of kernels) that produces the
four tensors the next `scaled_dot_product_attention` call consumes,
starting from the post-residual-RMSNorm `hidden_states` of a Qwen3-1.7B
attention block:

```
                                    +-----------------------+
hidden_states  --(W_Q proj)----+--> | per-head RMSNorm + RoPE | --> q  (1, 16, 512, 128)
                               |    +-----------------------+
                               |    +-----------------------+
               --(W_K proj)----+--> | per-head RMSNorm + RoPE | --> k  (1, 16, 512, 128)
                               |    | + GQA expand (8 -> 16)  |
                               |    +-----------------------+
                               |    +-----------------------+
               --(W_V proj)----+--> | GQA expand (8 -> 16)    | --> v  (1, 16, 512, 128)
                               |    +-----------------------+
                                    +-----------------------+
attention_mask  --------------+---> | causal + padding mask   | --> mask (1, 1, 512, 512)
                              |     +-----------------------+
position_ids ---inv_freq------+ (feeds RoPE)
```

This is everything *between* the post-RMSNorm `hidden_states` and the
inputs to `aten._scaled_dot_product_efficient_attention`. **Note**:
this task is **not** about attention itself — the attention kernel is
out of scope (and disallowed; see "Forbidden" below).

In inductor's output for prefill_512_b1, the prelude is:

  * 3 × `extern_kernels.mm` (cuBLAS Q/K/V projections)
  * `triton_per_fused..._1` — Q per-head RMSNorm + RoPE
  * `triton_per_fused..._2` — K per-head RMSNorm + RoPE
  * `triton_poi_fused..._where_3` — KV GQA-expansion (called twice, K & V)
  * `triton_poi_fused..._where_4` — causal+padding mask construction

The headline kernel `_where_3` consumes **24.99% of prefill_512_b1
walltime** (across all call sites). The full prelude is roughly **40%**
of prefill if you include the GEMMs and the other Triton kernels.

The eager PyTorch reference is in `reference.py`.

## Where to put your code

Produce `candidate.py` in this directory, exposing:

```python
import torch
from typing import Tuple

def run(
    hidden_states: torch.Tensor,   # (1, 512, 2048) bf16
    w_q: torch.Tensor,             # (2048, 2048)   bf16   F.linear style: out = h @ w.T
    w_k: torch.Tensor,             # (1024, 2048)   bf16
    w_v: torch.Tensor,             # (1024, 2048)   bf16
    w_q_norm: torch.Tensor,        # (128,)         bf16   per-head RMSNorm scale
    w_k_norm: torch.Tensor,        # (128,)         bf16
    inv_freq: torch.Tensor,        # (64,)          fp32   RoPE inverse frequencies
    position_ids: torch.Tensor,    # (1, 512)       int64
    attention_mask: torch.Tensor,  # (1, 512)       int64  1=keep, 0=pad
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      q     (1, 16, 512, 128) bf16
      k     (1, 16, 512, 128) bf16    GQA-expanded
      v     (1, 16, 512, 128) bf16    GQA-expanded
      mask  (1,  1, 512, 512) bf16    additive (0 / -inf), causal + padding
    """
```

**All inputs are read-only.** Do not mutate any of them. Return new
tensors (or tensors that do not alias inputs).

## Canonical input shapes (prefill_512_b1, Qwen3-1.7B)

| name              | shape              | dtype  |
|-------------------|--------------------|--------|
| hidden_states     | (1, 512, 2048)     | bf16   |
| w_q               | (2048, 2048)       | bf16   |
| w_k               | (1024, 2048)       | bf16   |
| w_v               | (1024, 2048)       | bf16   |
| w_q_norm          | (128,)             | bf16   |
| w_k_norm          | (128,)             | bf16   |
| inv_freq          | (64,)              | fp32   |
| position_ids      | (1, 512)           | int64  |
| attention_mask    | (1, 512)           | int64  |

## Algorithm details

1. **QKV projection.** Standard `F.linear`-style: each output is
   `hidden_states @ w.T`. The reference uses `torch.as_strided` to
   give cuBLAS the fast B-transposed view (a naive `h @ w.T` hits a
   ~5× slower algo on this hardware — see CHOICE.md for the trick).

2. **Reshape to (B, H, S, D).** Q-proj output `(B, S, 2048)` → view
   `(B, S, 16, 128)` → transpose to `(B, 16, S, 128)`. Same for K
   `(B, 8, S, 128)` and V `(B, 8, S, 128)`.

3. **Per-head RMSNorm.** For Q and K (NOT V), normalise over the
   head_dim=128 axis:
   ```
   var = x.pow(2).mean(dim=-1, keepdim=True)     # fp32
   out = x * rsqrt(var + eps) * weight            # weight is (128,)
   ```
   All math in fp32, cast to bf16 on store. Matches inductor's
   `per_fused..._1` / `_2`.

4. **RoPE construction & apply.** Build cos/sin tables from
   `position_ids` and `inv_freq`:
   ```
   freqs = position_ids[..., None].float() * inv_freq[None]   # (B, S, 64)
   emb   = cat([freqs, freqs], dim=-1)                        # (B, S, 128)
   cos   = emb.cos()         # bf16, broadcast (B, 1, S, D) over heads
   sin   = emb.sin()
   ```
   Apply to Q and K:
   ```
   rotate_half(x) = cat([-x[..., D//2:], x[..., :D//2]], dim=-1)
   x_rot          = x * cos + rotate_half(x) * sin
   ```

5. **GQA expansion (K & V).** From `(B, 8, S, 128)` to `(B, 16, S, 128)`
   by replicating each KV head over `groups = 16/8 = 2` consecutive
   Q-head positions. Inductor uses `head // 2` indexing — the natural
   `repeat_interleave(2, dim=1)`.

6. **Causal + padding mask.** Shape `(B, 1, S, S)` bf16:
   ```
   mask[b, 0, q, k] = 0.0 if (k <= q) and (attention_mask[b, k] != 0)
                    else float("-inf")
   ```

## How to evaluate

Run `python harness.py` from this directory. The harness:

1. Builds canonical inputs (deterministic seed).
2. Imports your `candidate.run`.
3. **Mutation check**: confirms every input tensor is bytewise
   unchanged. `FAIL_MUTATION` aborts the run.
4. **Determinism check**: calls `run(...)` twice on identical inputs
   and requires every output to match in fp32 to better than 1e-6
   RMSE. `FAIL_NONDETERMINISTIC` aborts.
5. **Correctness check vs `reference.run`** against two tolerance
   tiers, **per output tensor**:
   - `standard`: cos_sim ≥ 0.95, l1_rel ≤ 0.05, rmse ≤ 0.10
   - `strict`  : cos_sim ≥ 0.99, l1_rel ≤ 0.01, rmse ≤ 0.01

   The mask tensor has `-inf` entries; the harness internally replaces
   `-inf` with `-1e4` for the correctness reduction (so cos_sim stays
   finite). The candidate must still produce real `-inf` entries
   (anything below ~-1e3 in the mask will be visually equivalent
   to "-inf" for SDPA's softmax).
6. If standard passes, benchmark with
   `triton.testing.do_bench(..., return_mode="median")` (25 warmup,
   100 reps) and print median latency in microseconds.
7. Prints a JSON summary.

Verdict ladder (highest → lowest):

- `PASS_STRICT` — passes strict tolerance on all 4 outputs + non-mutating + deterministic.
- `PASS`        — passes standard tolerance + non-mutating + deterministic.
- `FAIL_MUTATION`
- `FAIL_NONDETERMINISTIC`
- `FAIL_CORRECTNESS`
- `ERROR`

**Aim for `PASS_STRICT`.** Doing the reductions and rsqrt in fp32 (what
inductor does) is well within strict tolerance. bf16-internal
reductions on the 128-element head_dim are tight but usually OK; bf16
reductions on longer axes will drift.

## Target

The codegen-vs-codegen baseline — inductor's prelude as a sequential
chain of 3 cuBLAS GEMMs + 5 Triton kernels — measured standalone with
`triton.testing.do_bench(..., return_mode="median")` runs in
**~4046 µs** on this hardware. That is the number to beat. The harness
reports `speedup_vs_inductor = 4045.73 / candidate_us`.

Per-component breakdown of the baseline (microbench median µs):

| component         | µs       | notes                                |
|-------------------|---------:|--------------------------------------|
| q_proj_mm         |   610    | cuBLAS, (512×2048) @ (2048×2048)     |
| k_proj_mm         |   776    | cuBLAS, (512×2048) @ (2048×1024)     |
| v_proj_mm         |  1539    | cuBLAS, same shape as k              |
| q_rmsnorm_rope    |    30    | inductor's `per_fused..._1`          |
| k_rmsnorm_rope    |    18    | inductor's `per_fused..._2`          |
| kv_expand_k       |    24    | inductor's `where_3` on K            |
| kv_expand_v       |    24    | inductor's `where_3` on V            |
| causal_mask       |     9    | inductor's `where_4`                 |
| **full chain**    | **4046** |                                      |

The GEMMs dominate — most of your improvement headroom is in **reducing
the work the GEMMs have to do** (e.g. fused QKV mm, or skipping the
GQA-expansion materialisation entirely if you eventually go for a
fused-attention design — see "Allowed approaches" below).

Eager reference latency on this hardware: **~2000-4500 µs** (highly
variable across runs — observed median ~2-4.5 ms in do_bench). The
eager path uses identical cuBLAS calls plus PyTorch primitives for
RMSNorm/RoPE/expand/mask, so most of its time is in the same three
GEMMs. The variability comes from cuBLAS algo selection and the
unified-memory residency state — don't read too much into any single
number. What matters: even the naive eager reference is in the same
ballpark as inductor's emitted chain, so the floor for a useful
speedup is not very high. **A reasonable Triton-fused RoPE+RMSNorm
that drops the GQA materialisation entirely should clear 3000 µs**.

## Stop-early policy

You have up to **5 attempts** at `python harness.py`. **As soon as you
get a `PASS_STRICT` that beats `~4000 µs`, stop and submit.** Don't
spend further attempts grinding for the last 5% — the comparison this
task feeds into cares much more about whether you reached `PASS_STRICT`
under-baseline than whether you reached the absolute hardware limit.

## Forbidden

- `torch.compile`, `@torch.compile`, `torch.jit.script`, `torch.jit.trace`.
  We are measuring **your** codegen, not the inductor compiler.

- **Any** scaled-dot-product-attention op or backend. Specifically:
    - `torch.nn.functional.scaled_dot_product_attention`
    - `torch.ops.aten._scaled_dot_product_efficient_attention`
    - `torch.ops.aten._scaled_dot_product_flash_attention`
    - `torch.ops.aten.scaled_dot_product_attention`
    - cuDNN flash attention / `_efficient_attention_forward`

  These are not what this task measures and would defeat the
  codegen-vs-codegen comparison. The harness scans `candidate.py`
  textually for these tokens and refuses to run if any are present.

- Calling out to `reference.run` (defeats the purpose).
- Mutating any of the input tensors.
- Non-deterministic kernels (e.g. atomic adds in non-associative order).

## Allowed approaches

- **Triton** kernels via `triton.jit` — the recommended path, matches
  inductor's structure. You can write one big Triton kernel that does
  RMSNorm + RoPE + GQA expand for Q (and similarly for K, V), or split
  into multiple smaller kernels.

- **cuBLAS via `torch.mm`** (with the `as_strided` trick, see
  reference.py) — this is what inductor does for the projections.
  Writing a custom GEMM that beats cuBLAS at (512×2048)@(2048×2048) bf16
  on Blackwell is unlikely to succeed; we recommend keeping the
  projection GEMMs as cuBLAS calls.

- **Raw CUDA** via `torch.utils.cpp_extension.load_inline` or similar.

- **Flash-Attention-style absorbed-prelude designs** — this is the
  biggest potential win on paper. The idea: instead of materialising
  Q, K, V, mask as separate tensors that get passed to SDPA, write your
  own fused kernel that **computes attention directly from the
  hidden_states + weights** and never materialises the GQA-expanded
  K/V or the (S × S) mask. **However**: if you go that route, you must
  NOT use any `scaled_dot_product_attention` op family (see Forbidden).
  You'd have to write the attention math yourself. **This is harder
  than it sounds** — and you still need to produce `(q, k, v, mask)`
  as the task contract, so you'd be writing fused attention *and* also
  exposing the four output tensors. Unless you have a clear plan, stay
  with the simpler decomposition.

## Hardware context

NVIDIA GB10 (Blackwell, sm_121), 48 SMs, unified LPDDR5X (~273 GB/s
peak). PyTorch 2.12 nightly cu128, Triton 3.7. The GEMMs are
compute-bound (this hardware's bf16 throughput); the Triton kernels are
all bandwidth-bound.

## Profiling tools available

- `triton.testing.do_bench` — preferred microbenchmark.
- `nsys`, `ncu` — system / kernel profilers (slower; use sparingly).
- You may write your own scratch scripts; just make sure `candidate.py`
  is the final artifact in this directory.

Good luck.
