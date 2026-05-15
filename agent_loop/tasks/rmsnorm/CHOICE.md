# RMSNorm task: kernel-variant choice

## Picked

**`triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9`**

## Why this one

Of the ~13 RMSNorm-shaped variants inductor emitted for Qwen3-1.7B, this is
the **simplest and the largest single-residual** variant by walltime share:

| variant | workload | walltime (ms) | %total | mean us | residual adds | num inputs |
|---|---|---:|---:|---:|---:|---:|
| `_11` | prefill_512_b1 | 33.77 | 1.89% | 259.7 | 3 | 5 ptrs |
| **`_9`**  | **prefill_512_b1** | **33.65** | **1.88%** | **258.8** | **1** | **3 ptrs** |
| `_13` | prefill_512_b1 |  3.75 | 0.21% | 375.3 | – | – |
| `_14` | decode_ctx512_b1 | 11.60 | 6.16% | 89.2 | 4 | 5 ptrs |
| `_12` | decode_ctx512_b1 | 11.58 | 6.15% | 89.0 | 3 | 5 ptrs |

`_11` edges out `_9` by 0.3% on prefill walltime, but `_9` has a much cleaner
signature (one residual, three pointers). It's the **same op family** every
attention/MLP block uses, just with fewer residuals fused in. For a
"single concept" agent task we want the version that most cleanly isolates
*residual-add + RMSNorm* without dragging in extra accumulation paths.

The decode variants (`_14`, `_12`) have higher percentage share within
decode (~6%), but decode is a small workload overall (188 ms total vs
1785 ms prefill); in absolute walltime prefill `_9` is 3x larger
(33.6 vs 11.6 ms). So prefill `_9` wins on absolute impact too.

## Shape

```
x        : (1, 512, 2048)  bf16
residual : (512, 2048)     bf16   (broadcast over leading singleton)
weight   : (2048,)         bf16
out      : (1, 512, 2048)  bf16
```

Total bytes (bf16, 2B each):
- read:  `x` 2 MiB + `residual` 2 MiB + `weight` 4 KiB  ~= 4 MiB
- write: `out` 2 MiB
- bandwidth bound: 6 MiB / 35.8 us ~= 175 GB/s effective
  (LPDDR5X peak is ~273 GB/s, so inductor is ~64% of peak)

## Residual structure

`_9` performs `x + residual` first, then RMSNorms the post-add tensor
with `weight`. The post-add residual is **not** written back to memory —
it lives in registers between the two passes (reduction pass, then a
broadcast-divide pass that re-recomputes `x + residual` and stores
only the normalized output).

```
s         = x + residual                # bf16 -> fp32 internally
var       = s.pow(2).mean(-1)
inv       = rsqrt(var + 1e-6)
out_bf16  = (s * inv * weight).to(bf16)
```

So the task contract is: `run(x, residual, weight, eps) -> out`. A single
output tensor, post-add NOT exposed. This matches inductor's actual
behavior — anything else would change the function contract.

## Standalone microbench

Run with `extract/microbench_inductor.py`, after stripping the
`@triton_heuristics.reduction(...)` autotune wrapper.

- **median_us: 35.84** (`do_bench`, 25 warmup × 100 reps)
- profiler-aggregate (across 130 call sites in prefill): 258.8 us — this
  is the number that shows up in `ranking.md` but is NOT the
  codegen-vs-codegen baseline.
- launch config inductor picked: `XBLOCK=2, R0_BLOCK=1024, num_warps=8,
  num_stages=1`. 256 grid programs (512 rows / 2 rows per program).
- Sanity-check vs eager fp32 reference passes strict tolerance:
  cos_sim ~1.0, rmse 6.6e-6, l1_rel 1.4e-8.
