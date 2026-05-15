# Blinded review: rmsnorm

Reviewer: claude (CLI) -- same model family as one writer (known limitation).
Prompt: `review/reviews/rmsnorm_prompt.txt`

---

## KERNEL_A

| Dimension | Score | Note |
|---|---|---|
| Correctness | 5 | fp32 promotion on every load, fp32 accumulator, explicit `to(tl.bfloat16)` on the single store. |
| Performance | 3 | Faithful two-pass clone of inductor — second pass reloads `x` and `residual`, doubling read traffic on the heavy tensors. R0_BLOCK=2048 at least makes the inner loop trivial. |
| Readability | 4 | Inherits inductor's `tmp0..tmp20` naming, but the two-pass split is clearly labeled and the structure reads cleanly. |
| Code length | 3 | The masking and `tl.range` loop are vestigial for the canonical shape (everything divides), but they buy generality cheaply. |
| Risk | 4 | Properly masked, handles non-canonical shapes, no private inductor imports. Safe but unlikely to beat target meaningfully. |

This is essentially inductor-rewritten-by-hand with the same two-pass shape; correct and robust, but it doesn't claim the obvious headroom (single-pass).

## KERNEL_B

| Dimension | Score | Note |
|---|---|---|
| Correctness | 4 | fp32 reduction and rsqrt are right, implicit bf16 cast on store via `out_ptr.dtype` is fine. Hard-coded `1.0/2048.0` is correct here but bakes in a shape assumption that isn't asserted. |
| Performance | 5 | Single-pass: `s = x + r` lives in registers, reduce, broadcast, multiply, store. Roughly half the DRAM traffic of A/C/D on the heavy tensors. BLOCK_M=4 × 8 warps on a 2048-wide row is well sized for sm_121. |
| Readability | 5 | Tight, idiomatic, no boilerplate — math reads top to bottom. |
| Code length | 5 | About as short as a correct version of this op can be. |
| Risk | 2 | No masking, grid hard-coded to 128, divisor hard-coded to 2048, BLOCK_N==H assumed. If anyone calls with seq_len ≠ 512 or H ≠ 2048 it silently produces garbage. No `assert` on shapes. |

The fastest design in the lineup, and the only one that exploits the "row fits in registers" observation the task explicitly flagged. Shape brittleness is the only thing holding it back from a clean ship.

## KERNEL_C

| Dimension | Score | Note |
|---|---|---|
| Correctness | 4 | fp32 throughout, divisor uses runtime `r0_numel` (more general than B). Output cast via `OUT_ptr.dtype.element_ty` is clean. |
| Performance | 2 | `num_warps=2` for a 2048-wide reduction is undersized — each warp owns 1024 elements, leaving cross-lane reduction and DRAM concurrency on the table. XBLOCK=1 also means 512 grid programs with no per-row amortization. Two-pass on top of that. |
| Readability | 4 | Concise, sensible names, clear two-pass structure. |
| Code length | 4 | About right. |
| Risk | 3 | Functionally correct on arbitrary shapes, but the tuning choices look like a guess. Likely slower than the inductor baseline, not faster. |

Cleanest of the two-pass implementations, but the launch config is the wrong shape for this hardware/op; expect it to underperform inductor.

## KERNEL_D

| Dimension | Score | Note |
|---|---|---|
| Correctness | 5 | Literal inductor emission — ground truth by construction, including redundant `.to(fp32)` no-ops. |
| Performance | 3 | This *is* the ~35.8µs baseline the task is trying to beat. Two-pass with R0_BLOCK chosen by inductor's autotuner. |
| Readability | 1 | Drowning in `triton_meta` / `inductor_meta` dicts, tmp-numbered names, dead variables (`roffset`, `rbase`, `r0_1`). |
| Code length | 1 | Vastly over-engineered for human consumption; ~half the lines are decorator metadata. |
| Risk | 2 | Imports from `torch._inductor.runtime.*` — private API, will break across PyTorch versions. Also note: this file defines the kernel but **no `run()` wrapper**, so it doesn't satisfy the task contract as written. |

Clearly the inductor-emitted reference, not a hand-written candidate. Useful as ground truth; not shippable as-is.

## Forced rank: ship one

**KERNEL_B.** It's the only candidate that actually capitalizes on the headroom the task description points at — single-pass over `x` and `residual` with the row resident in registers — and on this exact bandwidth-bound shape that's where the wins live. A is the safer fallback, but for a *fused prelude kernel pinned to one model's hidden size* the brittleness is acceptable if I add three `assert`s on shapes before launch. I would not ship D (private inductor imports + no `run`) and would not ship C (num_warps=2 looks like a misconfiguration).
