# Blinded review: sdpa_prelude

Reviewer: claude (CLI) -- same model family as one writer (known limitation).
Prompt: `review/reviews/sdpa_prelude_prompt.txt`

---

## KERNEL_A

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 4 | fp32 reductions/RoPE, correct rotate_half. Concat is read-only of inputs, so no mutation. |
| Performance reasoning | 2 | `torch.cat([w_q,w_k,w_v])` allocates ~16MB and a memcpy on every call — that alone likely eats the QKV-fusion win. One program per (b,s,h) leaves 8192 tiny programs with a 128-element reduction. |
| Readability | 4 | Clear kernel boundaries, descriptive arg names, one job per kernel. |
| Code length | 3 | Reasonable, but four kernels where one could plausibly bundle Q & K. |
| Risk | 2 | The fused-QKV claim is misleading — the cat is pure overhead, not a fusion. Stride args for `q_in_stride_h` are passed as the constant `HEAD_DIM` rather than `q_flat.stride(...)`, which is correct here but brittle if layout ever changes. |

Rationale: math is right and the kernels are easy to verify, but the headline optimization (stacked QKV mm) is implemented as a runtime `torch.cat` that materialises a fresh 4096×2048 weight tensor every call. That's bandwidth this design can't afford on Blackwell unified memory.

## KERNEL_B

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 5 | fp32 everywhere it matters, RoPE built inline (no cos/sin materialisation), GQA write via `static_range` is unambiguous. |
| Performance reasoning | 4 | BLOCK_S=16 amortises launch overhead and gives a 16×64 inner tile; reasonable warp count; only three Triton kernels post-GEMM. K/V outputs are written in expanded form directly, skipping the `repeat_interleave` materialisation. |
| Readability | 5 | Tile dims are `tl.constexpr`, masks are explicit, the two Q/K kernels parallel each other so diffs are obvious. |
| Code length | 4 | A bit long because Q and K kernels duplicate structure rather than templating, but each is easy to read independently. |
| Risk | 4 | I'd ship this. Main concern is the strided S-dim load pattern (stride H*D in S, contiguous in D) — coalescing is fine for D=128 but the strided outer dim wastes some L2. |

Rationale: this is the textbook decomposition — three cuBLAS GEMMs, three fused Triton kernels (Q, K-expanded, V-expanded), one mask. It avoids the GQA materialisation entirely by writing directly into the expanded layout. Predictable, debuggable, and the constexpr-heavy signature means good autotuning behaviour.

## KERNEL_C

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 3 | Math checks out (the `where(offs<64,...)` trick for rotate_half is clever), but Q is returned as a **non-contiguous transpose view** of `q_flat`. Strict-tolerance harness reductions won't care, but any downstream consumer expecting contiguous layout will silently pay for it. |
| Performance reasoning | 4 | Fuses K and V RMSNorm/RoPE/expand into a single kernel (one read of K, one read of V, two writes each). Q stays in its native (S, H*D) layout so no transpose/copy. Hardcoded `* 0.0078125` saves a register but is fragile. |
| Readability | 2 | No comments, magic numbers everywhere (`512`, `128`, `0.0078125`, `& 63`, `* 8`), and the in-place Q rewrite is easy to misread as a bug. |
| Code length | 5 | Shortest of the three by a wide margin — terse but not undercoded. |
| Risk | 2 | Hardcoded `SEQ=512`/`HIDDEN=2048`/`* 0.0078125` means any shape change is a silent miscompute. The transposed-view Q output is a footgun for callers. Aggressive but I wouldn't deploy without a shape guard. |

Rationale: clever and compact — the K+V fused kernel and the in-place Q transform are real wins — but the "works only at the canonical config" hardcoding and the strided-Q return value make it a maintenance hazard. Reads like something written under a deadline that happens to pass the harness.

## KERNEL_D

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 5 | It's the inductor-emitted GQA-expand kernel — single load, single store, indexing arithmetic is mechanical. |
| Performance reasoning | 3 | Pure memory copy, well-tuned by inductor's autotuner, but it's only one of six kernels needed for the full prelude. As a standalone submission for *this task* it doesn't produce the four output tensors. |
| Readability | 1 | Generated code: 60-word kernel name, full `triton_meta`/`inductor_meta` dicts, single-letter variable names with no comments. Not meant to be read. |
| Code length | 1 | There is no `run` function. It doesn't satisfy the task contract. |
| Risk | 1 | Cannot ship — it's an artifact, not a solution. As an inductor fragment it's fine; as a candidate it's incomplete. |

Rationale: this is plainly the inductor-emitted `where_3` GQA-expand kernel — the giveaways are the `triton_heuristics.pointwise` decorator, the `DeviceProperties(cc=121)` block, and the 60-word fused-op name. Judged against the task, it's not a candidate at all; judged against itself, it's just a 4-line indexing copy.

## Forced rank: ship one

**KERNEL_B.** It's the only one I'd put in front of a code reviewer without flinching: explicit fp32 promotion, no input mutation, no hardcoded shape constants, three clean kernels that map 1:1 to the algorithm description, and it skips the GQA `repeat_interleave` materialisation by writing into the expanded layout directly. KERNEL_A pays for a needless `torch.cat` of the weights on every call, KERNEL_C is faster-looking but riddled with magic numbers and returns a strided Q that will bite a future caller, and KERNEL_D isn't a complete submission. B is the boring, correct choice — which is the right choice for production.
