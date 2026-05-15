# Blinded review: swiglu

Reviewer: claude (CLI) -- same model family as one writer (known limitation).
Prompt: `review/reviews/swiglu_prompt.txt`

---

## KERNEL_A

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 4 | fp32 silu, cast back to bf16 — math is fine, but it writes through `in_out_ptr0`, i.e. mutates one of the inputs. |
| Performance reasoning | 4 | This is the baseline being targeted (~109.6 µs); standard inductor tiling, no vectorization. |
| Readability | 2 | Buried under `triton_heuristics.pointwise` metadata; kernel body is tiny once you find it. |
| Code length | 2 | Inductor codegen — heavily annotated but no `run` wrapper, so it isn't even a usable `candidate.py` as shown. |
| Risk | 2 | `in_out_ptr0` mutation pattern would fail this harness's mutation check, and is a footgun anywhere the caller still needs `x`. |

This is clearly the inductor emission. The math is unimpeachable, but as a standalone artifact it is dangerous: it expects a wrapper that owns the buffer, and trips the mutation guard if dropped in as-is.

## KERNEL_B

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 5 | fp32 compute via `tl.sigmoid`, explicit `.to(tl.bfloat16)` on store — easy to read, easy to trust. |
| Performance reasoning | 3 | BLOCK=256 with 8 warps is 1 elem/thread and no vectorization on a BW-bound op; probably leaves bandwidth on the table. |
| Readability | 5 | Plain, idiomatic Triton; one screen, no magic. |
| Code length | 5 | Right-sized. |
| Risk | 4 | No mask, relies on `n_elements % BLOCK_SIZE == 0`; trivially fixable, but a production version should mask. |

Cleanest of the four. Performance choice (256/8w) is a bit odd — 8 warps per 256 elements means most threads will be idle on vectorized lanes — but correctness and safety are solid.

## KERNEL_C

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 3 | Math is right, but module-level `_OUT` caching means consecutive `run()` calls return the *same* tensor — outputs alias. |
| Performance reasoning | 4 | XBLOCK=512, 8 warps, 1 stage matches inductor's geometry; launcher-lambda trick shaves a bit of Python overhead. |
| Readability | 3 | The cached-launcher dance and three module-level globals obscure a kernel that should be ~10 lines. |
| Code length | 3 | Over-engineered for an elementwise op. |
| Risk | 2 | Returning a cached buffer is a serious aliasing hazard: any caller that stores prior outputs gets silently clobbered, and the harness's determinism check could pass for the wrong reason (same pointer, same bytes). |

The micro-optimizations are real but the cached-output design is a production-grade footgun. Also: a launch lambda doesn't beat Triton's own launch cache by much.

## KERNEL_D

| Dimension | Score | Note |
|---|---|---|
| Correctness reasoning | 4 | Math is right; relies on Triton's implicit fp32→bf16 cast on store rather than being explicit. |
| Performance reasoning | 2 | BLOCK=64 with 4 warps is wildly under-tiled — 393K program instances, way too much launch/scheduling overhead for a BW-bound kernel. |
| Readability | 5 | Shortest and most readable of the bunch. |
| Code length | 4 | Concise, but the hardcoded `_GRID = (1*512*6144//_BLOCK,)` at import time bakes the canonical shape into the module. |
| Risk | 3 | Silently wrong on any other shape; tiny BLOCK will be noticeably slow even when it is correct. |

Reads beautifully but the tile size is the wrong answer for the question being asked, and the import-time grid means a shape change is a silent miscompute, not a crash.

## Forced rank: ship one

**KERNEL_B.** It's the only one with no built-in footgun: it doesn't mutate inputs (A), doesn't alias its own outputs across calls (C), and isn't hardcoded to a single shape (D). The performance tuning is suboptimal — I'd want to revisit BLOCK/num_warps and add vectorized loads — but those are local tweaks to a fundamentally sound, readable kernel, whereas the other three each have a structural problem that a code review should reject before perf even comes up.
