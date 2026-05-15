# Mechanical stats: 10 kernels

LoC = non-blank, non-comment-only. CC = cyclomatic complexity (sum / max). 
MI = maintainability index (0-100, higher better). bench is candidate_us median.


## swiglu

| author | LoC | bytes | CC (sum/max/blocks) | MI | jit fns | loads | stores | where | sum | dot | bench (us) | speedup vs inductor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude | 40 | 1359 | 6/4/3 | 59.1 | 1 | 2 | 1 | 0 | 0 | 0 | 118.8 | 0.92x |
| codex | 16 | 686 | 2/1/2 | 58.8 | 1 | 2 | 1 | 0 | 0 | 0 | 105.5 | 1.04x |
| kimi | 28 | 1061 | 5/4/2 | 65.1 | 1 | 2 | 1 | 0 | 0 | 0 | 103.4 | 1.06x |
| inductor | 31 | 2533 | 1/1/1 | 54.1 | 1 | 2 | 1 | 0 | 0 | 0 | 109.6 | 1.00x |

## rmsnorm

| author | LoC | bytes | CC (sum/max/blocks) | MI | jit fns | loads | stores | where | sum | dot | bench (us) | speedup vs inductor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude | 57 | 1986 | 4/3/2 | 45.0 | 1 | 5 | 1 | 1 | 1 | 0 | 29.7 | 1.21x |
| codex | 42 | 1169 | 2/1/2 | 54.3 | 1 | 3 | 1 | 0 | 1 | 0 | 31.7 | 1.13x |
| kimi | 109 | 3366 | 4/3/2 | 57.7 | 1 | 5 | 1 | 1 | 1 | 0 | 36.9 | 0.97x |
| inductor | 61 | 4326 | 3/3/1 | 41.8 | 1 | 5 | 1 | 1 | 1 | 0 | 35.8 | 1.00x |

## sdpa_prelude

| author | LoC | bytes | CC (sum/max/blocks) | MI | jit fns | loads | stores | where | sum | dot | bench (us) | speedup vs inductor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude | 192 | 8493 | 7/2/5 | 49.7 | 4 | 14 | 6 | 1 | 4 | 0 | 1513.5 | 2.67x |
| codex | 117 | 4558 | 4/1/4 | 43.0 | 3 | 14 | 6 | 5 | 2 | 0 | 1515.5 | 2.67x |
| kimi | 266 | 8484 | 5/1/5 | 44.1 | 4 | 18 | 6 | 1 | 4 | 0 | 1034.2 | 3.91x |
| inductor | 26 | 2728 | 1/1/1 | 53.3 | 1 | 1 | 1 | 0 | 0 | 0 | 4045.7 | 1.00x |

---

Notes:
- Inductor SDPA prelude is split across ~6 kernels (3 mm + 2 norm/RoPE + 2 GQA-expand + 1 mask). 
  We show the dominant single kernel here (`_where_3`, the GQA expand, 25% of prefill walltime), 
  but the speedup column compares against the *full* prelude microbench (4045.73 us).
- Agent SDPA candidates are single-file modules: they include host setup + multiple `@triton.jit` 
  functions, so direct LoC/CC comparison to a single inductor kernel is unfair on both sides. 
  Read the agent vs inductor SDPA rows as a documented mismatch, not apples-to-apples.
- SwiGLU / RMSNorm comparisons are fair: 1 kernel per file, similar boilerplate.
