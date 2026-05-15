# Unblinded review key: sdpa_prelude

| label | author |
|---|---|
| KERNEL_A | kimi |
| KERNEL_B | claude |
| KERNEL_C | codex |
| KERNEL_D | inductor |

## Rubric scores (transcribed from `sdpa_prelude.md`)

| author | correctness | performance | readability | code length | risk | avg |
|---|---|---|---|---|---|---|
| kimi (A)     | 4 | 2 | 4 | 3 | 2 | 3.0 |
| claude (B)   | 5 | 4 | 5 | 4 | 4 | 4.4 |
| codex (C)    | 3 | 4 | 2 | 5 | 2 | 3.2 |
| inductor (D) | 5 | 3 | 1 | 1 | 1 | 2.2 |

## Forced rank

Reviewer picked **KERNEL_B = claude**.

## Measured (full prelude microbench: 4045.7us inductor)

| author | candidate_us | speedup vs inductor |
|---|---|---|
| kimi     | 1034.2 | 3.91x |
| claude   | 1513.5 | 2.67x |
| codex    | 1515.5 | 2.67x |
| inductor | 4045.7 | 1.00x |

**Reviewer pick (claude) does NOT match the measured fastest (kimi).** Miss.

The reviewer flagged kimi's `torch.cat([w_q, w_k, w_v])` as "bandwidth this design can't afford" -- in fact it's the fastest by a wide margin (1.5x ahead of claude/codex). The cat happens once per call but is presumably amortised, and the fused matmul wins more than the cat costs. Conversely, claude's "textbook decomposition" (three cuBLAS GEMMs + three Triton kernels) reads cleanly but is materially slower.

The inductor entry (D) is the GQA-expand kernel only -- not a full submission. The reviewer correctly identified it as inductor codegen and flagged it as not shippable.

See `sdpa_prelude.md` for the full blinded review text.
