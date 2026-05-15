# Unblinded review key: rmsnorm

| label | author |
|---|---|
| KERNEL_A | kimi |
| KERNEL_B | codex |
| KERNEL_C | claude |
| KERNEL_D | inductor |

## Rubric scores (transcribed from `rmsnorm.md`)

| author | correctness | performance | readability | code length | risk | avg |
|---|---|---|---|---|---|---|
| kimi (A)     | 5 | 3 | 4 | 3 | 4 | 3.8 |
| codex (B)    | 4 | 5 | 5 | 5 | 2 | 4.2 |
| claude (C)   | 4 | 2 | 4 | 4 | 3 | 3.4 |
| inductor (D) | 5 | 3 | 1 | 1 | 2 | 2.4 |

## Forced rank

Reviewer picked **KERNEL_B = codex**.

## Measured

| author | candidate_us | speedup vs inductor |
|---|---|---|
| claude   | 29.7 | 1.21x |
| codex    | 31.7 | 1.13x |
| inductor | 35.8 | 1.00x |
| kimi     | 36.9 | 0.97x |

**Reviewer pick (codex) does NOT match the measured fastest (claude).** Miss.

The reviewer scored claude lowest on performance (num_warps=2 "looks like a guess") and predicted it would underperform inductor -- in fact it was the fastest. Codex's single-pass design got top performance marks; it benched second. The two-pass / shape-brittle distinction the reviewer used to discriminate doesn't perfectly predict actual us on this op.

See `rmsnorm.md` for the full blinded review text.
