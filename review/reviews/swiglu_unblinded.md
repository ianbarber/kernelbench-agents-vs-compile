# Unblinded review key: swiglu

| label | author |
|---|---|
| KERNEL_A | inductor |
| KERNEL_B | kimi |
| KERNEL_C | claude |
| KERNEL_D | codex |

## Rubric scores (transcribed from `swiglu.md`)

| author | correctness | performance | readability | code length | risk | avg |
|---|---|---|---|---|---|---|
| inductor (A) | 4 | 4 | 2 | 2 | 2 | 2.8 |
| kimi (B)     | 5 | 3 | 5 | 5 | 4 | 4.4 |
| claude (C)   | 3 | 4 | 3 | 3 | 2 | 3.0 |
| codex (D)    | 4 | 2 | 5 | 4 | 3 | 3.6 |

## Forced rank

Reviewer picked **KERNEL_B = kimi**.

## Measured

| author | candidate_us | speedup vs inductor |
|---|---|---|
| inductor | 109.6 | 1.00x |
| kimi     | 103.4 | 1.06x |
| codex    | 105.5 | 1.04x |
| claude   | 118.8 | 0.92x |

**Reviewer pick (kimi) matches the measured fastest (kimi).** Hit.

Notable hazards flagged:
- inductor (A): uses `in_out_ptr0` (mutates input) -- would fail this harness's mutation check.
- claude (C): module-level `_OUT` cache returns same tensor across `run()` calls -- aliasing footgun.
- codex (D): import-time hardcoded `_GRID` baked to canonical shape -- silent miscompute on shape change. BLOCK=64 also flagged as wildly under-tiled (reviewer wrong here -- it benched 1.04x in practice).

See `swiglu.md` for the full blinded review text.
