# Code review + stats — summary

Code-quality view of all 10 kernels (3 agents × 3 tasks + 1 inductor per task), combining mechanical stats (`stats.py` → `stats.json`/`stats.md`) and a blinded single-reviewer review (`reviewer.py`, reviewer = `claude` CLI, candidates shuffled with a deterministic sha256 seed before being shown).

## Per-task: did the reviewer pick the measured fastest?

| task | reviewer pick | measured fastest | match |
|---|---|---|---|
| swiglu | kimi (103.4 μs, 1.06×) | kimi | ✓ |
| rmsnorm | codex (31.7 μs, 1.13×) | claude (29.7 μs, 1.21×) | ✗ |
| sdpa_prelude | claude (1513 μs, 2.67×) | kimi (1034 μs, 3.91×) | ✗ |

**1 of 3.** Code quality is a weak predictor of measured μs.

Two telling misses:
- **rmsnorm:** reviewer flagged claude's `num_warps=2` as "looks like a guess... expect it to underperform inductor." It was actually the fastest.
- **sdpa_prelude:** reviewer dismissed kimi's `torch.cat([w_q, w_k, w_v])` stack as "bandwidth this design can't afford on Blackwell unified memory." It was 1.5× faster than the cleaner three-GEMMs designs.

## LoC / CC / MI

Apples-to-apples (swiglu, rmsnorm — sdpa is structurally not comparable because agents wrote full single-file modules while the inductor entry is one of six kernels):
- **LoC:** agents 16–109; inductor 31 / 61. codex consistently shortest; kimi consistently longest; inductor mid-pack.
- **MI (maintainability index):** agents 45–65; inductor 41–54. Agents win on MI most of the time, dragged down by inductor's `tmp0..tmpN` names and large `triton_meta`/`inductor_meta` decorators.
- **CC (cyclomatic complexity):** inductor flat 1–3 (straight-line codegen); agents 2–7 with mild host-side branching for shape handling and launcher caching.

## Most striking review notes

- From sdpa review of KERNEL_C (codex, after unblinding): *"Reads like something written under a deadline that happens to pass the harness."* The reviewer is judging intent and maintainability from style cues, not just measuring properties.
- *"I'd ship this"* applied to claude's sdpa_prelude — highest rubric avg (4.4) given to the kernel that benched mid-pack, demonstrating exactly the bench-vs-review gap this stage was designed to expose.

## Surprises about reviewer scoring

- **codex** got the highest rubric avg on rmsnorm (4.2), with claude lowest at 3.4 — and claude actually benched fastest. Terse single-pass code reads as obviously correct; claude's two-pass version (which faithfully clones inductor's structure with smarter eviction hints) was scored lower on performance reasoning but happened to be the fastest in practice.
- **kimi** got the highest avg on swiglu (4.4) and won that benchmark — the only task where review and bench agreed.
- **inductor** scored lowest on every task (2.8, 2.4, 2.2). Reviewer consistently gave it 5 on correctness but 1–2 on readability/length — and on every task identified it as the inductor emission via the decorator dicts and the `cc=121` device-props block.

## CLI style fingerprint (consistent across tasks)

- **codex** — terse, idiomatic, minimal boilerplate. Smallest files. Reviewer prefers it on simpler ops.
- **claude** — middle length, adds host-side machinery (cached launchers, module-level globals). Dinged for over-engineering on swiglu; rewarded for safety on sdpa.
- **kimi** — longest, most-commented, most aggressive design. Hardcodes shape constants. Punished in review for brittleness but won 2 of 3 benchmarks.

## What code review adds beyond the bench numbers

The benchmark says kimi's SDPA prelude is 3.91× faster than inductor's. The code review says a competent reviewer would block it from merging: hardcoded `* 0.0078125`, baked-in `SEQ=512` / `HIDDEN=2048`, and a `torch.cat` of the weight tensors that *looks* like a bandwidth disaster (and that the reviewer wrongly predicted would be one).

For 2 of 3 measured-fastest kernels, the reviewer correctly identified production-blocking issues no benchmark could catch: swiglu/claude's silent output aliasing via a cached `_OUT` global, sdpa/kimi's shape brittleness.

**The μs numbers say "agents can beat inductor"; the reviews say "but you wouldn't ship 5 of the 9 agent kernels without changes." The gap between those two signals — and the fact that the reviewer was wrong about which was fastest in 2/3 cases — is the most interesting finding of the stage.**

## Known limitations

1. Reviewer is `claude`, same model family as one writer. Blinding mitigates but doesn't eliminate the bias.
2. SDPA-prelude inductor entry is one kernel (the dominant `_where_3`), not the full 6-kernel pipeline. Treated as a documented structural mismatch.
3. Single review per task (no resampling). Rubric scores are one model's calibration on a 1–5 scale; rank order is more meaningful than the raw average.
