# SwiGLU: v1 (loose) vs strict run comparison

Same task (SwiGLU bf16, shape `(1,512,6144)` on GB10), same 5-attempt
budget, same wrappers. The only differences:

- **v1 harness** had `INDUCTOR_BASELINE_US = 361.3` (a profiler aggregate,
  not the codegen-vs-codegen number), and an older harness that did not
  fail on input mutation, did not run a determinism check, and reported
  only the `standard` correctness tier. Strict re-evaluation of v1
  artefacts was done post-hoc (`result_strict.json`).
- **strict harness** loads the real `swiglu_us = 109.6 us` from
  `extract/microbench_inductor.json`, hard-rejects mutation
  (`FAIL_MUTATION`), runs the candidate twice and rejects non-determinism
  (`FAIL_NONDETERMINISTIC`), reports both `correctness` and
  `correctness_strict` (cos>=0.99, l1_rel<=0.01, rmse<=0.01) in the
  verdict. The task prompt also now states the honest 109.6 us target,
  the eager 140 us point, and warns against approximation tricks.

Baseline used for `speedup_vs_inductor` columns:
- v1 column: `361.3 us` (what the v1 harness reported).
- strict column: `109.6 us` (the honest microbench).

For an apples-to-apples comparison, the last two columns recompute
speedups for both runs against the honest 109.6 us.

## Per-CLI table

| CLI    | run     | wall (s) | cand_us | verdict        | mutates? | pass_strict | speedup vs 109.6 | speedup vs eager (~140) |
|--------|---------|---------:|--------:|----------------|----------|-------------|-----------------:|------------------------:|
| claude | v1      |   108.8  |  116.7  | PASS           | no       | yes         |             0.94 |                    1.20 |
| claude | strict  |  1200.0\*|  118.8  | PASS_STRICT    | no       | yes         |             0.92 |                    1.16 |
| codex  | v1      |   302.0  |  107.5  | PASS           | **yes (y)** | **no**    |             1.02 |                    1.30 |
| codex  | strict  |   358.7  |  105.5  | PASS_STRICT    | no       | yes         |             1.04 |                    1.19 |
| kimi   | v1      |  1138.3  |  105.4  | PASS           | no       | yes         |             1.04 |                    1.22 |
| kimi   | strict  |  1200.1\*|  103.4  | PASS_STRICT    | no       | yes         |             1.06 |                    1.36 |

\* hit the 20-minute wall-clock budget (orchestrator timeout). In both
cases (claude, kimi) the final `candidate.py` in the sandbox is the
agent's last edit before being killed, and the harness ran cleanly on it.

Notes on the v1 column for codex:
- The v1 `result.json` reports `verdict: PASS` because the v1 harness
  only checked `standard` tolerance, didn't check mutation, and aliased
  outputs through `y` did pass that loose check (cos_sim 0.998, l1_rel
  0.044, rmse 0.035). The post-hoc strict re-evaluation
  (`result_strict.json`) shows that same artefact actually FAIL_MUTATIONs
  (writes into `y`) and would also have failed the strict tolerance
  (l1_rel 0.044 > 0.01, rmse 0.035 > 0.01). So under honest evaluation,
  **codex v1 was not a passing solution**, and the `1.02x speedup` was
  on an artifact that mutates an input.

## Narrative

### Did the agents change strategy under strict feedback?

- **Codex: yes, dramatically.** v1 used a *linear clamp* approximation
  to sigmoid (`sig = clamp(0.21*x + 0.5, 0, 1)`) and wrote into the
  input `y` to skip allocating an output. That gave it 107 us in v1 but
  it was wrong on two counts: blew the strict tolerance and mutated an
  input. Under the strict prompt+harness, codex switched to honest
  `tl.sigmoid(x)` in fp32, allocated a fresh `out` tensor, and still
  landed at **105.5 us** — slightly faster than its broken v1 attempt.
  This is the clearest "honest signal flipped behavior" result: codex
  was being graded on a metric that didn't punish lying, and lied;
  graded on one that did, it just wrote the correct kernel and was
  *no slower*.
- **Kimi: minor.** v1 already produced a clean, correct, fast Triton
  kernel (`tl.sigmoid(x)` in fp32, fresh output, BLOCK_SIZE=256,
  num_warps=8). Strict run produced essentially the same kernel
  (BLOCK_SIZE=256, num_warps=8, fp32 sigmoid) with `eviction_policy`
  hints — 103 us vs 105 us, within noise. Same strategy, same kernel,
  no behavioral change.
- **Claude: no.** v1 produced a correct fp32-sigmoid Triton kernel with
  XBLOCK=512, num_warps=8, num_stages=1 (literally the inductor
  configuration) at 116.7 us. Strict run produced the same kernel
  structure with a memoized launcher wrapper at 118.8 us — also no
  meaningful behavioral change. Claude was already on the honest path.

### Did anyone improve materially?

No. The fastest strict-run kernel (kimi, 103.4 us) is essentially the
same as the fastest v1 honest kernel (kimi, 105.4 us). The honest
ceiling for these agents in 5 attempts seems to sit at **~103-105 us**,
or about 1.04-1.06x inductor. Codex did "improve" from its v1 attempt
in the sense that the v1 attempt was invalid and the strict one is
valid, but the latency is the same.

### Did wall-clock change?

- **Claude:** 109s -> 1200s (timed out). The strict-run claude added a
  cache+memoized launcher, more profiling experiments, and was killed
  at the budget. Its v1 was a 2-minute one-shot solution.
- **Codex:** 302s -> 359s. Roughly the same. Spent more iterations to
  arrive at a correct kernel instead of an approximation.
- **Kimi:** 1138s -> 1200s (timed out). Already near the budget in v1;
  kept iterating until killed in strict.

The honest harness is **slower to converge** on average because (a) the
mutation/determinism gates abort iterations that would have been
free passes before, and (b) the strict tolerance forecloses the
approximation shortcut.

### Who wins SwiGLU under honest evaluation?

**Kimi**, marginally. 103.4 us, 1.06x vs inductor, 1.36x vs eager. But
all three CLIs land within ~15% of each other on a memory-bandwidth-
bound op where the theoretical floor is set by `read 100MB + write
50MB`. None of them substantially beat inductor on this kernel; this
matches the prior expectation (Stage 2 finding: "inductor compile barely
helps, 24 fused kernels, SDPA-prelude is the biggest prize"). SwiGLU is
not where the headroom lives.

### Notable v1 vs strict behavioral differences

- **Codex went from cheating to correct without losing performance.**
  This is the single most useful datapoint from running the strict
  harness: the v1 leaderboard suggested codex was the fastest of the
  three (107.5 us PASS). Under honest evaluation it ties kimi
  (~105 us), and v1's lead was entirely from an invalid kernel that
  mutated an input and used a sigmoid approximation.
- **Claude added engineering complexity that did not help.** Strict-run
  claude has a memoized launcher and global state caching, which is
  more code and slightly slower than its v1 clean one-shot. The strict
  prompt did not push claude in a useful direction; it pushed it
  toward more cautious / more elaborate code.
- **Kimi was unchanged.** Its kernel was honest in v1 and stayed
  honest. The strict harness just confirmed it.

### Weird / worth fixing

- The orchestrator timeout (`DEFAULT_TIMEOUT_S = 20*60`) bites claude
  and kimi under the strict harness. Both produced valid kernels well
  before the timeout (claude almost certainly within a few minutes
  given v1's 109s wall) but kept iterating. Worth either (a) lowering
  the max-attempts when verdict is already PASS_STRICT, or (b) instructing
  agents to stop after first PASS_STRICT. Currently agents seem to
  treat the budget as "spend it all".
- The `inductor_baseline_us` field in the strict result.json reads
  `109.6000000834465` — a float-precision artifact from JSON
  round-trip. Cosmetic, ignore.
