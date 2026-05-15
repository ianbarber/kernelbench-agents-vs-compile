# SwiGLU smoke — three-CLI comparison

Task: `agent_loop/tasks/swiglu` (bf16, shape `(1, 512, 6144)`).
All three runs used `--max-attempts 5` on the same DGX Spark (GB10, sm_121).

## Baselines for comparison

| Baseline                                 | Median us | Notes |
|------------------------------------------|----------:|-------|
| Eager reference (`F.silu(x) * y`)        |   ~141    | unfused; alloc + two passes |
| Inductor profiler-aggregate (decode/prefill mix) | 361.3 | originally used as headline — overstated, includes warm-up + launch jitter across 280 invocations in a full inference |
| **Inductor standalone microbench**       | **~109.6** | `extract/microbench_inductor.py` — kernel run via `do_bench` with the inductor-emitted XBLOCK=512 / num_warps=8 / num_stages=1. **This is the honest "what inductor's kernel does in isolation" number.** |

So the inductor *kernel* is already faster than eager (~109.6 vs ~141 us). The 361 us profiler number is a workload-level number, not a kernel-level one — it includes context, repeated launches in non-ideal cache state, and is the *wrong* number to quote as the inductor floor.

## Result table (tightened harness)

Harness now reports: `verdict` (PASS_STRICT > PASS > FAIL_MUTATION > FAIL_NONDETERMINISTIC > FAIL_CORRECTNESS > ERROR), `pass_standard`, `pass_strict`, `mutates_input`, `deterministic`. See `agent_loop/tasks/swiglu/harness.py`.

| CLI    | verdict        | pass_std | pass_strict | mutates_input | cand_us | inductor-microbench us | speedup_vs_microbench | speedup_vs_profiler_361us | vs eager (~141 us) |
|--------|----------------|:--------:|:-----------:|:-------------:|--------:|----------------------:|----------------------:|--------------------------:|-------------------:|
| claude | **PASS_STRICT** |   ✓     |     ✓       |       ✗       |  121.79 |                109.58 |             **0.90x** |                     2.97x |             1.16x  |
| codex  | **FAIL_MUTATION** |  ✓     |     ✗       |     ✓ (y)     |     —   |                109.58 |                   —   |                       —   |               —   |
| kimi   | **PASS_STRICT** |   ✓     |     ✓       |       ✗       |  103.42 |                109.58 |             **1.06x** |                     3.49x |             1.36x  |

(Numbers from `agent_loop/runs/swiglu_*_smoke/result_strict.json`.)

### Per-CLI notes

- **claude**: Exact fp32 sigmoid, `BLOCK=1024`, `num_warps=16`, `num_stages=2`. Non-mutating, deterministic, passes the strict tier (`cos_sim=0.99999587`, `l1_rel=0.0015`, `rmse=0.0017`). About 11% slower than inductor's standalone kernel.
- **codex**: Piecewise-linear sigmoid (`clip(x*0.21+0.5,0,1)`) + writes the output into `y` instead of allocating. The mutation makes it a **contract violation** — the reference doesn't write into `y`, so any downstream code that read `y` after this call would see corrupted values. Under the tightened harness this is a hard fail (`verdict: FAIL_MUTATION`, exit 3). For information: the candidate's output snapshot also fails strict tolerance (`l1_rel=0.044`, `rmse=0.035` — both above the strict `0.01` threshold), confirming the sigmoid approximation was the source of the previous "won the timing race" outcome.
- **kimi**: Exact fp32 sigmoid, tiny `BLOCK_SIZE=64`, `num_warps=4`, explicit eviction-policy hints. Non-mutating, deterministic, passes strict. The only candidate that **beat the inductor standalone kernel** (1.06x). Comes from going against the conventional "larger block" wisdom on this hardware.

## What changed in the methodology

1. **Mutation check.** Before running the candidate, we clone `x` and `y` to `x_orig`, `y_orig`; after the call, `torch.equal(x, x_orig)` and `torch.equal(y, y_orig)` must both hold. Mutating the input tensors is a contract violation — the reference doesn't mutate, and any caller passing in a shared buffer would be corrupted. This is now a hard fail.
2. **Strict-tier reporting.** In addition to the standard KBX tolerance (`cos_sim ≥ 0.95`, `l1_rel ≤ 0.05`, `rmse ≤ 0.10`), we now also evaluate the strict tier (`0.99 / 0.01 / 0.01`) and surface it as `correctness_strict`. Passing both promotes the verdict to `PASS_STRICT`. This catches sigmoid-approximation tricks that scrape past the loose bar.
3. **Determinism check.** We call the candidate twice on identical inputs and require the outputs to agree to better than `rmse < 1e-6` (or be bytewise equal). Non-deterministic kernels fail with `FAIL_NONDETERMINISTIC` (exit 4).
4. **Inductor floor is now the standalone microbench, not the profiler-aggregate.** `extract/microbench_inductor.py` launches inductor's own emitted Triton function (stripped of the `triton_heuristics.pointwise` autotune wrapper — that wrapper rejects the `XBLOCK` kwarg in our torch 2.12 nightly + triton 3.7 combo) under `triton.testing.do_bench`. Median is ~109.6 us at XBLOCK=512, num_warps=8, num_stages=1, matching the metadata the workload profiler recorded. The earlier 361 us "baseline" was a workload-aggregate, not a like-for-like microbench, and was a substantially unfair point of comparison.

## Reshaped headline

Under the old methodology: "All three CLIs beat inductor by 3-3.4x on SwiGLU."

Under the corrected methodology:

- **Codex's win is disqualified.** Its kernel mutated its inputs *and* could not have passed a strict-tolerance bar. The "skip-the-alloc + approximate-sigmoid" optimisation was buying speed by violating two contracts at once. The win is real *only if* you accept both, and we no longer do.
- **Inductor's actual kernel is essentially competitive.** The kernel itself is ~110 us; eager is ~141 us. Inductor is ~1.3x faster than eager just by fusing the silu and the multiply (no allocation, single pass over memory).
- **Kimi narrowly beats inductor at ~1.06x.** Real, modest, and strictly-correct — about 6 us out of 110.
- **Claude is slightly behind inductor at ~0.90x.** Honest, deterministic, strict-correct kernel, but slightly slower than the inductor-tuned config.
- The 3-3.4x speedup numbers from the original write-up were measured against the wrong baseline. Against the honest inductor microbench, the spread is 0.90x-1.06x — i.e. the agents are doing roughly *as well as* inductor on this op, not 3x better.

## Wrapper / auth notes

Unchanged from the previous run. `kimi info`/`codex --version` both worked; no wrapper modifications were made.

## Artifacts

- claude: `agent_loop/runs/swiglu_claude_smoke/` — `result.json` (old harness), `result_strict.json` (new harness). Candidate at `agent_loop/sandbox/swiglu_claude_smoke/candidate.py`.
- codex:  `agent_loop/runs/swiglu_codex_smoke/` — same layout. Candidate mutates `y` (contract violation).
- kimi:   `agent_loop/runs/swiglu_kimi_smoke/` — same layout.
- harness: `agent_loop/tasks/swiglu/harness.py` (and synced copies in each sandbox).
- inductor microbench: `extract/microbench_inductor.py` and `extract/microbench_inductor.json`.
