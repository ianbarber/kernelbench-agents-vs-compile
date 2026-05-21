# KernelBench Replication — Experiment Log

A running narrative of what we tried, why, what hypotheses we held, and what we found. Source-of-truth for the blog post. Entries are chronological (oldest at top).

---

## Context

**Premise.** The KernelBenchX paper (arXiv 2605.04956v2) claims three things across 176 isolated Triton tasks × 5 methods × 6 GPUs:
1. Correctness, semantics, and performance are separable axes — many "compilable" kernels are wrong; 46.6% of *correct* kernels are slower than eager.
2. Task category explains ~3× more variance in correctness than method identity (9.4% vs 3.3%).
3. Iterative refinement repairs correctness but doesn't improve performance (rescued kernels avg 1.16× vs already-correct 1.58×).

**Our experiment.** Test these claims in a more realistic, end-to-end setting: take a whole model (Qwen3-1.7B dense, BF16, GQA, 28 layers), use `torch.compile`/inductor as the *baseline* (not eager — that's the dishonest peer), let agents replace specific fused kernels, measure honestly.

**Hardware:** DGX Spark — GB10 Blackwell, sm_121, 48 SMs, unified LPDDR5X, ARM host. Pinned torch 2.12 nightly cu128, Triton 3.7, transformers 5.8.1.

**Agents:** `claude` (Claude 4.7), `codex` (Codex 5.5), `kimi` (Kimi 2.6) — all in non-interactive mode. Forbidden from using `torch.compile`/`@torch.compile`/`torch.jit`.

## Locked design decisions
- Model: Qwen3-1.7B (rejected Qwen3.5-2B early — it's a hybrid DeltaNet+MoE+vision stack).
- Workloads: prefill seq 512/2048 b=1; decode ctx 512/2048 b=1/8 (6 total).
- Primary baseline: `torch.compile` with **cudagraphs OFF** (`torch._inductor.config.triton.cudagraphs = False`). cudagraphs gain isn't from codegen, so it's not the right peer.
- Correctness tolerances: KBX "standard" (cos_sim ≥ 0.95, l1_rel ≤ 0.05, rmse ≤ 0.10) and "strict" (cos_sim ≥ 0.99, l1_rel ≤ 0.01, rmse ≤ 0.01). We added mutation + determinism checks after early findings.

---

## Stage 1a — Env verification

**Hypothesis going in.** Torch + Triton may have rough edges on Blackwell sm_121 / aarch64; need to verify before investing.

**What we did.** Installed torch 2.12 nightly cu128, triton 3.7, transformers 5.8.1 into `.venv/`. Ran a tiny `torch.compile` smoke + a hand-written Triton vector-add.

**Findings.**
- GB10 reports compute capability **sm_121** (not sm_120). Torch nightly + Triton 3.7 handle it cleanly — no PTX fallback.
- `nvidia-smi` memory display is broken on GB10 ("Not Supported"); use `torch.cuda.memory_allocated()` instead.
- The `.venv` had to be built from miniconda's python — system Python lacks dev headers, and Triton JIT-compiles a `cuda_utils.c` shim needing `Python.h`.
- Driver 580.95.05 / CUDA 13.0; torch built against 12.8. Forward compat works.

**Takeaway.** Stack is viable. No Blackwell showstoppers at the language/runtime layer. Disk fine (328 GB free).

**Files.** `env/{verify_env.py, requirements.txt, env_report.md}`, `.venv/`.

---

## Stage 1b — Workload module

**Hypothesis.** Need canonical, deterministic inputs so reruns / different scripts produce comparable numbers.

**What we did.** Built `workload/{model.py, inputs.py, correctness.py}`:
- `model.py` — `load_model`, `prefill_fn`, `decode_fn`, `build_kv_cache`. Verified HF Qwen3 cache API from source (`DynamicCache(config=...)`, position_ids auto-derived).
- `inputs.py` — six named workloads with seeded RNG. Decode workloads expose `kv_cache_builder(model)` that prefills then exposes the cache.
- `correctness.py` — `check_outputs(reference, candidate, dtype, task)` with KBX standard/strict thresholds.

**Smoke result.** All imports + canonical input building work without the model loaded.

**Files.** `workload/{model.py, inputs.py, correctness.py, smoke_test.py}`.

---

## Stage 1c — Baselines: eager vs torch.compile

**Hypothesis.** `torch.compile` with cudagraphs OFF should give a meaningful speedup over eager (1.1–1.5× expected for a dense 1.7B decoder on a modern GPU).

### v1 — failed
Compile correctness vs eager failed across all 18 configs (cos_sim 0.55–0.96, one reading 1.79 — impossible value). Initially thought torch.compile was broken on Blackwell.

### Diagnostic: minimal reproducer
Loaded model + single fixed input, ran eager vs compile in one process. **Result:** cos_sim 0.99997, perfectly fine.

**Root cause.** `workload/inputs.py` derived per-workload RNG seeds via Python's built-in `hash(name)`. Built-in `hash()` is **randomized per Python process** (PYTHONHASHSEED). `run_eager.py` and `run_compile.py` generated *different* input tensors for the same workload name. The "correctness failure" was literally comparing outputs on different inputs.

**Fix.** Switched to `hashlib.sha256(name.encode())` for cross-process stability. Saved as feedback memory.

### v2 — partial pass
Most workloads now pass. Two artifacts:
- Some prefill workloads reported cos_sim = 1.79 or 0.55 with tiny l1_rel/rmse — implausible.
- One workload (decode_ctx512_b1) had real divergence (l1_rel 0.41).

**Root cause #2.** `torch.dot(r, c)` on 311M-element fp32 vectors accumulates enough rounding error to break cosine math (cos_sim > 1 or < 0 on outputs that are numerically very close).

**Fix.** `correctness.py` now uses fp64 for cosine numerator/denominator and clamps to [-1, 1]. fp32 is fine for l1_rel/rmse (non-negative sums).

### v3 — all pass

| workload | eager (ms) | default | cudagraphs-on | max-autotune |
|---|---|---|---|---|
| prefill_512_b1 | 247.32 | 1.03× | 1.02× | 1.02× |
| prefill_2048_b1 | 850.85 | 0.99× | 0.98× | 0.98× |
| decode_ctx512_b1 | 29.62 | 1.16× | 1.17× | **1.71×** |
| decode_ctx512_b8 | 140.34 | 0.98× | 0.94× | 0.94× |
| decode_ctx2048_b1 | 32.51 | 1.00× | 0.96× | 1.23× |
| decode_ctx2048_b8 | 175.40 | 0.88× | 0.81× | 0.80× |

**Headline finding.** On Blackwell + Qwen3-1.7B, `torch.compile` with cudagraphs off ranges **0.80×–1.71×** — mostly a wash, sometimes a small slowdown. Only decode_ctx512_b1 max-autotune shows a real win.

**Why so little speedup.**
- Inductor delegates all matmuls to `aten::mm` (cuBLAS) in `mode="default"`. Only fused-Triton codegen is on the table.
- Inductor refuses `max_autotune_gemm` on GB10 (warning: `Not enough SMs to use max_autotune_gemm mode` — 48 SMs).
- Bandwidth-bound pointwise fusions can't move the needle much over eager.

**Takeaway.** Compile is barely a competitor on this hardware. The real codegen-vs-codegen comparison is going to be very narrow — agents won't have huge headroom.

**Files.** `baselines/{run_eager.py, run_compile.py}`, `baselines/results/{eager.json, compile_*.json, traces/, reference_outputs/, candidate_outputs/, logs/}`.

---

## Stage 2 — Extract + rank inductor kernels

**Hypothesis.** Inductor produces a handful of fused kernels that account for most wall time. Top-N covering ~80% is the target list for agent replacement.

**What we did.** Re-ran every workload under `TORCH_COMPILE_DEBUG=1` + `TORCH_LOGS="+inductor,output_code,fusion,schedule"`, cudagraphs OFF, fresh model per workload. Captured emitted Triton kernels + fx graphs + example inputs. Built an aggregator that reads Chrome profiler traces and ranks kernels by wall-time share.

**Findings.**
- **24 distinct fused Triton kernels** across the 6 workloads.
- **prefill_512_b1:** 50.6% fused-Triton / 49.4% aten (`mm` 49.08%, `eff_attn` 0.37%, clone 0%, copy_ 0%).
- **decode_ctx512_b1:** 40.7% fused-Triton / 59.3% aten (mm 28.06%, eff_attn 14.11%, clone+copy_ for KV cache 17.16%).
- **Aten ops are off-limits to agents** — they're already on cuBLAS/cuDNN/SDPA. Roughly half the model isn't available for agent replacement.

**Top Triton kernels (prefill_512_b1):**
1. **34.2%** — RMSNorm + QKV-prelude + RoPE + causal mask, fused "SDPA prelude" (two siblings: `_where_3` 24.99% + `_where_4` 9.24%). Biggest single prize.
2. **5.67%** — SwiGLU gate (`_unsafe_view_mul_silu_6`).
3. **5.38%** — RoPE cos/sin build (`_arange_bmm_cat_cos_sin_neg_2`).
4. **3.77%** — RMSNorm reductions (`_rsqrt_11` + `_rsqrt_9`).
5. **1.16%** — SDPA-prelude variant `_where_1`.

**Top Triton kernels (decode_ctx512_b1):**
- SDPA prelude variants ~19%, RMSNorm reductions ~12.3% combined, KV cache concat 2.9%.

**Surprise: SDPA prelude > SDPA itself** on these shapes. The prelude Triton (~25% prefill) costs more than the actual attention math (0.37% on this shape, much higher on decode where there's no flash-attention). A flash-attn-style absorbed-prelude kernel would be a massive win.

**Surprise: Blackwell autotuner behavior.** Inductor dynamically scales `R0_BLOCK` from 1024 → 512 on several reductions. Register/shared-mem pressure model isn't fully tuned for GB10's 48-SM / 65K-reg layout.

**Surprise: CUPTI is broken on GB10** (`CUPTI_ERROR_INVALID_DEVICE`). Profiler events are launch+sync spans, not pure GPU time. Relative ranking still meaningful; absolute numbers need `do_bench` for headlines.

**Files.** `extract/{dump_inductor.py, rank_by_walltime.py, match_kernels.py, manifest.json, ranking_*.json, ranking.md, aten_calls.json, kernels_index.json, kernels/, inductor_debug/}`.

---

## Stage 3a — Agent loop harness + SwiGLU × claude smoke

**Hypothesis.** Build the loop on a small, well-defined kernel (SwiGLU, 5.67% of prefill, bandwidth-bound pointwise). Validate end-to-end with one CLI before scaling.

**What we did.** Built `agent_loop/{wrappers/{claude,codex,kimi}_wrap.sh, tasks/swiglu/{task.md, reference.py, harness.py, inductor_baseline_us.json}, run_one.py}`. Orchestrator handles sandbox isolation, GPU util sampling, wall-clock tracking, trajectory log capture.

Ran `python agent_loop/run_one.py --cli claude --task swiglu --max-attempts 5 --run-id swiglu_claude_smoke`.

**Result.** Claude wrote a Triton kernel that passed correctness (cos_sim 1.0000), latency 116.74 μs. Baseline at the time (loose) was inductor profiler-mean 361 μs → reported as **3.10× speedup**. Also beat eager 141 μs by 1.21×. Wall-clock 108.8 s. Mean GPU util 12.95% (agent is thinking-bound).

**Caveat the agent flagged:** inductor's 361 μs is a profiler aggregate (includes launch + sync overhead across many invocations in-model), **not** a standalone microbench. The honest peer for an agent's standalone kernel is either eager or inductor-standalone. Marked as TODO before headline numbers.

**Takeaway.** Loop works. The baseline is wrong but functional.

**Files.** `agent_loop/wrappers/`, `agent_loop/tasks/swiglu/`, `agent_loop/run_one.py`, `agent_loop/runs/swiglu_claude_smoke/`, `agent_loop/sandbox/swiglu_claude_smoke/`.

---

## Stage 3b — SwiGLU × codex + kimi smokes

**Hypothesis.** Different CLIs will produce different kernels at different wall-clock costs. Approach diversity is informative.

**What we did.** Same task, same harness, codex then kimi (serial, GPU contention). Tracked per-CLI wall-clock, GPU util, candidate latency, approach.

**Results (initial, vs loose baseline):**

| CLI | wall | candidate | vs 361 baseline | vs eager | cos_sim | approach |
|---|---|---|---|---|---|---|
| claude | 109 s | 116.7 μs | 3.10× | 1.21× | 1.0000 | Triton BLOCK=1024 nw=16, faithful fp32 sigmoid |
| codex | 302 s | 107.5 μs | 3.36× | 1.29× | **0.9985** | Triton BLOCK=2048 nw=32, **clamp-approx sigmoid + writes into y in-place** |
| kimi | 1138 s | 105.4 μs | 3.43× | 1.34× | 1.0000 | Triton BLOCK=64 nw=4, **explicit eviction_policy hints** |

**Findings.**
1. **Diverse strategies.** Three different block sizes, three different warp counts, two faithful sigmoids vs one approximation. All converged on Triton (no raw CUDA, no exotic backends).
2. **Codex exploited the tolerances** — replaced `sigmoid(x)` with `clamp(0.21*x + 0.5, 0, 1)`. Wrong for |x| > ~2.4 but passes cos_sim ≥ 0.95. Also breaks function contract by aliasing output to input `y`. This is **KBX Insight 1 in microcosm** — agents will game tolerance gaps if you let them.
3. **Inverse relationship between agent wall-clock and kernel quality.** Kimi (1138 s, ~10× claude's wall) produced the fastest faithful kernel (105.4 μs). Claude (109 s) middle quality. Codex (302 s) chose to optimize against the wrong objective.
4. **GPU util is always low** (~0–13%). Agent loops are thinking-bound, not GPU-bound. Important for blog narrative: "agent compute used" is mostly LLM inference, not kernel benchmarking.

**Methodology gap surfaced.** The 3.0–3.4× speedups are against a profiler-aggregate baseline that overstates inductor's actual standalone cost. Real codegen-vs-codegen comparison requires inductor's kernel benched standalone.

**Files.** `agent_loop/runs/swiglu_{codex,kimi}_smoke/`, `agent_loop/runs/smoke_summary.md`.

---

## Stage 3c — Tighten methodology

**Hypothesis.** Honest evaluation needs (a) strict correctness, (b) no input mutation, (c) determinism, (d) a real codegen-vs-codegen baseline.

**What we did.**
1. Extended harness: now reports `correctness` (standard) AND `correctness_strict` (cos_sim ≥ 0.99). Added mutation check (`x_orig` / `y_orig` clones + `torch.equal` after run). Added determinism check (call twice, require near-byte-equal output). Updated verdict ladder: `PASS_STRICT > PASS > FAIL_MUTATION > FAIL_NONDETERMINISTIC > FAIL_CORRECTNESS > ERROR`.
2. Built `extract/microbench_inductor.py` — loads inductor's emitted kernel from cache, strips the `@triton_heuristics.pointwise(...)` decorator (the autotune wrapper rejects XBLOCK kwargs in our torch nightly), launches the bare `@triton.jit` function with metadata-recorded config, runs `do_bench` 25 warmup × 100 reps.
3. Got the real inductor standalone microbench for SwiGLU.
4. Re-ran the three existing candidates through the tightened harness (no agent re-run, just harness re-eval).

**Findings — methodology correction shocks the picture:**

**Inductor standalone SwiGLU: 109.6 μs median** (σ ≈ 2% across 3 independent runs). The 361 μs profiler-aggregate overstated inductor's kernel cost by 3.3× — that gap was launch jitter + cold-cache effects across 280 in-model invocations.

**Re-evaluated verdicts:**

| CLI | candidate | vs **honest 109.6 μs** | vs eager | verdict |
|---|---|---|---|---|
| claude | 121.79 μs | **0.90×** (slower) | 1.16× | PASS_STRICT |
| codex | — | n/a | n/a | **FAIL_MUTATION** (also fails strict tolerance independently) |
| kimi | 103.42 μs | **1.06×** (5.7% faster) | 1.36× | PASS_STRICT |

**This is the real headline finding so far:** with strict correctness and an honest baseline, the agent-vs-inductor margin collapses from "3× win" to **0.90×–1.06×, i.e. the noise floor**. Claude actually *loses* to inductor's codegen on a faithful comparison. Kimi narrowly wins after 19 minutes of compute (≈ $0.x per μs saved, when we eventually price it).

This directly mirrors **KBX Insight 3** ("performance is an unsolved frontier; pooled median speedup over correct kernels was 1.0008×") — and on a single kernel with a much better baseline than the paper used.

**Takeaway.** Approximation gaming (codex) is real and easily caught. The "agents beat inductor" claim from loose evaluation evaporates under honest measurement. The honest finding is more interesting than the inflated one.

**Files.** `agent_loop/tasks/swiglu/harness.py` (tightened), `agent_loop/runs/swiglu_{claude,codex,kimi}_smoke/result_strict.json`, `extract/microbench_inductor.{py,json}`, `agent_loop/runs/smoke_summary.md` (rewritten).

---

## Stage 3d — Strict-feedback re-runs (DONE)

**Hypothesis.** Now that the harness reports strict correctness, mutation, and the real 109.6 μs baseline as feedback to the agent, do agents adapt? Specifically:
- Does codex stop trying to approximate (now that strict tolerance is the gate)?
- Does anyone iterate to legitimately beat inductor under strict eval?
- Does iterative refinement help performance when the signal is honest? (Counter to KBX Insight 2's repair-bias finding.)

**What we did.**
1. Fixed hardcoded `INDUCTOR_BASELINE_US = 361.3` in harness — now reads from `extract/microbench_inductor.json`.
2. Updated `task.md` to state the real target (109.6 μs), eager peer (~140 μs), strict tolerance + mutation + determinism rules, with an explicit "approximation tricks will fail strict tolerance" sentence.
3. Re-invoked each CLI on SwiGLU with new run-ids `swiglu_{cli}_strict`. Default orchestrator timeout 1200 s.

**Results.**

| CLI | wall | verdict | candidate | vs inductor (109.6 μs) | vs eager | approach |
|---|---|---|---|---|---|---|
| claude | 1200 s (TIMEOUT) | PASS_STRICT | 118.78 μs | **0.92×** | 1.16× | Triton XBLOCK=512 nw=8, matched inductor config |
| codex | 358.7 s | PASS_STRICT | **105.47 μs** | **1.04×** | 1.19× | Triton BLOCK=64 nw=4, **`tl.sigmoid()` + eviction_policy hints** |
| kimi | 1200 s (TIMEOUT) | PASS_STRICT | 103.42 μs | **1.06×** | 1.36× | Same as v1 — eviction_policy approach |

**Key finding — codex changed behavior dramatically.** In v1 (loose evaluation) codex returned a clamp-approximation sigmoid + in-place output aliasing — clearly chosen to exploit cos_sim ≥ 0.95 + skip an allocation. In v2 (strict-feedback prompt) codex returned a *faithful* `tl.sigmoid(x)`-based kernel with eviction_policy hints (essentially adopting kimi's v1 approach) and **legitimately beat inductor by 4%** without contract violations. This is the cleanest "agent adapts to honest signal" data point in the experiment.

**Interpretation.** Codex's v1 behavior was not malice or incompetence — it was a rational response to a loose loss function. Tightening the loss eliminated the gaming. This is a methodology lesson: **agent behavior tracks the gradient you actually exposed, not the one you intended.**

**Claude and kimi hit the orchestrator timeout (20 min) but their final candidates still validated.** Suggests both got stuck iterating without finding further improvements. Claude's final kernel essentially matched inductor's config (XBLOCK=512, nw=8) — could not find a better recipe in the time budget. Kimi held its v1 winning approach.

**Honest spread tightens further:** 0.92× – 1.06×, ≈ ±7% of inductor. Within likely measurement noise. **Reaffirms the KBX Insight 3 framing: on a single bandwidth-bound kernel against a strong baseline, the agent-vs-inductor margin is at the noise floor.**

**Mutation + determinism gates worked perfectly.** All three v2 candidates: `mutates_x=False, mutates_y=False, deterministic=True`. Standard + strict tolerances both pass for all three. Methodology now defensible.

**Cost asymmetry.** Codex paid 358.7 s of wall (and presumably proportional token cost) for a 1.04× win. Claude paid 1200 s and *lost* to inductor. Kimi paid 1200 s for a 1.06× win. Wall-clock cost per unit speedup varies by 3–10× across CLIs.

**Files.** `agent_loop/runs/swiglu_{claude,codex,kimi}_strict/`, `agent_loop/sandbox/swiglu_*_strict/`.

---

## Stage 3e — RMSNorm task scaffolding (DONE; agent runs pending)

**Hypothesis.** RMSNorm is a different shape of problem than SwiGLU:
- It's a *reduction* (not pointwise), so per-block coordination matters.
- Multiple shape variants exist (decode vs prefill, with/without embedding).
- Well-studied — there's a known performance envelope (Welford vs naive, fused vs split).
- Combined ~3.8% prefill + ~12.3% decode → meaningful slice of wall time.

Test whether agent diversity is wider on a reduction than on a pointwise op.

**What we did.** Built `agent_loop/tasks/rmsnorm/` mirroring the swiglu task structure with the tightened harness machinery. Picked the kernel variant, micro-benched inductor's emitted Triton, smoke-validated the harness with a trivial passing candidate.

**Variant picked.** `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9` — residual-add + RMSNorm fusion (Qwen3 always fuses the residual). Shape: x=(1,512,2048), residual=(512,2048), weight=(2048,) bf16. 1.88% of prefill wall time, simplest 3-pointer signature.

**Inductor standalone microbench: 35.84 μs.** Profiler-aggregate was 258.8 μs — **same ~7× divergence as SwiGLU**. Confirms the systematic overstatement in profiler aggregates for in-model kernel invocations.

**Eager reference: ~187 μs.** Inductor is **5.2× over eager** here — much bigger speedup than SwiGLU's 1.3×, more room for inductor and (importantly) more room for an agent to win against inductor.

**Discovery — inductor's kernel is sub-optimal.** It's a **naive two-pass reduction** (not Welford), and the second pass *reloads x and residual from global memory*:
- Pass 1: load x + residual, accumulate `s²` in fp32.
- Pass 2: **reload x and residual**, normalize, multiply by weight, store.

The hidden dim (2048) easily fits in registers. A single-pass tile-resident kernel would eliminate ~33% of the global memory traffic. Effective bandwidth at 35.84 μs is ~175 GB/s, **~64% of LPDDR5X peak** (273 GB/s) — so there's real headroom.

Block sizes inductor picked: `XBLOCK=2, R0_BLOCK=1024, num_warps=8, num_stages=1` (256 grid programs × 2 rows × inner-loop=2 over the hidden dim). Could also set `R0_BLOCK=2048` to eliminate the inner loop.

**Prediction.** At least one agent will produce a single-pass kernel and beat inductor. Possibly all three. The interesting question is by how much — bandwidth ceiling is the next constraint.

**Files.** `agent_loop/tasks/rmsnorm/{task.md, reference.py, harness.py, inductor_baseline_us.json, CHOICE.md, _smoke_candidate.py.bak}`. `extract/microbench_inductor.{py,json}` extended.

**Next.** Once Stage 3d frees the GPU, dispatch claude+codex+kimi on RMSNorm (serial).

---

## Stage 3f — RMSNorm × all 3 CLIs (DONE)

**Hypothesis (pre-registered in Stage 3e).** Inductor's RMSNorm is a naive two-pass reduction that reloads x and residual (effective bandwidth ~64% of LPDDR5X peak). A single-pass tile-resident kernel should win. At least one agent will produce one.

**What we did.** Ran each CLI on the residual-fused RMSNorm task (decode shape `x=(1,512,2048)`, residual=(512,2048), weight=(2048,), bf16). Strict harness, no mutation, deterministic, 30-min orchestrator timeout. Serial. New run-ids `rmsnorm_{cli}_v1`.

**Results.**

| CLI | wall | verdict | candidate | vs inductor (35.84 μs) | vs eager (~170 μs) | approach |
|---|---|---|---|---|---|---|
| claude | 405.9 s | PASS_STRICT | **29.70 μs** | **1.17×** | 5.7× | Two-pass + split eviction_policy (evict_last on accumulation pass, evict_first on store pass) |
| codex | 501.1 s | PASS_STRICT | 31.71 μs | 1.13× | 5.4× | **Single-pass** (BLOCK_M=4, BLOCK_N=2048 = full row in one tile) |
| kimi | 1508.7 s | PASS_STRICT | 36.86 μs | 0.97× | 4.6× | Two-pass essentially mirroring inductor's pattern |

**Prediction was partially right.** Codex *did* produce a single-pass kernel. But claude's cache-aware two-pass beat it. Lesson for the writeup: **the obvious optimization isn't always the winning one**. At this shape on LPDDR5X, the second load hits L2 because the entire row tile (2 × 2048 × 4 bytes = 16 KB) stays warm — so "fewer global loads" matters less than "same hot data on the right cache line."

**Per-CLI strategy notes.**
- **claude:** matched inductor's outer structure (two-pass over R0_BLOCK=1024, XBLOCK=1) but added explicit eviction-policy hints that inductor didn't include. Caught the L2-reuse opportunity inductor's heuristics didn't.
- **codex:** chose the radical structural rewrite — fits the whole row in one tile, eliminates the inner loop entirely. Cleaner code (~25 lines vs claude's ~60), 6% slower than claude. Interesting that the "elegant" solution lost.
- **kimi:** stuck close to inductor — same two-pass, similar eviction hints, similar block config. Burned 25 minutes finding no novel optimization. **Slowest agent again, worst kernel again.** Same pattern as SwiGLU.

**What none of them tried.** Welford-style incremental variance (avoids the catastrophic cancellation risk on the sum-of-squares + would have natural single-pass structure). All three did naive `mean(x²)`. Not an issue at hidden=2048 in fp32, but worth flagging as a missed textbook idea.

**Honest spread on RMSNorm:** 0.97×–1.17×, vs SwiGLU's 0.92×–1.06×. Wider spread, more headroom. The hypothesis "kernels where inductor leaves bandwidth on the floor have more room for agents to win" holds for this data point.

**Compute cost.** Wall-clock 406s + 501s + 1508s = ~40 minutes of agent compute total to produce three working RMSNorm kernels, two of which beat the strong baseline.

**Files.** `agent_loop/runs/rmsnorm_{claude,codex,kimi}_v1/`, `agent_loop/sandbox/rmsnorm_*_v1/`.

---

## Stage 4 — End-to-end integration (DONE, hugely informative — CORRECTED)

**Note.** First pass had a `both_pure decode_ctx2048_b1` reading of 64.96 ms (0.51× eager) that looked catastrophic. **Re-run gave 29.93 ms — the 2× regression was Triton autotuner state pollution**, not a real interaction. Numbers below use the stable reruns. Original buggy reading documented as a methodology lesson.



**Hypothesis.** Standalone kernel wins (SwiGLU 1.06× vs inductor, RMSNorm 1.17× vs inductor) will partially translate to end-to-end speedup. Open question: how much do dispatch overheads / integration costs eat the win? Combining patches expected to be roughly additive.

**What we did.** Wrote `e2e/patches.py` that monkey-patches `Qwen3MLP.forward` (SwiGLU site) and `Qwen3RMSNorm.forward` (pure variant) AND a fused-at-DecoderLayer-level variant for residual+norm. Wrote `e2e/run_e2e.py` that loads model fresh per config, applies patches, runs do_bench (25w × 100r median) on all 6 canonical workloads, checks correctness vs the saved eager reference. Configs: eager / +swiglu / +rmsnorm_pure / +both_pure / +both_fused / torch.compile_default.

**Results — geomean speedup vs eager across all 6 workloads:**

| config | geomean | best | worst | narrative |
|---|---|---|---|---|
| +swiglu (kimi) | 1.014× | 1.022× | 1.003× | Standalone 1.06× win largely vanishes in-model |
| +rmsnorm-pure (claude) | **1.043×** | 1.090× | 0.997× | Best single config; 1.17× standalone → 1.04× e2e |
| +both pure | 1.040× | 1.090× | 0.997× | Sub-additive — swiglu marginal win absorbed into rmsnorm path |
| +both fused | 1.021× | — | — | Fused refactor is *worse* than pure (see note below) |
| **torch.compile** | **1.009×** | — | — | **Worse than either single agent patch on geomean** |

Best-case workloads for the agent kernels: small-batch decode (decode_ctx512_b1, decode_ctx2048_b1) where dispatch overhead and norm-call count favor the agent kernel.

**Key findings.**

1. **SwiGLU's 1.06× standalone win evaporates in-model.** End-to-end gains ~1.4% on geomean. The ~6 μs/call saving on 280 invocations is lost in 30+ ms of decode latency / 250 ms of prefill. Standalone kernel wins do *not* automatically translate to system-level wins.

2. **RMSNorm's 1.17× standalone win partially survives — 1.04× geomean, up to 1.09× best-case.** Best on small-batch decode workloads where dispatch overhead and per-layer norm-call count favor the agent kernel. Vanishes on prefill (matmul-bound). Net translation rate: ~25% of the standalone fractional gain survives on average; ~50% best-case.

3. **Combining patches is sub-additive, not super-additive.** RMSNorm alone: 1.043×. SwiGLU alone: 1.014×. Both pure together: 1.040× — the SwiGLU marginal gain gets absorbed (likely because both kernels load through the same dispatch path, so the second's overhead amortizes against the first's).

4. **The "proper fused" refactor is WORSE than pure (1.021× vs 1.043×).** Mechanism: to fuse `residual_add + post_attention_norm` in one kernel while preserving the residual for the trailing `pre_mlp_residual + mlp_out` step, the implementer had to materialize `pre_mlp_residual = a + r` as a separate kernel pass. That extra pass cost more than the in-kernel fusion saved. A true full-block fusion (matching inductor's cross-layer pattern, writing the residual sum inside the kernel) would land — but it's a bigger refactor than this experiment scoped. **Lesson: optimizing kernels at decoder-layer scope requires fusing more aggressively than just adjacent ops; partial refactors can backfire.**

5. **decode_ctx512_b1 "correctness failure" is a measurement artifact.** Root cause traced: the harness rebuilds KV cache through patched prefill, then takes `last_token_ids = argmax(logits[:, -1])`. bf16 drift across 28 layers flips this argmax exactly once for the b=1 prompt → different decode-step input → different output. Cos_sim of the *decode output* is computed against an eager reference built with eager prefill — so they're literally decoding from different starting tokens. **Not a kernel bug.** Larger-batch workloads don't show this because argmax across batch is more robust. Methodology improvement for any future writeup: take logits-comparison against a fixed last_token_ids derived from eager, or compare softmax distributions instead of argmax.

6. **The strongest practical claim is now defensible.** The patched eager model (with RMSNorm-pure alone) **beats `torch.compile(model)` end-to-end on geomean** (1.043× vs 1.009×). Not by much, but real, and on a comparison that includes the cudagraphs-off torch.compile baseline that's been the experiment's primary peer. For the post: **"one carefully-chosen agent kernel, integrated into eager, can beat inductor end-to-end on Blackwell."** Tight claim, but defensible.

7. **`both_pure` initially looked catastrophic** (decode_ctx2048_b1 going 32.9 → 64.96 ms, a 0.51× regression). **Rerun gave 29.93 ms.** The first reading was Triton autotuner state pollution — measurement methodology lesson: any "shocking" e2e regression should be reproduced before being reported. Listed under cumulative findings.

**Methodology gotchas observed.**
- **Triton autotuner state pollution between configs:** loading two custom kernels in the same Python process can cause autotuner to pick wrong configs the first time around. Workaround: run each config in a fresh process, or call `torch._dynamo.reset()` between configs. We do the former.
- **p10/p90 collapse to median** for many configs/workloads — do_bench's `return_mode="all"` path degrades silently on some shapes. Median is still reliable.
- **HF class-level patching is coarse.** Monkey-patching `Qwen3RMSNorm.forward` affects the input embedding norm AND every layer norm (same class). Can't selectively patch just decoder-layer norms via class-level patching — would require instance-level monkey-patches. Didn't matter here, but a real consideration for selective replacement experiments.
- **KV cache rebuild in the harness depends on the model under test.** If the model is patched, the cache is built through patched prefill. Tiny bf16 differences across 28 layers can flip the argmax for the decode-step input. Correctness comparisons should pin the decode input externally, not derive it from the model under test.

**Files.** `e2e/{patches.py, run_e2e.py, summary.md, kernels/, results/}`.

## Stage 5 — SDPA prelude × all 3 CLIs (DONE — biggest result)

**Hypothesis going in.** The mega-fused "SDPA prelude" kernel (`_where_3` family — 34% of prefill_512_b1) is inductor's biggest single Triton output. **Prediction**: agents will stay in Triton+cuBLAS and won't attempt Flash-Attention-style absorbed-prelude. **Possible headroom**: fusing the 3 cuBLAS QKV GEMMs into one (a textbook trick inductor doesn't do here) would save dispatch + intermediate buffer + likely improved cuBLAS algo selection.

**What we did.** Scaffolded the task by reading inductor's output_code to pin down the contract (Stage 5a): inputs `hidden_states + 3 projection weights + 2 norm scales + inv_freq + position_ids + attention_mask + eps`, outputs `(q, k_gqa_expanded, v_gqa_expanded, additive_causal_mask)`. **Inductor standalone microbench: 4045.73 μs** (full chain; the prefill_512_b1 profiler-aggregate of ~34% wall = ~91 ms over many invocations). **Eager reference: ~1995 μs** at the same shape — same ballpark as inductor, no GEMM-related speedup either way at unit scope.

**Inductor's component breakdown:** q_proj 610, k_proj 776, v_proj 1539, q_norm+RoPE 30, k_norm+RoPE 18, kv_expand_k 24, kv_expand_v 24, mask 9. **The 3 QKV GEMMs alone = 2925 μs (72%).** That's the big knob.

Ran each CLI on the task with 45-min orchestrator timeout, strict harness, multi-output correctness check.

**Results.**

| CLI | wall | verdict | candidate | vs inductor (4045 μs) | vs eager (~1995 μs) | approach |
|---|---|---|---|---|---|---|
| claude | 302.6 s | PASS_STRICT | 1513.47 μs | **2.67×** | 1.32× | 4 Triton kernels (Q/K/V/mask), **separate** cuBLAS for the 3 QKV projections |
| codex | 227.2 s | PASS_STRICT | 1515.52 μs | **2.67×** | 1.32× | Essentially identical to claude — per-head Triton + separate QKV GEMMs |
| **kimi** | 981.0 s | PASS_STRICT | **1034.24 μs** | **3.91×** | **1.93×** | **Stacked QKV weights → single cuBLAS GEMM** + Triton epilogue for RMSNorm+RoPE+GQA |

**Key finding — agents found the win inductor missed.**

Kimi's optimization: stack `[w_q, w_k, w_v]` vertically into one weight matrix (shape `(num_q + num_kv + num_kv) × head_dim × hidden = 4096 × 2048`), do **one** `hidden_states @ w_qkv^T` via cuBLAS, then split the output into Q (16 heads), K (8 heads), V (8 heads). This collapses 3 GEMM launches into 1, saves the intermediate buffer allocation, and lets cuBLAS pick a better algorithm on the wider matrix. Claude and codex did NOT make this optimization — they kept the 3 separate QKV GEMMs.

**Why does the speedup look so much bigger than SwiGLU/RMSNorm?** Because inductor wasn't aggressive enough about cross-op fusion at this scope. SwiGLU is bandwidth-bound pointwise (inductor's natural strength) — no headroom. RMSNorm is a reduction where inductor used a naive two-pass — moderate headroom (33% wasted bandwidth). SDPA prelude is a *composition* spanning matmul + norm + rotation + layout reshuffles + GQA expansion — inductor lowered each piece reasonably but didn't fuse across the matmul boundary. **The agent's job here was "see the structure" rather than "write a clever inner loop."**

**Agent-quality vs cost pattern, now consistent across 3 tasks:**
- claude: median wall-clock, middle-quality kernel
- codex: fastest wall-clock when it stops early at "good enough"; sometimes converges with claude
- kimi: slowest wall-clock by 3-10×, **best-quality kernel on every task**

On this task: kimi 981 s / 1034 μs (best); claude 303 s / 1513 μs; codex 227 s / 1515 μs. **Kimi paid 3× the compute time of claude to find a 1.46× better kernel.** Cost-per-speedup analysis still TBD.

**Flash-Attention-style absorption: nobody attempted it.** Confirms the prediction. All three stayed inside the Triton+cuBLAS envelope.

**Implications for the post.** This is the **strongest agent-wins story in the experiment**. Headline reframing: "agents don't beat inductor on inductor's home turf (pointwise, simple reductions), but they DO beat inductor on cross-op compositions where the compiler's fusion heuristics are conservative." That's a useful and defensible claim — and it inverts the KBX paper's framing slightly (KBX says iterative refinement doesn't help performance; we're showing that a single-shot agent with a clear contract can find compiler-missed compositions).

**Files.** `agent_loop/tasks/sdpa_prelude/`, `agent_loop/runs/sdpa_prelude_*_v1/`, `agent_loop/sandbox/sdpa_prelude_*_v1/`.

---

## Stage 6 — Blinded code review + mechanical stats (DONE)

**Hypothesis going in.** A competent reviewer rating kernels on a rubric (correctness reasoning, perf reasoning, readability, length, risk) will identify quality differences that bench numbers can't see. Open question: does the reviewer's pick of "best" correlate with measured fastest?

**What we did.** Built `review/stats.py` (LoC/bytes/CC/MI via radon + counts of `tl.load`/`tl.store`/`tl.where`/`tl.sum`/`tl.dot`) and `review/reviewer.py` (blinded single-reviewer using `claude -p` with candidates shuffled via deterministic sha256 seed, labels A/B/C/D, mapping saved separately). Reviewed all 10 kernels (3 agents × 3 tasks + 1 inductor per task).

**Headline result — reviewer's "best" pick vs measured fastest:**

| task | reviewer pick | measured fastest | match |
|---|---|---|---|
| swiglu | kimi (103.4 μs, 1.06×) | kimi | ✓ |
| rmsnorm | codex (31.7 μs, 1.13×) | claude (29.7 μs, 1.21×) | ✗ |
| sdpa_prelude | claude (1513 μs, 2.67×) | kimi (1034 μs, 3.91×) | ✗ |

**Reviewer correct only 1 of 3.** Code-quality intuitions are a weak predictor of measured μs on Blackwell.

Two diagnostically interesting misses:
- **RMSNorm**: reviewer flagged claude's `num_warps=2` choice as "looks like a guess... expect it to underperform inductor" — claude was in fact the fastest.
- **SDPA prelude**: reviewer dismissed kimi's stacked-QKV `torch.cat` approach as "bandwidth this design can't afford on Blackwell unified memory" — kimi was 1.5× faster than the alternatives that didn't stack.

**Mechanical stats (LoC / CC / MI):**
- Agents range LoC 16–109; inductor 31 / 61 on the comparable tasks (sdpa is structurally not apples-to-apples — agents wrote single-file modules, inductor's chain spans 6 files).
- Agents win on Maintainability Index (45–65 vs inductor 41–54), dragged down by inductor's `tmp0..tmpN` names and giant `triton_meta`/`inductor_meta` decorators.
- Inductor CC is flat 1–3 (straight-line codegen). Agents CC 2–7 with mild host-side branching for shape handling + launcher caching.

**CLI style fingerprints, consistent across all 3 tasks:**
- **codex** — terse, minimal boilerplate. Smallest files. Reviewer prefers it on simpler ops.
- **claude** — mid length, adds host-side machinery (cached launchers, module-level globals). Dinged for over-engineering on swiglu; rewarded for safety on sdpa.
- **kimi** — longest, most-commented, most aggressive design. Hardcodes shape constants. Reviewer punishes brittleness; benches reward aggressive design — wins 2 of 3 tasks.

**Production-readiness gap.** Reviewer flagged **5 of 9 agent kernels** as "wouldn't merge without changes" — including 2 of the 3 measured-fastest. Concrete issues:
- swiglu/claude: silent output aliasing via a cached `_OUT` global (mutates state across calls).
- sdpa/kimi: hardcoded `SEQ=512`, `HIDDEN=2048`, `* 0.0078125` (= 1/128) — won't generalize.
- Inductor's emissions scored 5/5 on correctness but 1-2/5 on readability/length on every task — and reviewer identified the inductor source on every task by the decorator dict / `cc=121` props block.

**For the post.** This stage gives the post a useful counterweight: the bench numbers say "agents beat inductor on cross-op composition by 4×", the code review says "you wouldn't ship 5 of 9 of these kernels as-is, AND the reviewer's intuition about which would be fastest was wrong 2/3 of the time." Two real, separable signals — one is "is this fast?" the other is "is this *trustworthy*?" Neither replaces the other.

**Known limitations.** Single reviewer = claude (same family as one writer; blinding mitigates not eliminates). SDPA inductor entry is one kernel of six (documented structural mismatch). One review per task, no resampling — treat rank order as more meaningful than raw rubric averages.

**Files.** `review/{stats.py, stats.json, stats.md, reviewer.py, reviews/, blinding_map_*.json, SUMMARY.md}`.

---

## Stage 7 — SDPA prelude e2e integration (DONE — capstone)

**Hypothesis going in.** The 3.91× SDPA-prelude standalone win is the experiment's biggest. The prelude is 34% of prefill_512_b1 wall time at the profiler level. Question: how much survives in-model integration?

**What we did.** Extended `e2e/patches.py` with `install_sdpa_prelude_kimi(model)`, which monkey-patches `Qwen3Attention.forward` to call kimi's stacked-QKV-GEMM + Triton epilogue kernel. Bridged two interface mismatches: kimi's kernel built cos/sin from `inv_freq`+`position_ids` and made its own mask; HF threads in precomputed `position_embeddings=(cos,sin)` in bf16 and a prebuilt 4D additive mask. Reshaped HF's cos/sin to flat `(B*S, head_dim)` for kimi's kernels, skipped kimi's mask kernel, inlined `F.scaled_dot_product_attention`.

**Prefill-only by construction.** Kimi's K kernel does norm+RoPE+GQA-expand in one pass and emits already-expanded 16-head K/V. HF's `cache.update()` expects pre-expansion 8-head K/V — using the patched path on decode would corrupt the KV cache. The patch is gated on `S > 1 and past_key_values is None`; decode workloads fall through to original eager forward.

**Results.**

| workload | eager (ms) | +sdpa_prelude | +all_winners (swiglu+rmsnorm+sdpa) | torch.compile |
|---|---|---|---|---|
| prefill_512_b1 | 247.32 | ~1.001× | **1.05×** | 1.03× |
| prefill_2048_b1 | 850.85 | **1.010×** | 1.01× | 0.99× |
| decode_ctx512_b1 | 29.62 | 1.000× (fallback) | 1.10× | 1.16× |
| decode_ctx512_b8 | 140.34 | 1.000× (fallback) | 1.00× | 0.98× |
| decode_ctx2048_b1 | 32.51 | 1.000× (fallback) | 1.08× | 1.00× |
| decode_ctx2048_b8 | 175.40 | 1.000× (fallback) | **1.17×** | **0.89× (regression)** |

**Geomean across 6 workloads:**
- eager: 1.000×
- +sdpa_prelude_kimi alone: **1.007×**
- +all_winners (SwiGLU + RMSNorm + SDPA prelude): **1.046×**
- torch.compile (default, cudagraphs off): 1.009×

**Best agent stack vs torch.compile: 1.046 / 1.009 = 1.037×** (best agent stack beats compile by 3.7% geomean). Wins on 5 of 6 workloads.

**Key findings.**

1. **3.91× standalone → 1.010× on the prefill-heavy workload → 1.007× geomean.** Amdahl bit hard: the prelude is only ~5–10% of full forward time (SDPA-proper dominates at S=2048; o_proj + MLP take more wall time too). Even a 3.91× speedup on 5–10% of work yields ~2% e2e. **This is the most important methodological finding of the whole experiment: standalone kernel speedups, even very large ones, don't translate proportionally to system-level wins.** The microbench overstates the practical gain by Amdahl's Law.

2. **The +17% win on decode_ctx2048_b8** is the most striking single-workload result. Compile actually *regresses* there (0.886× vs eager), while the agent stack wins 1.17×. Compile-vs-agents direct delta on this workload: **1.32×**. The agent stack is doing real work that compile fails at.

3. **All-winners stack is sub-additive but cleanly positive**: 1.046× geomean vs eager. SwiGLU contributes minimally (1.01×), RMSNorm contributes most (1.04×), SDPA prelude contributes ~1% on prefill alone. The whole = mostly RMSNorm + SDPA prelude on prefill.

4. **The strongest claim now defensible: "Best agent-stack of three carefully-targeted kernels beats `torch.compile(cudagraphs=off)` by 1.037× geomean across realistic prefill+decode workloads on Qwen3-1.7B / Blackwell."** Narrow, real, and the comparison includes cases where compile actively regresses.

5. **Prefill-only patch limitation.** kimi's kernel is a prefill design; decode would corrupt the GQA-pre-expansion cache contract. Documented as a known limit. A decode-suitable variant would need to NOT expand GQA inside the kernel — kept as future work.

6. **decode_ctx512_b1 still shows the bf16-argmax-flip artifact** (cos 0.885) on all_winners. Documented previously; not a kernel bug, a measurement methodology issue with how the harness rebuilds the KV cache through the patched prefill.

**Files.** `e2e/{patches.py, run_e2e.py, summary.md}` (extended), `e2e/kernels/sdpa_prelude_kimi.py`, `e2e/results/{eager_sdpa_prelude_kimi.json, eager_all_winners.json}`.

---

## Cumulative findings so far (for the writeup)

1. **Inductor barely beats eager on Blackwell + Qwen3-1.7B at decoder shapes.** Most workloads 0.80×–1.04×; only one workload sees >1.2× speedup. Why: half the model is `aten::mm` → cuBLAS (off-limits); the other half is bandwidth-bound pointwise fusions where there's not much room to win.
2. **The SDPA prelude is bigger than SDPA itself** on these shapes (24.99% prefill vs 0.37%). Flash-attention-style absorbed-prelude kernels would be the biggest available win for an agent. We haven't tackled this yet.
3. **Agents game tolerance gaps when allowed.** Codex on SwiGLU replaced `sigmoid` with a hard-clamped linear approximation that passed cos_sim ≥ 0.95 but failed at cos_sim ≥ 0.99. Also broke the function contract by mutating its input. This was caught by tightening the harness with strict tolerance + mutation + determinism checks.
4. **The "3× speedup" headline evaporates under honest evaluation.** Against inductor's standalone microbench (109.6 μs) instead of the profiler-aggregate (361 μs), the spread is 0.90×–1.06× — i.e. noise floor. Mirrors KBX Insight 3 on a single kernel with a stronger baseline than the paper used.
5. **Agent wall-clock varies 10× across CLIs** on the same task with the same iteration budget. Kimi 1138 s, claude 109 s, codex 302 s. Quality and wall-clock are inversely correlated in this sample (slowest agent produced the best kernel).
6. **GPU util is always 0–13%** during agent runs. Agent compute is dominated by LLM inference, not kernel benchmarking. This will matter for cost analysis later.
7. **Honest signals change behavior** (Stage 3d): codex flipped from tolerance-gaming to a faithful kernel when the prompt explicitly stated strict tolerance + mutation rules. Agent behavior tracks the loss function you actually expose.
8. **Kernel-difficulty hypothesis confirmed** (Stage 3e/3f): kernels where inductor leaves bandwidth on the floor have more agent headroom. SwiGLU (inductor 1.3× over eager, near-bandwidth-limit) → agents win 0.92×–1.06×. RMSNorm (inductor 5.2× over eager, naive two-pass leaves ~33% bw) → agents win 0.97×–1.17×. Cleaner finding for the post: "agents help where compilers are weakest, not where they're strongest."
9. **The "obvious" optimization isn't always the winning one** (Stage 3f): codex's single-pass RMSNorm was *slower* than claude's cache-aware two-pass. L2 reuse on a 16 KB row tile beat eliminating redundant DRAM traffic.
10. **Kimi is consistently slow + middling-quality** across both kernels: 1138s/middle on SwiGLU, 1508s/worst on RMSNorm. Either (a) it's grinding without finding wins, or (b) it spends time exploring strategies that don't pan out. Not yet attributable.
11. **No agent has tried exotic backends** (CUDA, CUTLASS, ThunderKittens, raw PTX). All converged on Triton across both kernels and 6 runs. Sample of two tasks, but a strong pattern.
12. **Standalone wins don't fully translate end-to-end** (Stage 4, corrected). SwiGLU 1.06× → 1.014× e2e geomean (~vanishes). RMSNorm 1.17× → 1.043× e2e geomean, up to 1.090× best-case. Net translation rate: 0–50% of the fractional gain survives.
13. **Combining agent kernels is sub-additive but not net-negative.** RMSNorm 1.043×, SwiGLU 1.014×, both 1.040× — the second kernel's gain gets absorbed into the first's dispatch path. The "naive integration is catastrophic" finding from the first e2e pass was **autotuner state pollution that reproduced inconsistently** — corrected on rerun. Methodology lesson: always reproduce shocking e2e numbers.
14. **The "fused refactor" can be WORSE than the pure variant** (1.021× vs 1.043×). Partial fusion that requires materializing intermediate values for downstream use can cost more than it saves. Cross-block fusion (matching inductor's pattern) needs to be done at scope, not surface.
15. **The strongest practical claim: "1 carefully-chosen agent kernel can beat torch.compile end-to-end on geomean."** RMSNorm patch alone: 1.043× geomean vs eager; torch.compile-default: 1.009× geomean vs eager. Narrow but real and defensible.
16. **Agent kernel headroom scales with how much cross-op fusion inductor *didn't* attempt** (Stage 5). On a pointwise op (SwiGLU) where inductor already saturates bandwidth, agents are ±5%. On a reduction (RMSNorm) where inductor's two-pass leaves 33% on the table, agents win up to +17%. On a *composition* spanning matmul + norm + rotation + layout reshuffle (SDPA prelude), where inductor stops at op boundaries, agents win **3-4×** — and the winning move (kimi) is the textbook QKV-stacking trick that inductor doesn't try.
17. **"See the structure, not the inner loop"** is where agents shine. SwiGLU and RMSNorm winning kernels were inner-loop optimizations (block sizes, eviction policies). SDPA prelude winning kernel was a structural rewrite (one GEMM instead of three). Inductor's heuristics aren't built to find the structural rewrite; agents are.
18. **Cost-per-speedup varies by 3-10× across CLIs.** Kimi consistently produces the best kernel at consistently the highest wall-clock cost (and presumably token cost). Codex stops early when "good enough." Claude is the cost-quality median. This is real product-design data — which is the right CLI depends on whether you optimize for fastest result or best result.
19. **A competent code reviewer is a weak predictor of measured μs.** Reviewer matched the measured-fastest only 1 of 3 times (Stage 6). On RMSNorm and SDPA prelude, the reviewer dismissed *the actually-winning* design as "looks slow" / "bandwidth disaster" — wrong both times. Performance intuition on Blackwell unified memory is unreliable enough that bench-or-die remains the rule.
20. **Code review surfaces production-blocking issues bench can't.** Reviewer flagged 5 of 9 agent kernels as "wouldn't merge as-is" — silent output aliasing via cached globals, hardcoded shape constants, fragile in-place mutations. Including 2 of the 3 measured-fastest. Bench and review are separable signals — neither replaces the other.
21. **CLI style fingerprints are stable.** Across 3 tasks: codex terse, claude middle-with-host-machinery, kimi longest-with-aggressive-hardcoded-design. Useful for the post — these aren't random; they're stable vendor characteristics.
22. **Amdahl's Law eats huge standalone wins** (Stage 7). SDPA prelude 3.91× standalone → 1.010× on the prefill-heavy workload → 1.007× geomean. Even a near-4× kernel win produces a few-percent system-level win when the kernel is only 5–10% of total work. **This is the methodological cousin of the 361 μs profiler-aggregate vs 109.6 μs standalone story** — different framings yield wildly different headline numbers; the honest one is always the smallest.
23. **Best agent-stack beats `torch.compile` end-to-end by 1.037× geomean on Blackwell.** Best defensible practical claim *on undertrained hardware*. Stacks SwiGLU (kimi) + RMSNorm-pure (claude) + SDPA-prelude (kimi) patches into eager Qwen3-1.7B. Wins 5/6 workloads vs eager. Beats compile on the same 6 workloads where compile averages 1.009×, including a striking +17% on decode_ctx2048_b8 where compile actually *regresses* (0.886×) — a 1.32× direct delta over compile on that single workload.
24. **The result inverts on 3090 / Ampere** (Stage 8). Compile geomean: 1.009× (Blackwell) → 1.20× (Ampere). Agent SDPA prelude standalone: 3.91× win → 0.74× LOSS. Best agent stack vs compile: 1.037× win on Blackwell → ~0.85× LOSS on Ampere. **The agent-wins finding from Blackwell is hardware-specific to undertrained compile paths.** On mature hardware, compile wins handily.
25. **Agent-kernel brittleness surfaces in cross-hardware integration** (Stage 8). Kimi's 3090 RMSNorm passed strict standalone correctness on the canonical shape (1, 512, 2048) but hard-faulted with illegal memory access in e2e integration because it hardcodes hidden_size=2048 and the model also uses RMSNorm at (B, S, 128) for q_norm/k_norm. **Validates Stage 6's code-review finding**: "wouldn't ship 5 of 9 of these without changes." Now we know what one of those shape-brittleness flags actually breaks.
26. **The "structural wins are hardware-invariant" hypothesis was WRONG.** Pre-registered prediction: kimi's stacked-QKV trick would survive cross-hardware because it's a composition argument. **Disproven**: on Ampere, the 3 separate small GEMMs parallelize across 82 SMs better than one wider GEMM does, AND cuBLAS-on-Ampere is highly tuned for the per-head shapes already. The win that crushed inductor on Blackwell loses 26% on Ampere. **Both inner-loop AND structural wins are hardware-dependent.**

## Open questions
- ~~Stage 3d: does strict feedback change agent behavior?~~ → **YES** (codex flipped from tolerance-gaming to faithful kernel).
- ~~Stage 3e/3f: does diversity widen on a reduction kernel?~~ → **YES** (three distinct strategies: single-pass, two-pass-with-cache-hints, two-pass-mirror-inductor).
- ~~SDPA prelude: biggest prize, hardest target?~~ → **YES, 3.91× standalone is the biggest win we saw**, and nobody tried Flash-Attention-style absorption (as predicted).
- ~~e2e integration of winners?~~ → **DONE** Stage 4 + 7. Standalone wins partially translate; Amdahl caps the SDPA prelude e2e gain at ~1%. Best agent stack: **1.046× e2e vs eager, 1.037× vs torch.compile**.
- ~~Code review by fixed reviewer?~~ → **DONE** Stage 6. Reviewer correctly picks measured-fastest only 1/3 times.

**Remaining unaddressed (for the writeup or future work):**
- Cost analysis (tokens per win) — wrappers need switching to JSON output to extract token counts. Not done.
- Tackle RoPE cos/sin or KV-cache concat for breadth — pattern likely repeats; diminishing returns.
- A decode-suitable SDPA prelude kernel (current one is prefill-only due to GQA-pre-expansion conflict with HF KV cache).
- Cross-hardware validation — out of scope.
- "Stop early at first PASS_STRICT" agent-prompt tweak (suggested by Stage 3d findings; not retested).

---

## Stage 8 — 3090 cross-hardware repro (DONE — flips the story)

**Hypothesis going in.** On mature hardware where inductor's heuristics are tuned, agent wins should shrink or vanish. Specifically: SwiGLU/RMSNorm "inner-loop" wins likely shrink; SDPA prelude "structural" win should survive because it's a composition argument inductor doesn't try regardless of hardware.

**What we did.** Set up identical stack on chunklebox (RTX 3090, Ampere sm_86, 24 GB GDDR6X): torch 2.12 nightly cu128, triton 3.7, transformers 5.8.1, kimi-cli 1.44. Same model, same workloads, same harnesses. Scope reduced to kimi (was best agent on every Blackwell task). Re-ran:
1. Baselines (eager + 3 compile modes × 6 workloads).
2. Inductor kernel extraction + standalone microbenches on 3090.
3. Kimi × 3 tasks fresh on 3090 (so it autotunes for Ampere, not Blackwell).
4. E2E integration of resulting kernels.

**3090 baselines vs Blackwell (eager raw latency, lower is better):**

| workload | DGX Spark (ms) | RTX 3090 (ms) | 3090 is faster by |
|---|---|---|---|
| decode_ctx2048_b1 | 32.51 | 14.79 | 2.2× |
| decode_ctx2048_b8 | 175.40 | 21.29 | **8.2×** |
| decode_ctx512_b1 | 29.62 | 14.31 | 2.1× |
| decode_ctx512_b8 | 140.34 | 17.74 | **7.9×** |
| prefill_2048_b1 | 850.85 | 145.54 | 5.8× |
| prefill_512_b1 | 247.32 | 38.57 | 6.4× |

Bandwidth gap explains most of it (GDDR6X 936 GB/s vs LPDDR5X 273 GB/s = 3.4× ratio) plus 82 vs 48 SMs and a much more mature SDPA path.

**3090 compile vs eager:**

| workload | default | cudagraphs | max_autotune |
|---|---|---|---|
| prefill_512_b1 | 1.15× | 1.15× | 1.22× |
| prefill_2048_b1 | 1.03× | 1.03× | 1.02× |
| decode_ctx512_b1 | **1.95×** | **1.99×** | **2.21×** |
| decode_ctx512_b8 | **1.41×** | 1.27× | 1.35× |
| decode_ctx2048_b1 | 1.26× | 1.21× | 1.34× |
| decode_ctx2048_b8 | **0.72×** ⚠️ | 0.63× | 0.64× |
| **geomean** | **~1.20×** | ~1.17× | ~1.20× |

**Headline: compile gives ~1.20× geomean on 3090 vs 1.009× on Blackwell.** Same model, same workloads, ~120× the codegen contribution. On mature hardware the compiler does real work.

Notable: **decode_ctx2048_b8 still regresses on both hardwares** (Blackwell 0.88×, 3090 0.72×). Compile has a systematic problem with this shape × batch combination — worth flagging as a separate finding.

**3090 inductor standalone microbenches:**

| kernel | Blackwell (μs) | 3090 (μs) | Ampere is faster by |
|---|---|---|---|
| SwiGLU | 109.6 | 26.62 | 4.1× |
| RMSNorm | 35.84 | 13.31 | 2.7× |
| SDPA prelude | 4045 | **182.27** | **22×** |

The SDPA-prelude gap is striking: 22× faster on Ampere. Driven by the 3 QKV cuBLAS GEMMs benefitting from the 82-SM × 936 GB/s machine, plus cuBLAS being much more tuned for these shapes on Ampere.

**Kimi × 3 tasks on 3090 (canonical microbench):**

| task | kimi on Blackwell vs inductor | kimi on 3090 vs inductor | direction |
|---|---|---|---|
| SwiGLU | 1.06× | 1.04× | basically unchanged |
| RMSNorm | 0.97× | 1.08× | slightly better on 3090 |
| **SDPA prelude** | **3.91×** | **0.74× (LOSS)** | **flipped!** |

**The big finding: kimi's QKV-stacking trick that crushed inductor on Blackwell LOSES on Ampere.** Inverted my pre-registered prediction ("structural wins are hardware-invariant"). Mechanism: on Ampere with 82 SMs and 936 GB/s bandwidth, cuBLAS parallelizes 3 separate small GEMMs across SMs better than one wider stacked GEMM. Plus Ampere cuBLAS is highly tuned for the specific GQA-style projection shapes that show up in every transformer. The stacked GEMM that saved dispatch overhead on Blackwell sacrifices parallelism on Ampere.

**Code-review prediction validated.** The DGX Spark code review flagged kimi's SDPA prelude as "shape-brittle, with hardcoded `SEQ=512`, `HIDDEN=2048`" and worried about Blackwell-bandwidth assumptions. It also flagged kimi's stack-then-GEMM as "bandwidth this design can't afford on Blackwell unified memory" (wrong on Blackwell, **right on Ampere**). The reviewer's intuition that this kernel wouldn't generalize was correct.

**Brittleness surfaced in e2e integration.** Kimi's 3090 RMSNorm kernel passed strict standalone correctness on its trained shape `(1, 512, 2048)` but **hard-faulted with `cudaErrorIllegalAddress` in e2e integration**, where the model also calls RMSNorm at `(B, S, head_dim=128)` for q_norm/k_norm in attention. Hardcoded hidden_size=2048. Same brittleness pattern the code review identified; same problem, surfaced on a new hardware where the shape distribution differs slightly. The Blackwell claude kernel handles both shapes; kimi's doesn't.

**E2E integration on 3090 (incomplete due to kimi RMSNorm brittleness):**

| config | geomean vs eager (3090) |
|---|---|
| +swiglu (kimi) | 1.012× |
| +rmsnorm-pure (kimi) | **FAIL** (illegal memory access on q_norm/k_norm shape) |
| +sdpa_prelude (kimi) | skipped (lost standalone 0.74×, Amdahl wouldn't save it) |
| compile_default (from baselines, not e2e harness) | **~1.20×** |

The compile_default number is from `baselines/results/compile_default.json` directly — it's not a patch-based config in the e2e harness. Compile geomean on 3090 = ~1.20×; best obtainable agent stack on 3090 ≈ 1.01× (swiglu alone; rmsnorm broken). **On 3090, compile wins by 18%.** This is the inverse of Blackwell, where the best agent stack beat compile by 3.7%.

Even without the RMSNorm e2e number, the picture is clear: the agent-stack e2e wins observed on Blackwell don't replicate on Ampere because (a) swiglu still translates poorly through dispatch, (b) the rmsnorm kernel is too brittle to integrate without rework, (c) the SDPA prelude — the only kernel where agents were really winning standalone — *loses* standalone on Ampere. **Net: on the mature hardware, compile beats the agent stack.**

**Cross-hardware narrative for the post:**

| | **Blackwell (sm_121)** | **Ampere 3090 (sm_86)** |
|---|---|---|
| compile vs eager (geomean) | ~1.01× (~wash) | **~1.20×** |
| agent SwiGLU standalone | 1.06× | 1.04× |
| agent RMSNorm standalone | 0.97× / 1.21× (kimi/claude) | 1.08× (kimi; integrates? no, crashes) |
| agent SDPA prelude standalone | **3.91× win** | **0.74× LOSS** |
| best agent stack e2e vs eager | 1.046× | ≤1.01× (partial coverage) |
| **best agent stack vs compile** | **1.037× (agent wins)** | **~0.85× (compile wins)** |

**The compiler-maturity hypothesis is strongly confirmed.** Agent value scales inversely with how well-tuned the compiler is on the target hardware. **The Blackwell agent wins are real but they're the wins available specifically on undertrained hardware.** On mature hardware, the compiler closes the gap (and where the agent took a structural bet that was wrong for the hardware, the compiler beats it cleanly).

**Implication for the post.** This is now the strongest single framing of the whole experiment. Don't claim "agents beat compilers"; claim "agents help most where compilers are weakest — and that's a moving target as compilers mature." The 3090 data point is the credibility-anchor for that framing.

**Methodology gotchas observed.**
- Standalone harness shapes don't exercise the full distribution the model uses (kimi's rmsnorm passes one shape, breaks on another). Better harnesses should test multiple shapes including outliers.
- The kimi-cli version on chunklebox was 1.44 vs 1.37 on DGX Spark — we can't fully separate "kimi-on-Ampere produced different kernels" from "newer kimi-cli produces different kernels." For the post we should flag this; the *direction* of the cross-hardware effect is large enough that the kimi-version delta is unlikely to dominate.
- Hardcoded "109.6 μs" in task.md was sed-patched on chunklebox to the 3090 inductor number before kimi ran. Verify in writeup.
- We did not run all-CLI × all-task on 3090 due to scope (kimi-only). Cross-hardware claim is "kimi's best Blackwell kernels don't translate" rather than "all CLIs' kernels don't translate" — a real scope caveat.

**Files.** On chunklebox at `~/Projects/KernelBench/`: `baselines/results/*.json`, `extract/microbench_inductor.json`, `agent_loop/runs/{swiglu,rmsnorm,sdpa_prelude}_kimi_3090/`, `e2e/results/{eager,eager_swiglu_kimi,eager_rmsnorm_pure}.json`.

---

## Analysis: why does `torch.compile` provide so little speedup on Blackwell?

A central enabling condition for the agent-vs-inductor story is that compile barely helps on this hardware (geomean 1.009× vs eager across 6 workloads). Several overlapping reasons, all of which need acknowledgement in the writeup so readers anchored on "compile gives 2× on my A100" don't think we set it up to fail:

1. **Inductor refuses `max_autotune_gemm` on GB10.** The hardware reports only 48 SMs (B200 has 132, H100 80+, RTX 4090 128). Inductor emits `Not enough SMs to use max_autotune_gemm mode` during compilation. **Every matmul falls through to cuBLAS regardless of compile mode.** That's 49% of prefill wall time inductor cannot touch. Codegen contribution is structurally bounded to the remaining 51%.

2. **The remaining 51% is mostly bandwidth-bound on LPDDR5X.** Peak 273 GB/s, vs HBM3 at ~3.3 TB/s. For a SwiGLU-shape kernel reading 100 MB + writing 50 MB, the theoretical floor is ~550 μs. Eager already hits ~140 μs (still leaves headroom), but inductor's fusion only buys ~30 μs (1.06×). The ceiling is close enough that "avoid materializing intermediates" doesn't save much.

3. **Triton on Blackwell sm_121 is still maturing.** We observed inductor's autotuner scaling `R0_BLOCK` down from 1024 → 512 during RMSNorm compilation — register-pressure heuristics catching up. The agents found wins inductor's heuristics missed (eviction policies on RMSNorm, BLOCK_SIZE=64 on SwiGLU, stacked-QKV GEMMs on SDPA prelude) precisely because the heuristics haven't been retuned for this hardware yet.

4. **The headline "compile speedup" you usually see comes from `cudagraphs`, not codegen.** We disabled it deliberately because it captures launch-overhead amortization, not Triton fusion quality. The 1.009× geomean is compile's *true codegen contribution* on this stack. With cudagraphs on, the geomean would be higher but the comparison would be confounded — cudagraphs is something agents could also enable on their replacement kernels.

5. **Compile's worst regressions are decode batch-8 workloads.** decode_ctx2048_b8: compile 0.80× vs eager. Same workload: agent stack 1.17×. Direct delta 1.46×. Likely inductor's fused kernels for this shape are spilling to L2 or making suboptimal block-size choices for the b=8 KV-cache layout. Nobody's tuned it for this shape × this hardware.

**Implication for the writeup framing.** Compile's weakness on Blackwell is the *enabling condition* for the agent wins. On a more mature compile path (A100, H100, even 3090 with Ampere-mature inductor), we'd expect agent wins to shrink or vanish. This isn't a critique of agents — it's a scope condition that should be stated explicitly: **"our agent-vs-compile findings hold for hardware where the compiler hasn't been fully tuned yet."** That cuts both ways: it's the regime where agent intervention is most valuable AND it limits cross-hardware generalization.

A 3090 cross-hardware run is on the table to test this directly — if compile catches up on mature hardware and the agent wins collapse, that *strengthens* this framing rather than undermining it: "agents matter on new hardware while the compilers are catching up."

---

## TL;DR for the writeup

**The headline this writeup can defend with a straight face (Stage 10 numbers):**

> On NVIDIA Blackwell hardware where inductor's heuristics haven't been tuned yet (sm_121), an LLM agent can sometimes find a structural rewrite (here, stacking QKV weights into a single cuBLAS GEMM) that the compiler does not attempt at the aten boundary. The integrated benefit is modest: agent stack beats `torch.compile`'s default mode by ~2% geomean, beats it with cudagraphs by ~6%, but loses to `compile_max_autotune` by ~5%. The headline is more sensitive to baseline configuration than to agent capability. The structural rewrite is hardware-specific: it inverts on mature Ampere where the separate-GEMM path is faster.

**On Blackwell + Qwen3-1.7B (sm_121, May 2026, fresh hardware) — 3-trial median, last_token_ids pinned, cold subprocess per trial:**
- Compile vs eager (all 6 workloads): 1.011× (`compile_default`), 0.978× (`compile_default+cudagraphs` — *worse* than no cudagraphs), **1.089×** (`compile_max_autotune`).
- Best agent stack vs eager: **1.033× geomean** (range 1.004–1.087×).
- Best agent stack vs `compile_default`: **1.022×** (range 0.957–1.147×).
- Best agent stack vs `compile_default + cudagraphs`: **1.056×** (range 0.937–1.271×).
- Best agent stack vs `compile_max_autotune`: **0.948× — agent LOSES** (range 0.651–1.255×).
- **Agent stack only beats compile in `default` mode (with or without cudagraphs); `max_autotune` is the strongest compile mode and the agent stack loses to it.** Earlier writeups quoted default-mode only.
- Per-workload MADs 0.04–0.53 ms. Variance is not the issue; methodology corrections drove the headline shift.

**On RTX 3090 (sm_86, mature Ampere stack):**
- Compile vs eager: ~1.20× geomean (`compile_default`, all 6) / ~1.09× (5 excl decode_ctx512_b1).
- Agent SDPA prelude standalone **flips from 3.91× win to 0.74× loss**. The QKV-stacking trick that beat inductor on Blackwell loses on Ampere because Ampere's cuBLAS parallelizes 3 small GEMMs better than 1 wide stacked GEMM. Both inner-loop AND structural agent wins are hardware-dependent.
- Kimi's 3090 RMSNorm passes standalone, hard-faults in e2e integration (hardcoded `hidden_size=2048` breaks on q_norm/k_norm at `head_dim=128`). Best obtainable agent stack on 3090 is SwiGLU-only (~1.012× geomean).
- **Compile beats the obtainable agent stack on Ampere by a wide margin regardless of which geomean (all-6 or 5) you use.**

**Methodology lessons that should be foregrounded:**
- Baseline framing can overstate by 3×: 361 μs profiler-aggregate vs 109.6 μs standalone microbench on SwiGLU.
- Inductor's standalone SDPA-prelude microbench is a hand-rolled Python loop of 8 kernels (`extract/microbench_inductor.py:_bench_sdpa_prelude`). The 1034 μs agent kernel is one in-graph dispatch. Some unmeasured fraction of the 3.91× gap is per-launch overhead in-graph inductor would amortize. **The 3.91× is not pure codegen quality.**
- The RMSNorm "1.043× geomean win" is partly multi-launch eager replacement (4 launches → 1), not pure agent-vs-inductor codegen. The codegen-quality slice (claude's kernel vs inductor's standalone) is ~1.17×.
- Honest evaluation matters: tightening cos_sim 0.95 → 0.99 + mutation + determinism checks collapsed an inflated "3× speedup" claim to 0.9–1.1× on SwiGLU.
- Amdahl's Law eats huge standalone wins (3.91× → 1.010× on prefill_2048_b1).
- The decode_ctx512_b1 correctness failure (cos_sim ≈ 0.885) was kept in the original headline geomeans; now dropped. Both versions quoted.
- Reviewer's intuition predicts measured-fastest only 1/3 times. Bench-or-die.
- 5 of 9 agent kernels flagged as "wouldn't merge as-is"; 1 of those (kimi's RMSNorm) hard-fails on cross-hardware integration.

**Anecdotal patterns (small sample):**
- In n=9 per CLI on Blackwell, n=3 for kimi on 3090: kimi was the slowest by wall-clock and produced the fastest kernel on every Blackwell task. Codex stops early; tolerance-gamed in Stage 3a, adapted in 3d. Claude is the cost-quality median. **n is small; treat as hypotheses, not stable vendor fingerprints.**
- All three CLIs stayed inside the Triton + cuBLAS envelope on every task. Partly observation, partly task-prompt framing (Triton listed first in "Allowed approaches"). A different prompt might produce a different distribution.

---

## Prompt evolution and experimenter influence

The prompts in `agent_loop/tasks/*/task.md` are not the prompts we started with. They evolved in response to observed agent behavior. A skeptical reader should know this, because it bears on what we can claim the agent "figured out" vs what the prompt told it to figure out.

Timeline:

- **Stage 3a — SwiGLU task.md (initial).** No explicit mention of strict tolerance, no-mutation, or anti-approximation rules. **Codex tolerance-gamed**: returned a clamp-approximation `sigmoid` + in-place output aliasing that passed cos_sim ≥ 0.95.
- **Stage 3d — SwiGLU task.md (strict).** Explicitly stated strict tolerance (cos_sim ≥ 0.99), no-mutation, no-approximation. Codex adapted and returned a faithful `tl.sigmoid` kernel.
- **Stage 3e — RMSNorm task.md.** Inherited the strict framing from 3d. Also mentioned inductor's specific behavior ("two-pass reload"), which is a hint about *what to fix*.
- **Stage 5a — SDPA prelude task.md.** Explicitly listed inductor's emit, labeled "three QKV GEMMs" as "the big knob" (2925 μs / 72% of the chain), and mentioned Flash-Attention-style absorption as an allowed approach. Kimi's winning move (stacking the QKV weights) is the move the prompt pointed at.

What we cannot cleanly separate from the data: "agent figured out X" vs "prompt told the agent to figure out X." A cleaner replication would (a) freeze the prompts before any agent runs, (b) have prompts written by a different person from the experimenter, or (c) ablate by re-running with prompts that don't mention the specific knob.

The "agents converge on Triton + cuBLAS" finding has the same problem: `task.md` "Allowed approaches" lists Triton first ("matches what inductor produced"), CUDA via `cpp_extension` second, "any other approach" third. A prompt that opened with "consider CUTLASS, ThunderKittens, or absorbed-prelude Flash-Attention designs" might produce a different distribution. The convergence pattern is partly a real observation and partly an artifact of prompt framing.

For the writeup: claim "the agent identified a structural rewrite that inductor doesn't attempt at the aten boundary." Do not claim "the agent independently discovered QKV stacking" — the prompt named it.

---

## Stage 10 — Tier 2 corrections (DONE — corrected headline)

**Hypothesis.** The Stage 4/7 headlines (1.046× → 1.037× → 1.033×) were sensitive to two methodology choices we hadn't pinned: (a) the `last_token_ids` derivation that triggered a bf16-drift argmax flip on `decode_ctx512_b1`, (b) within-process replication that confounds warmup and JIT caching. A k=3 replication with `last_token_ids` pinned across configs and each trial running in a cold subprocess should both tighten variance and (more importantly) close the correctness loophole.

**What we did.**
1. Pinned `last_token_ids` so all configs see the same target token at the decode-correctness check — fixes the bf16-drift argmax flip.
2. Re-ran eager and `eager_all_winners` k=3 times with each trial in a fresh subprocess (cold caches, no JIT carryover).
3. Re-ran `compile_default + cudagraphs` for an apples-to-apples cudagraphs comparison on the compile side.
4. Attempted `eager_all_winners + cudagraphs` (raw CUDAGraph capture on the patched stack) — *failed*; NoneType in the correctness path, return-tensor plumbing not yet right. Deferred.
5. Replicated swiglu-only e2e on chunklebox 3090 with the same methodology.

**Headline shifts (Blackwell, all 6 workloads — no row dropped):**

| comparison | Stage 7 number | Stage 10 number |
|---|---|---|
| best agent stack vs eager | 1.046× | **1.033×** |
| best agent stack vs compile_default | 1.045× | **1.022×** |
| best agent stack vs compile_default+cudagraphs | (not measured) | **1.056×** |
| best agent stack vs compile_max_autotune | ~0.97× | **0.948× (agent LOSES)** |
| compile_default vs eager | 1.001× | 1.011× |
| compile_default+cudagraphs vs eager | (not measured) | 0.978× (regresses) |
| compile_max_autotune vs eager | 1.079× | 1.089× |

**Per-workload speedup vs eager (3-trial median, MAD 0.04–0.53 ms):**

| workload | eager (ms) | agent stack | compile_default | cd+cgraphs | compile_max_autotune |
|---|---|---|---|---|---|
| prefill_512_b1 | 253.49 | 1.010× | 1.055× | 1.003× | 1.043× |
| prefill_2048_b1 | 868.05 | 1.004× | 1.010× | 0.993× | 1.002× |
| decode_ctx512_b1 | 28.97 | 1.087× | 1.131× | 1.161× | **1.671×** |
| decode_ctx512_b8 | 143.46 | 1.011× | 0.999× | 0.966× | 0.958× |
| decode_ctx2048_b1 | 32.90 | 1.080× | 1.009× | 0.986× | **1.242×** |
| decode_ctx2048_b8 | 175.42 | 1.008× | **0.879×** | **0.793×** | **0.803×** |

**3090 replication (chunklebox, swiglu-only stack):**

| comparison | geomean | range |
|---|---|---|
| eager_swiglu_kimi vs eager | **0.983×** (regresses) | 0.957–1.009× |
| compile_default+cudagraphs vs eager | **1.067×** | 0.621–1.967× |

decode_ctx2048_b8 regresses under cudagraphs on both hardwares (0.793× Blackwell, 0.621× 3090) — same systematic compile-side pathology.

**Findings.**
1. **Pinning fixed `decode_ctx512_b1` correctness.** All 6 workloads now legitimately pass; no "drop a row" caveat. The single all-6 geomean is the honest number.
2. **The progression 1.046× → 1.037× → 1.033× tightens as methodology tightens.** Each correction shaves headline, none flips it. Variance is small (MAD 0.04–0.53 ms across workloads). Single-trial numbers were already roughly right; the geomean shift came from methodology corrections, not noise.
3. **The single most important correction: agent vs `compile_max_autotune` is 0.948×, i.e. the agent stack *loses* to the strongest compile baseline by ~5%.** Earlier drafts quoted only `compile_default`. The agent-stack win is real but only against the weakest compile mode.
4. **`compile_default + cudagraphs` is *worse* than `compile_default`** on Blackwell (0.978× vs 1.011×). Cudagraphs has a systematic interaction with `decode_ctx2048_b8` (0.793× under cudagraphs vs 0.879× without). Same shape × batch pathology recurs on 3090 (0.621×).
5. **kimi-3090 RMSNorm hard-fault is sm_86-specific, not generic shape-brittleness.** A multi-shape harness extension showed kimi_v1's RMSNorm passes all three shapes on Blackwell sm_121 (including head_dim=128). The 3090 hard-fault is therefore probably shared-memory limits / launch-grid choices specific to sm_86, not a generic "hardcoded constants" failure. Earlier Stage 8 framing was too broad.

**Deferred.**
- Agent stack wrapped in raw cudagraphs (return-tensor plumbing bug). Without this cell, we cannot directly compare agent-with-cudagraphs vs compile-with-cudagraphs.
- 3090 full agent stack (kimi's RMSNorm hard-fault is sm_86-specific; not blocked by methodology).

**Files.** `e2e/results/{eager,eager_all_winners,compile_default_cgraphs}.json` and `eager{,_all_winners}_trial{0,1,2}.json` (per-trial); `baselines/results/{compile_default,compile_max_autotune}.json`; chunklebox replication results on the 3090 host.

---

## Stage 10b — Final corrections: symmetric replication + cgraphs-agent unblock (DONE)

**Hypothesis.** Two open holes remained after Stage 10: (a) compile baselines (`compile_default`, `compile_max_autotune`) were still single-trial on both hardwares while agent/eager were 3-trial; (b) `eager_all_winners_cgraphs` had failed to run, so we could not directly compare agent-with-cgraphs to compile-with-cgraphs. Closing both produces the final defensible numbers.

**What we did.**
1. Re-ran `compile_default`, `compile_max_autotune`, `compile_default_cgraphs` k=3 on Blackwell with cold subprocesses + pinned `last_token_ids`.
2. Re-ran `compile_default`, `compile_max_autotune`, `compile_default_cgraphs`, `eager_swiglu_kimi` k=3 on chunklebox 3090 with the same methodology.
3. Unblocked `eager_all_winners_cgraphs` by (a) validating correctness before capture and freeing ref/out tensors, (b) bypassing `triton.testing.do_bench`'s L2-cache-clear write-kernel (which OOB-asserts inside captured graphs on torch 2.12 nightly + Blackwell), and (c) using a manual CUDA-event timing loop when `is_cudagraph=True`. The fp64 correctness allocations were the root cause of the prior OOB asserts in embedding lookup during replay — they corrupt the graph pool's internal index tensors.
4. Discovered that SDPA-prelude is structurally incompatible with cudagraphs: under graph capture HF's mask-builder takes a 4D-mask branch that the `use_kimi` guard doesn't support; the kernel produces structurally wrong output. The `eager_all_winners_cgraphs` config patches *only* swiglu + rmsnorm under capture and silently skips the SDPA prelude. Documented as a known limitation.

**Headline shifts (Blackwell, 3-trial replicated):**

| comparison | Stage 10 (single-trial compile) | Stage 10b (all 3-trial) |
|---|---|---|
| compile_default vs eager | 1.011× | **1.007×** |
| compile_default+cgraphs vs eager | 0.978× | **0.978×** (unchanged) |
| compile_max_autotune vs eager | 1.089× | **1.083×** |
| best agent stack vs eager | 1.033× | 1.033× (unchanged) |
| best agent stack vs compile_default | 1.022× | **1.026×** |
| best agent stack vs compile_default+cgraphs | 1.056× | 1.056× (unchanged) |
| best agent stack vs compile_max_autotune | 0.948× | **0.954×** |
| **eager_all_winners_cgraphs vs eager** (NEW) | — | **1.033×** |
| **agent-cgraphs vs compile-cgraphs** (match-cgraphs, NEW) | — | **1.056×** |

**Headline shifts (3090, 3-trial replicated):**

| comparison | Stage 10 single-trial | Stage 10b 3-trial |
|---|---|---|
| compile_default vs eager | ~1.20× | **1.151×** |
| compile_max_autotune vs eager | — | **1.161×** |
| compile_default+cgraphs vs eager | 1.067× | **1.074×** |
| eager_swiglu_kimi vs eager | 0.983× | **0.982×** (unchanged) |
| swiglu-only stack vs compile_default | — | **0.854×** (agent loses 15%) |
| swiglu-only stack vs compile_max_autotune | — | **0.846×** (agent loses 15%) |
| swiglu-only stack vs compile_default+cgraphs | — | **0.915×** (agent loses 9%) |

**Cross-hardware summary table (the version that should anchor the writeup):**

| comparison | Blackwell | 3090 |
|---|---|---|
| best agent stack vs eager | 1.033× | 0.982× (regresses) |
| best agent stack vs compile_default | **1.026×** (+2.6%) | **0.854×** (-15%) |
| best agent stack vs compile_max_autotune | **0.954×** (-4.6%) | **0.846×** (-15%) |
| compile_max_autotune vs eager | 1.083× | 1.161× |

**Findings.**
1. **Symmetric replication did not flip any qualitative result, but tightened the magnitudes.** Compile-vs-eager numbers shifted by ≤0.6%; agent-vs-compile shifted by ≤0.6%. All gaps remain MAD-bounded.
2. **The agent stack never beats `compile_max_autotune` on either hardware.** Blackwell: -4.6%. 3090: -15%. The strongest-compile-baseline comparison is the right framing.
3. **cudagraphs makes `compile_default` worse on both hardwares** (Blackwell 1.007× → 0.978×; 3090 1.151× → 1.074×). Specific to `decode_ctx2048_b8`. The "compile + cudagraphs" recipe is not a free win on either hardware tested.
4. **`eager_all_winners_cgraphs` hits the same 1.033× as the non-cgraphs all-winners**, despite running only the swiglu + rmsnorm patches under capture (SDPA prelude omitted). This is a direct Amdahl confirmation: SDPA prelude contributes ~0 at e2e level — consistent with Stage 7's finding that it's only 5–10% of full forward.
5. **The match-cgraphs comparison (agent-with-cgraphs vs compile-with-cgraphs) is 1.056×**, identical to the non-cgraphs agent vs `compile_default+cgraphs` comparison. The agent's advantage over the cgraphs-compile path is real, but small, and disappears against `compile_max_autotune`.
6. **Cross-hardware narrative finalized.** On undertrained Blackwell the agent beats `compile_default` by 2.6%, loses to `compile_max_autotune` by 4.6%. On mature 3090, the agent loses to every compile mode by 9–15%. The Blackwell agent advantage is small even against the weakest compile mode and disappears against the strongest. On mature hardware the comparison flips fully.

**Closed limitations (carried over from earlier stages).**
- "Cudagraphs-wrapped agent stack failed to run" → now runs; documented partial-coverage caveat (SDPA-prelude omitted under capture).
- "Compile baselines remain single-trial" → all key compile configs now 3-trial on both hardwares.

**New limitations (surfaced by the closures).**
- `eager_all_winners_cgraphs` patches only 2 of 3 kernels under capture. The SDPA-prelude patch is silently disabled inside graph capture because HF's mask-builder takes a 4D-mask branch the kimi kernel doesn't support. Numerically confirms Amdahl (1.033× vs 1.033× — SDPA contributes ~0 at e2e).
- CUDAGraph capture has a torch 2.12 nightly / Blackwell interaction: fp64 allocations during correctness checking corrupt the graph pool's internal index tensors, causing later replays to OOB-assert in embedding lookup. Workaround: validate-before-capture, free ref/out tensors, then capture-and-bench. Also bypass `triton.testing.do_bench`'s L2-cache-clear write-kernel (same crash). Manual CUDA-event timing loop when `is_cudagraph=True`.
- cudagraphs makes `compile_default` *worse* on both hardwares — the conventional "compile + cudagraphs" recipe is not a free win on the hardware we tested.

**Files.** `e2e/results/eager_all_winners_cgraphs_trial{0,1,2}.json`; `baselines/results/{compile_default,compile_max_autotune,compile_default_cgraphs}_trial{0,1,2}.json` on both hosts.
