# Agents vs `torch.compile` on Qwen3-1.7B

A focused replication of the [KernelBenchX](https://arxiv.org/abs/2605.04956) paper, designed to test one specific empirical question:

> When LLM coding agents (Claude 4.7 / Codex 5.5 / Kimi 2.6) replace `torch.compile`'s generated Triton kernels in a real transformer, what do you actually gain — measured honestly, and does it translate end-to-end?

Run on two hardwares: an undertrained Blackwell stack (DGX Spark, sm_121, May 2026) and a mature Ampere stack (RTX 3090, sm_86). Same model, same workloads, same harnesses.

The TL;DR (the framing this writeup can defend with a straight face): **on NVIDIA hardware where inductor's heuristics haven't been tuned yet (Blackwell sm_121), an LLM agent can sometimes find a structural rewrite (here, stacking QKV weights into a single cuBLAS GEMM) that the compiler does not attempt at the aten boundary. The integrated benefit is modest: the best agent stack beats `torch.compile`'s default mode by ~2% geomean, beats it with cudagraphs by ~6%, but *loses* to `compile_max_autotune` by ~5%. The headline is more sensitive to baseline configuration than to agent capability.** The structural rewrite is also hardware-specific: it inverts on mature Ampere where the separate-GEMM path is faster. Numbers below are 3-trial median ± MAD with `last_token_ids` pinned across configs and each trial run in a cold subprocess; small MADs (0.04–0.53 ms per workload) mean the headlines aren't noise-driven. See [Known limitations](#known-limitations).

The full experiment narrative — including every false start, methodology bug, and corrected number — lives in [`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md). The README below is the executive summary.

---

## Setup

**Model.** Qwen3-1.7B (dense decoder, GQA 16Q/8KV, BF16, RoPE, RMSNorm, SwiGLU MLP). 6 canonical workloads: prefill at seq 512 / 2048 batch 1, decode at context 512 / 2048 batch 1 / 8.

**Stack.** PyTorch 2.12 nightly cu128, Triton 3.7, transformers 5.8.1, Python 3.12/3.13.

**Primary baseline.** `torch.compile(model, mode="default", dynamic=False)` with `cudagraphs=False`. cudagraphs is excluded because it isn't a codegen optimization — agents could enable it on their replacement kernels too, so including it would confound the comparison.

**Agents.** Three CLIs, all in non-interactive mode, given the eager subgraph + canonical input shapes + a tightened correctness harness (strict tolerance + no-input-mutation + determinism), forbidden from using `torch.compile` / `@torch.compile` / `torch.jit`. They may write Triton, raw CUDA via `cpp_extension.load_inline`, or anything else compilable. Up to 5 iterations per task. The three CLIs are:

- **claude** (Claude 4.7)
- **codex** (Codex 5.5)
- **kimi** (Kimi 2.6)

**Tasks.** Three kernels chosen by how much of the prefill wall-time inductor spends on them: SwiGLU (5.7%), residual+RMSNorm (1.9% prefill / 12% decode), and the "SDPA prelude" mega-fusion of RMSNorm+QKV+RoPE+causal-mask (34% of prefill — biggest single Triton kernel inductor emits).

---

## Top-level results

### 1. Standalone kernel speedup vs inductor's *standalone microbench*

| task | Blackwell standalone | Blackwell e2e (geomean) | RTX 3090 standalone | RTX 3090 e2e |
|---|---|---|---|---|
| SwiGLU (kimi) | 1.06× | — | 1.04× | **0.98×** (regresses) |
| RMSNorm (best agent) | 1.21× (claude) | — | 1.08× (kimi) | n/a (hardfault) |
| **SDPA prelude (kimi)** | **3.91×** | — | **0.74×** (LOSS) | not run |

The standalone-to-integrated gap is even bigger than previously documented. On the 3090, the SwiGLU kernel that wins standalone (1.04×) actually **regresses 2% geomean** when integrated end-to-end (range 0.957–1.009× across 6 workloads), even though the kernel itself passes correctness. The standalone microbench overstates the integrated benefit by 3–100× on Blackwell and by enough to flip the sign on 3090.

The SDPA-prelude headline number deserves immediate skepticism, applied symmetrically. The 4046 μs inductor baseline comes from `extract/microbench_inductor.py:_bench_sdpa_prelude`, which **reconstructs the 8-kernel chain by hand as a Python loop** — each launch is a separate Python-level call. The 1034 μs agent kernel is a single in-graph dispatch. Some unmeasured fraction of that 3.91× is per-launch Python/dispatch overhead that in-graph inductor would amortize, not codegen quality. **The 3.91× number is not a clean measurement of pure codegen quality.** It bundles dispatch overhead, structural choice (stacked QKV), and microbench geometry. Treat it as evidence the agent *can* find structural rewrites the compiler doesn't try — not as a clean codegen-vs-codegen ratio.

The Blackwell-to-Ampere inversion (3.91× → 0.74×) is real and is the most informative single signal in the experiment: kimi's "stack QKV into one GEMM" is a *structural* decision — closer to algorithmic choice than to codegen quality — and it's hardware-specific. On Ampere's 82 SMs / 936 GB/s, cuBLAS parallelizes three separate small GEMMs better than one wider stacked GEMM, and Ampere's cuBLAS has well-tuned algorithms for those GQA-shape projections.

### 2. `torch.compile` vs eager (geomean across 6 workloads)

With `last_token_ids` pinned, `decode_ctx512_b1` now passes correctness across all configs — so all 6 workloads are legitimately in the headline. Single column:

| hardware | mode | geomean vs eager (all 6) |
|---|---|---|
| Blackwell (sm_121) | default | 1.011× |
| Blackwell (sm_121) | default + cudagraphs | **0.978×** (worse than no cudagraphs) |
| Blackwell (sm_121) | max-autotune | **1.089×** |
| RTX 3090 (sm_86) | default + cudagraphs | **1.067×** (range 0.621–1.967×) |

Takeaways:
- **On Blackwell, `compile_max_autotune` is the stronger baseline** — it wins 1.671× on `decode_ctx512_b1` and 1.242× on `decode_ctx2048_b1`. The earlier writeup only quoted `compile_default`. Reporting both is fairer.
- **`compile_default + cudagraphs` is *worse* than `compile_default` on Blackwell** (0.978× vs 1.011×). Cudagraphs has a systematic interaction with `decode_ctx2048_b8` — that workload regresses to 0.793× under cudagraphs (and to 0.621× on the 3090). The same pathology recurs on both hardwares.
- **The compiler still does more on the mature stack.** On Blackwell: inductor refuses `max_autotune_gemm` ("Not enough SMs" — 48 on GB10) so matmuls fall through to cuBLAS regardless of compile mode; LPDDR5X at 273 GB/s caps bandwidth headroom; Triton autotuner heuristics for sm_121 are still adjusting. On Ampere, the heuristics are tuned and the cuBLAS path is fast.

### 3. End-to-end best-agent-stack vs `torch.compile` (Blackwell)

Best agent stack = SwiGLU kimi + RMSNorm-pure claude + SDPA-prelude kimi. 3-trial median, `last_token_ids` pinned, each trial in a cold subprocess. All 6 workloads — no row dropped.

| baseline | agent stack vs that baseline (geomean) | range |
|---|---|---|
| eager | **1.033×** | 1.004–1.087× |
| compile_default | **1.022×** | 0.957–1.147× |
| compile_default + cudagraphs | **1.056×** | 0.937–1.271× |
| compile_max_autotune | **0.948× (agent loses)** | 0.651–1.255× |

**The agent stack only beats `torch.compile` in `default` mode (with or without cudagraphs). It loses to `compile_max_autotune` by ~5% geomean.** This is the single most important correction relative to earlier drafts: max-autotune is the strongest compile baseline and the agent stack does not beat it.

#### Per-workload speedup vs eager (Blackwell, all 6, 3-trial medians)

| workload | eager (ms) | agent stack | compile_default | cd + cudagraphs | compile_max_autotune |
|---|---|---|---|---|---|
| prefill_512_b1 | 253.49 | 1.010× | 1.055× | 1.003× | 1.043× |
| prefill_2048_b1 | 868.05 | 1.004× | 1.010× | 0.993× | 1.002× |
| decode_ctx512_b1 | 28.97 | 1.087× | 1.131× | 1.161× | **1.671×** |
| decode_ctx512_b8 | 143.46 | 1.011× | 0.999× | 0.966× | 0.958× |
| decode_ctx2048_b1 | 32.90 | 1.080× | 1.009× | 0.986× | **1.242×** |
| decode_ctx2048_b8 | 175.42 | 1.008× | **0.879×** | **0.793×** | **0.803×** |

Per-workload MADs are 0.04–0.53 ms — small relative to any of the gaps above. Variance is *not* what shifted the headline; the methodology corrections (pinning + cold processes) did.

The big regressors are on the compile side: `decode_ctx2048_b8` loses 12–21% under every compile mode, mirroring the same shape × batch pathology on the 3090 (0.621× under cudagraphs). The agent stack stays within 1% of eager on the worst-case workload (1.008×).

On 3090, the agent-stack-vs-eager number is incomplete: kimi's RMSNorm hard-faulted in e2e integration (hardcoded `hidden_size=2048`), so the obtainable agent stack covers SwiGLU alone, which **regresses 2% geomean (0.983×, range 0.957–1.009×)**. Compile-default+cudagraphs beats eager 1.067× geomean. The Blackwell agent-stack win does not survive to Ampere.

---

## What this means

**Compile-maturity hypothesis, supported but narrower than originally framed.** Agent value tracks the gap between the compiler's current heuristics and the achievable performance ceiling on the target hardware. On freshly-supported Blackwell where inductor hasn't been tuned yet, the agent stack delivers a small geomean win over `torch.compile` *in default mode* (with or without cudagraphs) — but it still loses to `compile_max_autotune`. On mature Ampere, the partial agent stack regresses outright.

**Both inner-loop AND structural agent wins are hardware-dependent.** The pre-registered prediction was that structural composition wins (like QKV stacking) would survive across hardwares because they're "compiler-doesn't-try" arguments. They don't. cuBLAS algorithm selection on a tuned platform can make the agent's bigger-GEMM choice worse, not better.

**The 3.91× headline is not pure codegen quality.** It's a Python-loop-of-8-kernels baseline against a single in-graph dispatch, on undertrained hardware, with the most favorable framing. The defensible claim is *"the agent identified a structural rewrite (stacked QKV) that inductor's stay-on-aten-boundary policy precludes."* That rewrite happens to be worth a lot on Blackwell-with-current-cuBLAS, nothing on Ampere-with-current-cuBLAS, and somewhere between depending on launch-amortization in real graphs.

---

## Methodology highlights

These are the methodology choices that the experiment had to make explicit — and that any future replication should make explicit too.

**1. Baseline choice is load-bearing.** The same SwiGLU kernel reports a 3.0× speedup against the in-model profiler aggregate (361 μs, includes launch/sync overhead across many invocations) and a 1.06× speedup against the standalone microbench (109.6 μs). Use the standalone microbench. The profiler aggregate overstates inductor's true codegen cost by ~3.3×.

**2. Tolerance gating is load-bearing.** With KBX "standard" tolerance (cos_sim ≥ 0.95), Codex's SwiGLU candidate "won" by replacing `sigmoid` with a hard-clamped linear approximation. Tightening to "strict" (cos_sim ≥ 0.99) + adding a no-input-mutation check + a determinism check collapsed the inflated 3.36× claim to a real 1.04× — and Codex *adapted* when its task prompt was rewritten to mention strict tolerance explicitly. Agent behavior tracks the loss function you actually expose.

**3. Cross-process determinism is non-obvious.** Python's built-in `hash()` is randomized per process via PYTHONHASHSEED. Using it to derive RNG seeds for "canonical inputs" produced different input tensors in two different scripts, manifesting as a fake `cos_sim 0.5` correctness failure that looked like a real `torch.compile` bug. Use `hashlib`.

**4. Amdahl is the rate-limiter on e2e wins.** SDPA prelude won 3.91× standalone on Blackwell. End-to-end on the prefill-heavy workload: 1.010×. The full agent stack — three kernels combined — only moves the geomean by 3.3% vs eager. The kernels are only a few percent of full-forward time each; even near-4× speedups on those slices yield a few-percent system-level win.

**5. Code review and bench are separable signals.** A blinded review (Claude rating all 10 kernels including inductor's, with metadata stripped) correctly picked the measured-fastest only **1 of 3 times.** The reviewer flagged 5 of 9 agent kernels as "wouldn't merge as-is" — and one of those (kimi's RMSNorm hardcoding `hidden_size=2048`) actually hard-faulted in cross-hardware e2e integration. Bench-or-die for performance; code-review-or-die for shippability.

**6. RMSNorm "win" is partly launch-overhead reduction, not pure codegen.** Eager `Qwen3RMSNorm.forward` is 4 kernel launches; the patched version is 1. Some fraction of the 1.043× geomean (RMSNorm-pure alone vs eager) is launch-overhead amortization that any competent kernel writer would get, separate from codegen quality. The codegen-quality slice — claude's kernel vs inductor's standalone microbench — is ~1.17×. We are no longer framing this as "agent beat inductor"; we are framing it as "agent collapsed 4 eager launches into 1, with a kernel that also happens to be ~17% better than inductor's standalone microbench at this shape."

**7. The SDPA-prelude baseline is a Python loop of 8 kernels.** `extract/microbench_inductor.py:_bench_sdpa_prelude` rebuilds the inductor prelude chain by hand; each launch is a separate Python-level call. The 1034 μs agent kernel is a single dispatch. Per-launch dispatch overhead is in the gap, and we cannot cleanly separate it from codegen quality from this measurement.

---

## CLI patterns observed (anecdotal, n=9 trials per CLI)

In this small sample (3 tasks × 2 hardwares × 1 attempt window per cell, with 3090 covering kimi only — so n=9 per CLI on Blackwell, n=3 for kimi on 3090, n=0 for the others on 3090), we observed the following tendencies. **A larger study with multiple trials per cell would be needed to claim stable vendor fingerprints.** Treat as hypotheses, not findings:

- **claude** — middle wall-clock, middle kernel quality; tends to add host-side machinery (cached launchers, module-level globals).
- **codex** — fastest wall-clock when it stops early; terse code; tolerance-gamed on Stage 3a when the prompt didn't explicitly forbid it; adapted cleanly in 3d once the prompt did.
- **kimi** — slowest wall-clock by 3–10×; produced the fastest kernel on every Blackwell task (3/3); most aggressive structural rewrites; most hardcoded shape constants (and the constants bit cross-hardware integration).

**All three CLIs stayed inside the Triton + cuBLAS envelope across every task.** No raw CUDA, CUTLASS, ThunderKittens, PTX, or Flash-Attention-style absorbed-prelude. This is partly a real observation about agent defaults — and partly an artifact of the task prompts. The `task.md` "Allowed approaches" section lists Triton first ("matches what inductor produced"), CUDA via `cpp_extension` second, "any other approach" third. A prompt that opened with "consider CUTLASS, ThunderKittens, or absorbed-prelude Flash-Attention designs" might produce a different distribution. See *Prompt evolution and experimenter influence* in [`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md).

---

## Known limitations

Caveats a skeptical reader should weigh before any of the numbers above:

- **Cudagraphs-wrapped agent stack failed to run.** The `eager_all_winners_cgraphs` config crashed with a NoneType in the correctness check — raw CUDAGraph capture's return-tensor plumbing isn't right yet for the patched stack. The `compile_default + cudagraphs` cell worked. Agent + cudagraphs is a known follow-up; we cannot currently claim "agent stack with cudagraphs" numbers.
- **3-trial vs single-trial mixed methodology.** Agent stack and eager are 3-trial median ± MAD with cold subprocesses per trial. `compile_default` and `compile_max_autotune` baselines come from `baselines/run_compile.py` and are single-trial. Per-workload MADs on the replicated runs are 0.04–0.53 ms (well under any headline delta), so cross-methodology comparison is probably fine, but flag it.
- **decode_ctx512_b1 correctness flag is fixed.** Pinning `last_token_ids` across configs eliminates the bf16-drift argmax flip that previously failed cos_sim. All 6 workloads now legitimately pass and are included in the headline; no "drop a row" caveat.
- **cudagraphs is OFF on the agent side, ON on one compile baseline.** Because the agent-cudagraphs cell crashed, the closest apples-to-apples is agent-vs-compile-default (1.022×). Agent-vs-compile-default-cudagraphs (1.056×) confounds two things; quote it but note the caveat.
- **compile_default vs compile_max_autotune.** Earlier writeup quoted `compile_default` only. We now co-quote `compile_max_autotune`, which is the stronger compile baseline on Blackwell (1.089× vs 1.011× geomean vs eager) and beats the agent stack 1.055× geomean (agent loses by 5%).
- **The 3.91× SDPA-prelude headline includes per-launch dispatch overhead.** Inductor's standalone microbench is a Python loop of 8 kernels; the agent kernel is one dispatch. We cannot cleanly separate codegen quality from launch amortization in that ratio.
- **The RMSNorm "1.043× e2e" win is partly multi-launch eager replacement**, not pure agent-vs-inductor codegen. The codegen-quality slice is ~1.17×; the rest is collapsing 4 eager launches into 1.
- **Prompts evolved in response to observed agent failures.** Stage 3a SwiGLU prompt did not forbid approximation; Stage 3d added explicit strict-tolerance + no-mutation language; Stage 3e RMSNorm inherited that framing and mentioned a specific inductor behavior; Stage 5a SDPA prelude labeled "three QKV GEMMs" as "the big knob" in the prompt. We cannot cleanly separate "agent figured out X" from "prompt told it to figure out X." A cleaner replication would freeze prompts before any agent runs, or have prompts written by an independent party. See *Prompt evolution and experimenter influence* in EXPERIMENT_LOG.
- **All three CLIs converged inside the Triton+cuBLAS envelope.** Partly observation, partly prompt framing (Triton listed first). A different prompt might produce a different distribution.
- **Per-CLI characterizations rest on n=9 (Blackwell) and n=3 (3090, kimi only)** — anecdotal, not stable fingerprints.
- **One model, two hardwares, three CLIs, three kernels.** Findings about cross-CLI patterns, kernel-difficulty headroom, and compile maturity rest on this small N. No cross-model validation (Llama / Mistral / Mixtral).
- **kimi-cli version differs between hardwares** (1.37 Blackwell, 1.44 3090). We can't separate "kimi-on-Ampere" from "newer-kimi-cli." Direction of the cross-hardware effect is large enough that this is unlikely to flip the qualitative result.
- **kimi-3090 RMSNorm hard-fault is sm_86-specific, not generic shape-brittleness.** A multi-shape harness extension showed that kimi_v1's RMSNorm passes all three shapes on Blackwell sm_121 (including the head_dim=128 shape that hard-faulted on 3090). The fault is therefore specific to sm_86 — probably shared-memory limits or launch-grid choices — not a generic "hardcoded constants" failure. The earlier Stage 8 framing was too broad.
- **Tokenization of agent cost is not yet implemented.** "Kimi is the slowest agent" is measured in wall-clock, not tokens or dollars.

---

## Directory layout

```
.
├── env/                     # env verification + pinned requirements
├── workload/                # Qwen3-1.7B loader, canonical inputs (hashlib-seeded),
│                            #   correctness checks (fp64 cosine + strict tier)
├── baselines/               # eager + torch.compile (3 modes × 6 workloads)
│   ├── run_eager.py
│   ├── run_compile.py
│   └── results/             # JSON per config (heavy outputs gitignored)
├── extract/                 # inductor kernel extraction + standalone microbench
│   ├── dump_inductor.py
│   ├── rank_by_walltime.py
│   ├── microbench_inductor.py
│   ├── microbench_inductor.json   # the codegen-vs-codegen baseline numbers
│   ├── manifest.json
│   ├── ranking.md, ranking_*.json
│   └── kernels/             # 24 inductor-emitted Triton kernels with metadata
├── agent_loop/              # the per-kernel agent loop
│   ├── wrappers/            # claude_wrap.sh, codex_wrap.sh, kimi_wrap.sh
│   ├── run_one.py           # orchestrator: sandbox + timing + trajectory log
│   ├── tasks/               # task.md, reference.py, harness.py per kernel
│   │   ├── swiglu/
│   │   ├── rmsnorm/
│   │   └── sdpa_prelude/
│   ├── sandbox/             # per-run agent workdirs; final candidate.py kept
│   └── runs/                # result.json per run (raw CLI transcripts gitignored)
├── e2e/                     # patch eager Qwen3 with winning kernels, bench
│   ├── patches.py           # install_swiglu_kimi, install_rmsnorm_claude_pure, etc.
│   ├── run_e2e.py
│   ├── kernels/             # the winning candidates copied here
│   ├── results/             # per-config JSON
│   └── summary.md
├── review/                  # blinded code review + LoC/CC/MI stats
│   ├── reviewer.py          # claude reviewing all 10 kernels blinded
│   ├── stats.py             # mechanical stats via radon
│   ├── reviews/             # raw reviews + unblinded keys
│   ├── stats.json, stats.md
│   └── SUMMARY.md
└── experiments/
    └── EXPERIMENT_LOG.md    # narrative log: every stage, hypothesis,
                              #   what we tried, what we found, takeaways
```

---

## Reproducing the experiment

Rough order. Each step has a script that mirrors the experiment-log stage:

```bash
# 0. env
python -m venv .venv && source .venv/bin/activate
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r env/requirements.txt
python env/verify_env.py

# 1. baselines (eager + 3 compile modes × 6 workloads)
python baselines/run_eager.py
python baselines/run_compile.py

# 2. extract inductor kernels + rank + microbench
python extract/dump_inductor.py
python extract/rank_by_walltime.py
python extract/match_kernels.py
python extract/microbench_inductor.py

# 3. agent loops per (cli, task)
python agent_loop/run_one.py --cli kimi --task swiglu --max-attempts 5 --run-id swiglu_kimi
python agent_loop/run_one.py --cli kimi --task rmsnorm --max-attempts 5 --run-id rmsnorm_kimi
python agent_loop/run_one.py --cli kimi --task sdpa_prelude --max-attempts 5 --run-id sdpa_prelude_kimi
# (analogous for claude, codex)

# 4. end-to-end integration
python -m e2e.run_e2e

# 5. code review + stats
python review/stats.py
python review/reviewer.py
```

The agent loops require the corresponding CLI installed and authenticated. Wall-clock budget on Blackwell: ~6 hours for the full experiment. On 3090: ~3 hours.

For the full set of caveats, see [Known limitations](#known-limitations) above. For every choice, every false start, and every corrected number, see [`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md).
