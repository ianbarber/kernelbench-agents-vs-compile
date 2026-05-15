# Agents vs `torch.compile` on Qwen3-1.7B

A focused replication of the [KernelBenchX](https://arxiv.org/abs/2605.04956) paper, designed to test one specific empirical question:

> When LLM coding agents (Claude 4.7 / Codex 5.5 / Kimi 2.6) replace `torch.compile`'s generated Triton kernels in a real transformer, what do you actually gain — measured honestly, and does it translate end-to-end?

Run on two hardwares: an undertrained Blackwell stack (DGX Spark, sm_121, May 2026) and a mature Ampere stack (RTX 3090, sm_86). Same model, same workloads, same harnesses.

The TL;DR: **agent value scales inversely with how mature the compiler is on the target hardware.** On Blackwell, a careful agent stack beats `torch.compile` end-to-end by 3.7%. On Ampere, `torch.compile` beats the same agent stack by ~18% — and the kernel that crushed inductor on Blackwell (a stacked-QKV GEMM trick) actually loses 26% standalone on Ampere. Both inner-loop AND structural agent wins turn out to be hardware-dependent.

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

| task | Blackwell | RTX 3090 |
|---|---|---|
| SwiGLU (kimi) | 1.06× | 1.04× |
| RMSNorm (best agent) | 1.21× (claude) | 1.08× (kimi) |
| **SDPA prelude (kimi)** | **3.91×** | **0.74×** (LOSS) |

The SDPA-prelude flip is the centerpiece. Kimi's winning move on Blackwell was stacking the three QKV projection weights into a single cuBLAS GEMM, saving dispatch overhead + intermediate buffer. On Ampere with 82 SMs and 936 GB/s GDDR6X, cuBLAS parallelizes 3 separate small GEMMs across the chip better than one wider stacked GEMM does — *and* Ampere's cuBLAS has well-tuned algorithms for those exact GQA-shape projections. The optimization inverts.

### 2. `torch.compile` vs eager (geomean across all 6 workloads)

| hardware | compile geomean speedup vs eager |
|---|---|
| Blackwell (sm_121) | **1.009×** (essentially a wash) |
| RTX 3090 (sm_86) | **~1.20×** |

The compiler delivers ~200× more speedup on the mature stack. On Blackwell:
- Inductor refuses `max_autotune_gemm` (`Not enough SMs to use max_autotune_gemm mode` — only 48 SMs on GB10) → all matmuls fall through to cuBLAS regardless of compile mode. 49% of prefill wall time is structurally unavailable to compile's codegen.
- LPDDR5X at 273 GB/s leaves little headroom for fusion (bandwidth ceiling close to eager).
- Triton autotuner heuristics for sm_121 register pressure are still adjusting (observed `R0_BLOCK` scaling 1024 → 512 mid-compile).

### 3. End-to-end best-agent-stack vs `torch.compile`

| hardware | agent stack vs eager (geomean) | torch.compile vs eager | agent stack vs compile |
|---|---|---|---|
| Blackwell | **1.046×** (wins 5/6 workloads) | 1.009× | **1.037× — agent wins** |
| RTX 3090 | ≤1.01× (partial; RMSNorm crashed on 3090) | ~1.20× | **~0.85× — compile wins** |

The agent-stack-beats-compile claim is **hardware-bound**. It's real on Blackwell. It does not survive to Ampere.

---

## What this means

**Compile-maturity hypothesis, confirmed.** Agent value tracks the gap between the compiler's current heuristics and the achievable performance ceiling on the target hardware. On freshly-supported hardware where the compiler hasn't been tuned yet, agents can find wins inductor's heuristics miss. On mature hardware where the compiler has had years of attention, that gap shrinks toward zero — and an agent's hardware-specific bet can actively backfire (as the QKV-stacking on Ampere did).

**Both inner-loop AND structural agent wins are hardware-dependent.** The pre-registered prediction was that structural composition wins (like QKV stacking) would survive across hardwares because they're "compiler-doesn't-try" arguments. They don't. cuBLAS algorithm selection on a tuned platform can make the agent's bigger-GEMM choice worse, not better. There's no kind of optimization that's automatically hardware-portable.

**The 3.91× headline number is a Blackwell-specific artifact.** Honest framing for a writeup: "on undertrained hardware, agent intervention can find substantial wins in cross-op compositions the compiler doesn't try; those wins do not necessarily port to mature hardware."

---

## Methodology highlights

These are the methodology choices that the experiment had to make explicit — and that any future replication should make explicit too.

**1. Baseline choice is load-bearing.** The same SwiGLU kernel reports a 3.0× speedup against the in-model profiler aggregate (361 μs, includes launch/sync overhead across many invocations) and a 1.06× speedup against the standalone microbench (109.6 μs). Use the standalone microbench. The profiler aggregate overstates inductor's true codegen cost by ~3.3×.

**2. Tolerance gating is load-bearing.** With KBX "standard" tolerance (cos_sim ≥ 0.95), Codex's SwiGLU candidate "won" by replacing `sigmoid` with a hard-clamped linear approximation. Tightening to "strict" (cos_sim ≥ 0.99) + adding a no-input-mutation check + a determinism check collapsed the inflated 3.36× claim to a real 1.04× — and Codex *adapted* when its task prompt was rewritten to mention strict tolerance explicitly. Agent behavior tracks the loss function you actually expose.

**3. Cross-process determinism is non-obvious.** Python's built-in `hash()` is randomized per process via PYTHONHASHSEED. Using it to derive RNG seeds for "canonical inputs" produced different input tensors in two different scripts, manifesting as a fake `cos_sim 0.5` correctness failure that looked like a real `torch.compile` bug. Use `hashlib`.

**4. Amdahl is the rate-limiter on e2e wins.** SDPA prelude won 3.91× standalone on Blackwell. End-to-end on the prefill-heavy workload: 1.010×. The kernel is only ~5-10% of full-forward time; a near-4× speedup on that fraction yields a few-percent system-level win.

**5. Code review and bench are separable signals.** A blinded review (Claude rating all 10 kernels including inductor's, with metadata stripped) correctly picked the measured-fastest only **1 of 3 times.** The reviewer flagged 5 of 9 agent kernels as "wouldn't merge as-is" — and one of those (kimi's RMSNorm hardcoding `hidden_size=2048`) actually hard-faulted in cross-hardware e2e integration. Bench-or-die for performance; code-review-or-die for shippability.

---

## CLI patterns observed (stable across 2 hardwares × 3 tasks)

- **claude** — middle wall-clock, middle kernel quality, builds host-side machinery (cached launchers, module-level globals). Cost-quality median.
- **codex** — fastest wall-clock when it stops early at "good enough"; terse code, minimal boilerplate; will tolerance-game if the prompt permits it, adapts cleanly when strict-tolerance is stated.
- **kimi** — slowest wall-clock by 3-10×; produces the best kernel on every task on every hardware; most aggressive structural rewrites; most hardcoded shape constants (and the constants bite cross-hardware).

Nobody tried raw CUDA, CUTLASS, ThunderKittens, PTX, or Flash-Attention-style absorbed-prelude designs across any of the tasks. All three converged inside the Triton + cuBLAS envelope.

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

---

## Scope caveats

- **One model, two hardwares, three CLIs, three kernels.** Findings about cross-CLI patterns, kernel-difficulty headroom, and compile maturity rest on this small N. Replicating on a different architecture (MoE / hybrid / multimodal) would test how much of this is Qwen3-specific.
- **kimi-cli version differs between hardwares** (1.37 on Blackwell, 1.44 on 3090). We can't fully separate "kimi-on-Ampere produced different kernels" from "newer kimi-cli produces different kernels," though the *direction* of the cross-hardware effect is large enough that the kimi-version delta is unlikely to dominate.
- **`torch.compile` was tested with cudagraphs OFF only.** With cudagraphs ON, compile's e2e numbers improve substantially — but the comparison gets confounded (agents could also wrap their kernels in cudagraphs). The honest codegen-vs-codegen comparison is what's reported.
- **No cross-model validation.** Same conclusions on Llama / Mistral / Mixtral are TBD.
- **Tokenization of agent costs is not yet implemented.** "Kimi is the slowest agent" is measured in wall-clock, not tokens or dollars.

For full detail on every choice, every false start, and every corrected number, see [`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md).
