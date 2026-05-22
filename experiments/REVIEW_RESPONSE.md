# Response to the review

A reader gave the [original writeup](https://github.com/ianbarber/kernelbench-agents-vs-compile/commit/901a5a0) a sharp critique. Most of it landed. This file walks through each of the 8 specific follow-ups the review proposed and what we did about them.

Repo state: closed 5 of 8 follow-ups with new data; closed 1 partially through restructuring; left 2 explicitly open with honest documentation. The headline numbers tightened materially — the original 1.037× agent-vs-`compile_default` becomes 1.026× under replicate methodology, and the **new finding (agent loses to `compile_max_autotune` by 4.6% on Blackwell, by 15% on 3090) was previously invisible** because we only quoted `compile_default`.

Commits referenced below:
- `901a5a0` — initial commit (the version the review was written against).
- `38487b1` — Tier 1 writeup + Tier 2 code (cudagraphs configs, replicate mode, pinned tokens, multi-shape RMSNorm harness, optional sdpa import).
- `3299302` — symmetric compile baselines + cudagraphs cell + final corrected numbers.

---

## 1. ✅ Replicate everything 3× from cold processes, report median ± MAD

**Done.** Added `--trials N` to `e2e/run_e2e.py` (commit `38487b1`); each trial is a cold `subprocess.call` with the inductor cache wiped at startup, then the parent aggregates median + MAD per workload. Trial files preserved in `e2e/results/*_trial*.json`.

3-trial replicated on Blackwell: `eager`, `eager_all_winners`, `eager_all_winners_cgraphs`, `compile_default`, `compile_max_autotune`. On 3090: `eager`, `eager_swiglu_kimi`, `compile_default`, `compile_max_autotune`.

**Result:** MADs are 0.04–0.53 ms per workload — well under all the headline deltas. **The methodology correction (pinning, cold processes) shifted the headline, not the noise.** Single-trial numbers were directionally fine; they were just missing a systematic bias the review correctly identified.

The `compile_default+cgraphs` cell remains single-trial (commit `3299302`); rerunning it 3× is on the list but unlikely to change the picture given the variance already characterized.

## 2. ✅ Add the cudagraphs cells

**Done, with one caveat.** Both `compile + cudagraphs` and `agent + cudagraphs` cells ran on Blackwell. `compile_default_cgraphs` ran on 3090.

The agent-with-cudagraphs implementation took two iterations:
1. First attempt crashed with `NoneType` in correctness (the captured forward's return-tensor wasn't being propagated back).
2. Final fix (commit `3299302`): static output buffer captured during `with torch.cuda.graph(g)`, returned from `replay()`. Plus a torch-2.12-nightly-specific workaround — fp64 allocations during the correctness check corrupted the graph pool's internal index tensors, so we validate correctness **before** capture, free the ref/cand tensors, then capture-and-bench with a manual CUDA-event timing loop.

**Caveat:** `eager_all_winners_cgraphs` runs only swiglu + rmsnorm under capture. The SDPA-prelude patch's `use_kimi` guard takes a different mask-builder branch inside graph capture (HF emits a 4D additive mask under capture vs the `attention_mask=None` it uses outside), and the kimi kernel produces structurally wrong output for the 4D mask. We documented this in `Known limitations`. That this stripped-down stack still hits 1.033× vs eager — same as full all_winners without cudagraphs — independently confirms the Amdahl finding from Stage 7: SDPA prelude contributes ~0 at e2e level because it's only 5–10% of forward time.

**Headline result:** `cudagraphs` makes `compile_default` *worse* on both hardwares (Blackwell 1.007× → 0.978×, 3090 1.151× → 1.074×). decode_ctx2048_b8 dominates the regression on both. This was previously unmeasured.

## 3. ✅ Run `compile_max_autotune` as a co-baseline

**Done, on both hardwares, replicated.** Added as proper e2e config in `e2e/run_e2e.py` (commit `3299302`).

**Result:** `compile_max_autotune` is the strongest compile baseline on both hardwares (Blackwell 1.083× vs eager, 3090 1.161×) and the **agent stack loses to it on both hardwares** (Blackwell -4.6%, 3090 -15%). This is the single biggest correction the symmetric baselines surfaced. The original writeup quoted only `compile_default` for the e2e comparison; that was the most favorable available baseline. With `compile_max_autotune` in the table, the "agents beat compile" headline only survives against `compile_default` on Blackwell, and only by 2.6%.

## 4. ✅ Pin `last_token_ids` externally on decode

**Done.** Added `pin_last_token=True` path to `workload/inputs.py` (commit `38487b1`). Pinned tokens stored at `baselines/results/eager_last_token_ids/<workload>.pt`, derived once from eager via `tools/regenerate_pinned_tokens.py`. All e2e configs decode from the same starting token; the KV cache itself is still built using the model under test (cache contents must be consistent with the patched kernels).

**Result:** `decode_ctx512_b1` now passes standard correctness across all patched configs on Blackwell. **No longer dropped from the headline geomean.** The 1.046× we previously quoted under "all 6 workloads" was confounded by including a row whose correctness was failing; under pinning it's a legitimate 1.033×.

## 5. ✅ Test each agent kernel on the full shape distribution it sees in-model

**Done, with a finding that contradicts our prior interpretation.** Multi-shape RMSNorm harness in `agent_loop/tasks/rmsnorm/harness.py` (commit `38487b1`): tests `(1, 512, 2048)`, `(8, 1, 2048)`, and `(1, 512, 128)` (head_dim path used by q_norm / k_norm) before benching. New verdict `FAIL_SHAPE_GENERALIZATION` in the ladder.

**The contradicting finding:** kimi's RMSNorm kernel actually passes all three shapes on Blackwell sm_121 — the constexpr-N kernel recompiles correctly per shape. The 3090 hard-fault we previously attributed to "hardcoded `hidden_size=2048`" is therefore **sm_86-specific** (likely shared-memory limits / launch grid quirks), not generic shape brittleness. We updated Known limitations to reflect this.

The harness extension still has defense-in-depth value — would catch the next class of shape-brittle agent kernel before integration.

## 6. ❌ Freeze the task prompts before any agent runs

**Not done.** Re-running all agent loops with frozen prompts would cost ~13 hours of agent compute and we judged the cost-benefit unfavorable for the writeup's claims.

**Mitigation:** Documented the prompt-evolution timeline honestly in `EXPERIMENT_LOG.md` "Prompt evolution and experimenter influence" subsection. Specifically: Stage 3a SwiGLU task.md had no mention of strict tolerance or mutation rules (Codex tolerance-gamed); Stage 3d added explicit strict-tolerance + no-mutation + no-approximation language (Codex adapted); Stage 5a SDPA-prelude task.md lists inductor's emit explicitly and labels the three QKV GEMMs as "the big knob" (the optimization kimi went on to make).

We flag this as a Known limitation: the experiment cannot cleanly separate "the agent figured out X" from "the prompt told it to figure out X." A cleaner replication would freeze the prompts or use a separate prompt author.

## 7. △ Partially closed via restructuring

**Workload weighting alternatives:** the v1 writeup pass added a section with uniform / prefill-heavy / decode-heavy geomeans. The v2 pass removed that section because pinning closed the "drop decode_ctx512_b1" caveat, so a single all-6 geomean is now legitimately the honest number.

In place of the weighting-sensitivity section, the README now prints the **per-workload speedup table directly** (`prefill_512_b1: 1.010×, decode_ctx512_b1: 1.087×, decode_ctx2048_b8: 1.008×, ...`). A reader weighting prefill more heavily can compute their own geomean. We think this is more informative than the three-row weighting table; happy to add the weighting table back if that's preferred.

## 8. ❌ Run more than one trial per (CLI, task)

**Not done.** Per-CLI claims still rest on n=9 (3 CLIs × 3 tasks × 1 trial). Running a full n=3-per-cell matrix would be 27 agent runs × ~30 min average = ~13 hours of agent compute.

**Mitigation:** Softened the per-CLI fingerprint claims in the README to anecdotal: *"In this small sample (n=9 per CLI across both hardwares, single attempt window), we observed... A larger study with multiple trials per cell would be needed to claim stable vendor fingerprints."* This is now explicitly framed as observation, not characterization.

---

## What the post can now claim with a straight face

> "On undertrained Blackwell (sm_121, May 2026), an LLM agent can sometimes find a structural rewrite (here, stacking QKV weights into a single cuBLAS GEMM) that the compiler does not attempt at the aten boundary. The integrated benefit is modest: agent stack beats `torch.compile`'s default mode by 2.6% geomean, beats it with cudagraphs by 5.6%, but **loses to `compile_max_autotune` by 4.6%**. On mature Ampere (RTX 3090, sm_86) the agent stack loses to every compile mode by 9–15%. The headline is more sensitive to baseline configuration than to agent capability, and the agent stack never beats `compile_max_autotune` on either hardware tested."

That framing follows directly from the reviewer's "defensible version" and is supported by the data in `e2e/results/*.json`.

## Thanks

The review made the writeup substantially more honest. The single most useful thing was insisting on `compile_max_autotune` as a co-baseline; without that, we'd have continued quoting the `compile_default`-only headline and implicitly overstating the agent-vs-compiler claim. Two follow-ups (frozen prompts, n>1 per CLI cell) remain explicitly open and are flagged as such in Known limitations.
