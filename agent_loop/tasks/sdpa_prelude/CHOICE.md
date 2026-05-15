# SDPA-prelude task: scope decision and inductor contract

## Picked

The full **SDPA prelude** for the first attention block of Qwen3-1.7B
at prefill_512_b1 — every op from post-residual-RMSNorm `hidden_states`
to the inputs of `aten._scaled_dot_product_efficient_attention`:

  1. **Q/K/V projection GEMMs** (3 × `aten::mm` via cuBLAS)
  2. **Q per-head RMSNorm + RoPE-apply** (`triton_per_fused..._1`)
  3. **K per-head RMSNorm + RoPE-apply** (`triton_per_fused..._2`)
  4. **KV head expansion (GQA: 8 → 16 heads)** via clone+layout
     (`triton_poi_fused..._where_3`, called twice — K and V)
  5. **Causal + padding mask construction**
     (`triton_poi_fused..._where_4`, writes the mask to two output buffers
     that are byte-identical; we expose just one to the candidate)

## Why this scope

The original target was `_where_3` alone (24.99% of prefill_512_b1).
Reading the kernel.py:

```python
# Just an indexed clone with x2 (head axis) // 2 — GQA replication.
tmp0 = tl.load(in_ptr0 + (x0 + 128*(x2 // 2) + 1024*x1 + 2097152*x3), None).to(tl.float32)
tl.store(out_ptr0 + (x4), tmp0, None)
```

So `_where_3` is **purely a layout transform** (8-KV-head → 16-Q-head
expand + transpose to contiguous `(B, H, S, D)`). On its own it's a
memcpy-with-strided-load: ~24 us per call at prefill_512_b1
(xnumel=1,048,576). The "24.99%" share comes from being invoked ~560×
across all decode/prefill workloads in the profile.

`_where_4` is similar in spirit: build a (1,1,S,S) bf16 mask, write to
two buffers.

The actual *useful work* — the part where a fast custom kernel could win
real time — lives in the upstream `per_fused` kernels (RMSNorm + RoPE)
and the GEMMs. Tasking the agent with `_where_3` alone would just be a
"write a slightly cleverer memcpy" challenge.

So we expand the scope to the **full prelude** an attention block uses
before SDPA. This:

  - Captures the whole 5.4% + 24.99% + 9.24% ≈ 40% of prefill_512_b1
    that inductor spends preparing tensors for SDPA.
  - Makes the **Flash-Attention-style absorbed-prelude** approach
    natural (an agent that wants to fuse Q/K/V projection + RoPE + mask
    construction together can do so — and most importantly, can avoid
    materialising the GQA-replicated K/V, which is the biggest waste).
  - Honestly reflects what would have to change in the model code for
    this region to get faster: not "improve `_where_3`", but "rewrite
    the whole pre-SDPA chain to skip the materialisations".

## Contract (the task interface)

### Inputs to `run(...)`

| name            | shape                | dtype   | notes                            |
|-----------------|----------------------|---------|----------------------------------|
| `hidden_states` | (1, 512, 2048)       | bf16    | post-residual-RMSNorm hidden     |
| `w_q`           | (2048, 2048)         | bf16    | (out_features, in_features)      |
| `w_k`           | (1024, 2048)         | bf16    |                                  |
| `w_v`           | (1024, 2048)         | bf16    |                                  |
| `w_q_norm`      | (128,)               | bf16    | per-head Q RMSNorm scale         |
| `w_k_norm`      | (128,)               | bf16    | per-head K RMSNorm scale         |
| `inv_freq`      | (64,)                | fp32    | RoPE inv-freqs (head_dim // 2)   |
| `position_ids`  | (1, 512)             | int64   | typically `arange(512)`          |
| `attention_mask`| (1, 512)             | int64   | 1=keep, 0=pad                    |
| `eps`           | scalar               | float   | 1e-6 by convention               |

### Outputs (4-tuple)

| name   | shape                | dtype  | notes                                       |
|--------|----------------------|--------|---------------------------------------------|
| `q`    | (1, 16, 512, 128)    | bf16   | per-head RMSNorm-ed and RoPE-applied        |
| `k`    | (1, 16, 512, 128)    | bf16   | RMSNorm-ed, RoPE-applied, GQA-expanded ×2   |
| `v`    | (1, 16, 512, 128)    | bf16   | GQA-expanded ×2 (no norm, no RoPE)          |
| `mask` | (1, 1, 512, 512)     | bf16   | additive (0 / -inf), causal + padding       |

### Ops the candidate must implement end-to-end

  1. `q_flat = hidden_states @ w_q.T`
  2. `k_flat = hidden_states @ w_k.T`
  3. `v_flat = hidden_states @ w_v.T`
  4. Reshape `q_flat -> (1, 16, 512, 128)`, transpose head/seq axes.
     Same for k_flat → (1, 8, ...), v_flat → (1, 8, ...).
  5. Per-head RMSNorm over the head_dim=128 axis, with fp32
     accumulation (matches inductor: `s.pow(2).mean / sqrt(... + eps)`).
     Applied to `q` and `k`, NOT to `v`.
  6. Apply RoPE to `q` and `k`. RoPE construction: `freqs = pos[..., None] *
     inv_freq[None]`, `emb = cat([freqs, freqs], dim=-1)`, then
     `cos/sin = emb.cos()/emb.sin()`. RoPE-apply: `x * cos +
     rotate_half(x) * sin` where `rotate_half([a, b]) = [-b, a]`.
  7. GQA-expand K and V from 8 heads to 16: each KV head is paired with
     `groups = NUM_Q_HEADS / NUM_KV_HEADS = 2` consecutive Q heads.
     Inductor's `_where_3` uses `head // 2` indexing — so head 0 and 1
     of the Q-aligned K share the same KV head 0, etc.
  8. Causal + padding mask of shape `(1, 1, 512, 512)`:
     `mask[b, 0, q, k] = 0 if (k <= q) and (attn_mask[b, k] != 0) else -inf`.

## Could we pin the contract from inductor's source?

**Yes**, by reading `extract/inductor_debug/cache/b2/.../output_code.py`
(the prefill_512_b1 cache) we trace exactly which buffers feed
`aten._scaled_dot_product_efficient_attention.default`:

```
buf14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(
    buf10,  # Q   shape (1, 16, 512, 128) bf16
    buf11,  # K   shape (1, 16, 512, 128) bf16  -- GQA-expanded from buf8 (1,8,...)
    buf12,  # V   shape (1, 16, 512, 128) bf16  -- GQA-expanded from buf9 (1,8,...)
    reinterpret_tensor(buf13, (1, 16, 512, 512), ...),  # mask
    False,
    scale=0.08838834764831845,
)
```

So `(q, k, v, mask)` is exactly what SDPA takes. The agent's `run` must
produce these four tensors, end of story.

## Simplifications versus inductor's literal behaviour

We do simplify slightly:

  - **One mask buffer instead of two.** Inductor's `_where_4` writes the
    same mask into `buf13` *and* `buf39` because two consecutive
    attention blocks consume mask buffers that get destroyed in-place.
    We expose a single mask — the agent only computes it once. The
    correctness check is against this one mask; the harness doesn't
    test the "double write" peculiarity.

  - **`inv_freq` is exposed as a precomputed tensor.** Inductor still
    runs the `arange(64) * 1/theta` chain inline in `_per_..._1` and
    `_per_..._2`. We give the agent the precomputed `inv_freq` so they
    don't have to repeat that microscopic broadcast — it shaves a few
    cycles from the inner kernel and is the path Qwen3 actually takes
    in eager (the rotary module precomputes inv_freq once and reuses
    it). The agent could also accept the cos/sin tables directly if
    they prefer, but exposing `inv_freq + position_ids` keeps the task
    closer to "compute the RoPE rotation per call".

  - **The Q-projection mm output stride.** Inductor stores Q-proj output
    as `(512, 2048)` flat bf16 and then per-head-reshapes inside the
    Triton kernel. We let the agent pick whatever intermediate layout
    they want; only the final `(1, 16, 512, 128)` Q tensor must match.

  - **Causal mask: bool vs additive.** The agent must produce the
    *additive* (0 / -inf) bf16 mask, because that's what SDPA expects.
    A bool mask is not equivalent.

## Could `_where_3` and `_where_4` be combined into one task?

Yes — they're both consumed by the same SDPA call. Splitting them into
two tasks would lose the cross-kernel optimisation opportunity (an
agent that materialises K/V *and* computes a mask could skip
materialisation entirely if it does its own attention). So we bundle
both into this single `sdpa_prelude` task.

## "Outer" inputs from upstream `aten::mm`

The Q/K/V projection GEMMs (`extern_kernels.mm`) are *part* of this
task — the agent receives the weight tensors and is expected to either
call cuBLAS themselves (via `torch.mm`, `torch.matmul`,
`F.linear`, etc.) or write a custom GEMM. Calling cuBLAS is allowed
and faithful to what inductor does. Writing a custom GEMM that beats
cuBLAS at this shape on Blackwell bf16 is exceptionally hard — agents
will almost certainly call cuBLAS.

Note on the GEMM stride trick: a naive `torch.mm(x, w.T)` on this
hardware hits a slow cuBLAS algo (~3 ms for a (512,2048)@(2048,2048)
bf16). Inductor uses `reinterpret_tensor(w, (in, out), (1, in))`
(a stride-swap view) which routes to the fast algo (~600 µs).
The reference does the same trick via `torch.as_strided`. Agents
following the reference will naturally inherit the fast path.

## Inductor microbench results (this hardware)

Full chain median: **4046 µs** (3 GEMMs dominate).

Per-component:

| component         | median µs | notes                           |
|-------------------|----------:|---------------------------------|
| q_proj_mm         |    610.27 | cuBLAS, (512,2048)@(2048,2048)  |
| k_proj_mm         |    776.02 | cuBLAS, (512,2048)@(2048,1024)  |
| v_proj_mm         |   1538.88 | cuBLAS, same shape as k         |
| q_rmsnorm_rope    |     29.81 | inductor's `per_fused..._1`     |
| k_rmsnorm_rope    |     18.40 | inductor's `per_fused..._2`     |
| kv_expand_k       |     23.55 | inductor's `where_3` on K       |
| kv_expand_v       |     23.55 | inductor's `where_3` on V       |
| causal_mask       |      9.18 | inductor's `where_4`            |
| **full chain**    | **4045.73** | sequential do_bench           |

The component sums (3024 µs) don't add up to the full-chain median
(4046 µs) because do_bench's amortised launch overhead is non-additive
across components — the full-chain measurement is what an agent actually
has to beat. Agents likely have the most headroom in:

  - **Skipping the K/V GQA materialisation** (`_where_3` × 2 ≈ 47 µs
    saved) by writing an SDPA kernel that broadcasts KV indices itself.
  - **Fusing RMSNorm and RoPE into the Q-projection epilogue** (saving
    one full read of buf2/buf5 per call ≈ 30 µs each).
  - **Replacing the V mm** with a fused version (this is the slowest
    GEMM in the bench — but cuBLAS is hard to beat).

We expect agents to **stay inside the Triton envelope** and emulate
inductor's structure (3 cuBLAS calls + a few Triton kernels), because
that's the path of least resistance and they already saw it work for
swiglu/rmsnorm. The Flash-Attention-style absorbed prelude is allowed
but agents are unlikely to attempt it without explicit nudging.
