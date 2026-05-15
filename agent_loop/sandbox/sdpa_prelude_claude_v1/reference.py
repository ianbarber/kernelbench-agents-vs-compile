"""Eager PyTorch reference for the SDPA-prelude fused op (Qwen3-1.7B layout).

This matches inductor's prelude for `aten._scaled_dot_product_efficient_attention`:
it takes the post-residual-RMSNorm `hidden_states` and produces the four tensors
that SDPA consumes:

    q     : (1, 16, S, 128)  bf16   query, RoPE-applied, per-head RMSNormed
    k     : (1, 16, S, 128)  bf16   key, same as Q, GQA-expanded from 8 KV heads
    v     : (1, 16, S, 128)  bf16   value, GQA-expanded
    mask  : (1, 1, S, S)     bf16   additive causal+padding mask (0 / -inf)

The corresponding inductor kernels (prefill_512_b1) are:
  * `extern_kernels.mm`              -- Q/K/V projection GEMMs (cuBLAS)
  * `triton_per_fused..._1`          -- Q RoPE + per-head RMSNorm
  * `triton_per_fused..._2`          -- K RoPE + per-head RMSNorm
  * `triton_poi_fused..._where_3`    -- GQA expansion (KV: 8 heads -> 16) ×2
                                         (the kernel that dominates at 24.99%)
  * `triton_poi_fused..._where_4`    -- causal-mask construction (9.24%)
                                         (writes to two mask buffers; we expose
                                          one — both are identical)

The total prelude (excluding SDPA itself) sums to ~5.4% + 24.99% + 9.24% =
roughly 40% of prefill_512_b1 if we count both `_where_3` invocations and
both per_fused RoPE kernels — and that's why this task exists.

Contract notes:
  - The inductor flow keeps the Q-projection mm output (buf2) as a flat
    (512, 2048) bf16 buffer, then reshapes to (1, 16, 512, 128) via
    a per-head-RMSNorm-and-RoPE Triton kernel. We do the same.
  - The K/V projections are (512, 1024) bf16; K goes through its own
    per-head RMSNorm+RoPE; V is just reshaped. Both get GQA-expanded
    8 -> 16 by replicating along the head axis (each KV head paired
    with 2 Q heads).
  - The causal mask is `0` where (q_pos >= k_pos) AND (attention_mask
    at k_pos != 0), `-inf` elsewhere. Inductor stores it as bf16
    (matches what SDPA expects).
  - Inputs are read-only; we MUST NOT mutate them.
  - All reductions/norms done in fp32, cast back to bf16 on the store.
"""
from __future__ import annotations

from typing import Tuple

import torch

EPS_DEFAULT = 1e-6
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 2


def _rms_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-(head, position) RMSNorm over the head_dim=128 axis.

    Args:
        x: bf16, shape (..., 128).
        weight: bf16, shape (128,).
    Returns:
        bf16 tensor of x's shape.
    """
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    return (x_f32 * inv * weight.to(torch.float32)).to(x.dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE: x * cos + rotate_half(x) * sin.

    Args:
        x: (B, H, S, D) bf16.
        cos, sin: (1, 1, S, D) bf16 broadcastable over (B, H).
    Returns:
        bf16 tensor of x's shape.
    """
    out_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    cos_f32 = cos.to(torch.float32)
    sin_f32 = sin.to(torch.float32)
    half = x_f32.shape[-1] // 2
    x1 = x_f32[..., :half]
    x2 = x_f32[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x_f32 * cos_f32 + rotated * sin_f32).to(out_dtype)


def _build_rope_tables(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos / sin tables of shape (1, 1, S, head_dim) bf16.

    Args:
        position_ids: (B, S) int64.
        inv_freq: (head_dim // 2,) fp32.
    """
    # inv_freq: (D/2,) -> (1, D/2)
    # position_ids: (B, S) -> (B, S, 1)
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    # (B, S, D/2) -> concat to (B, S, D)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    # Reshape to (B, 1, S, D) so it broadcasts across heads.
    cos = cos.unsqueeze(1).to(torch.bfloat16)
    sin = sin.unsqueeze(1).to(torch.bfloat16)
    return cos, sin


def _build_causal_mask(
    attention_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Build the (1, 1, S, S) additive bf16 mask used by SDPA.

    `mask[b, 0, q, k] = 0` if (q >= k) and attention_mask[b, k] != 0
                       else -inf.

    Inductor's `_where_4` builds exactly this and writes it to bf16 storage.
    """
    device = attention_mask.device
    B = attention_mask.shape[0]
    q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    k_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
    causal = (k_idx <= q_idx)  # (1, 1, S, S) bool
    keep_k = (attention_mask != 0).view(B, 1, 1, seq_len)  # (B, 1, 1, S) bool
    keep = causal & keep_k  # (B, 1, S, S)
    out = torch.where(
        keep,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), float("-inf"), device=device, dtype=torch.float32),
    )
    return out.to(torch.bfloat16)


def run(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_q_norm: torch.Tensor,
    w_k_norm: torch.Tensor,
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = EPS_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce Q, K, V, mask tensors ready for SDPA.

    Args:
        hidden_states: (B, S, hidden=2048) bf16 — post-residual-RMSNorm.
        w_q: (2048, 2048) bf16  — Q projection weight (in the F.linear sense:
            out = hidden @ w_q.T).
        w_k: (1024, 2048) bf16  — K projection weight.
        w_v: (1024, 2048) bf16  — V projection weight.
        w_q_norm: (128,) bf16   — per-head Q RMSNorm scale.
        w_k_norm: (128,) bf16   — per-head K RMSNorm scale.
        inv_freq: (64,) fp32    — RoPE inverse frequencies (half head_dim).
        position_ids: (B, S) int64.
        attention_mask: (B, S) int64 — 1=keep, 0=pad.
        eps: RMSNorm epsilon.

    Returns:
        (q, k, v, mask):
            q     : (B, 16, S, 128) bf16
            k     : (B, 16, S, 128) bf16   (GQA-expanded from 8 KV heads)
            v     : (B, 16, S, 128) bf16   (GQA-expanded)
            mask  : (B,  1, S, S)   bf16   additive (0 / -inf)
    """
    B, S, hidden = hidden_states.shape
    assert hidden == NUM_Q_HEADS * HEAD_DIM, hidden  # 2048 = 16 * 128
    assert w_q.shape == (NUM_Q_HEADS * HEAD_DIM, hidden), w_q.shape
    assert w_k.shape == (NUM_KV_HEADS * HEAD_DIM, hidden), w_k.shape
    assert w_v.shape == (NUM_KV_HEADS * HEAD_DIM, hidden), w_v.shape

    # --- QKV projections (cuBLAS; matches extern_kernels.mm in inductor). ---
    # `linear` is hidden @ w.T. We mimic inductor's `reinterpret_tensor` trick
    # (stride-swap to (in_features, out_features)) which routes to the fast
    # cuBLAS algo. A naive `hidden @ w.T` on Blackwell bf16 hits a 4-5x slower
    # path -- using as_strided keeps the reference latency honest.
    hidden_flat = hidden_states.reshape(B * S, hidden)
    w_q_T = torch.as_strided(w_q, (hidden, NUM_Q_HEADS * HEAD_DIM), (1, hidden))
    w_k_T = torch.as_strided(w_k, (hidden, NUM_KV_HEADS * HEAD_DIM), (1, hidden))
    w_v_T = torch.as_strided(w_v, (hidden, NUM_KV_HEADS * HEAD_DIM), (1, hidden))
    q_flat = torch.mm(hidden_flat, w_q_T)  # (B*S, 2048)
    k_flat = torch.mm(hidden_flat, w_k_T)  # (B*S, 1024)
    v_flat = torch.mm(hidden_flat, w_v_T)  # (B*S, 1024)

    # --- Reshape to per-head form: (B, S, n_heads, head_dim) then transpose
    # to (B, n_heads, S, head_dim).
    q = q_flat.view(B, S, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    k = k_flat.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    v = v_flat.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2).contiguous()

    # --- Per-head RMSNorm on Q and K (NOT on V). ---
    q = _rms_per_head(q, w_q_norm, eps)
    k = _rms_per_head(k, w_k_norm, eps)

    # --- RoPE tables and apply to Q and K. ---
    cos, sin = _build_rope_tables(position_ids, inv_freq)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    # --- GQA expansion: replicate KV heads to match Q head count. ---
    # k, v are (B, 8, S, 128) -> expand to (B, 16, S, 128).
    # The inductor `_where_3` does this with `x2 // 2` indexing, i.e.
    # each KV head paired with `GQA_GROUP=2` consecutive Q heads.
    k = k.repeat_interleave(GQA_GROUP, dim=1).contiguous()
    v = v.repeat_interleave(GQA_GROUP, dim=1).contiguous()

    # --- Causal + padding mask. ---
    mask = _build_causal_mask(attention_mask, S)

    return q, k, v, mask
