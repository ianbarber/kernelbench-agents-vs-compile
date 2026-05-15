"""Fused SDPA-prelude: Q/K/V projections + per-head RMSNorm + RoPE + GQA-expand + mask.

Strategy:
  1. Single cuBLAS GEMM for Q+K+V projections (weights stacked, output split).
  2. Triton kernel fuses Q RMSNorm + RoPE directly on the GEMM output.
  3. Triton kernel fuses K RMSNorm + RoPE + GQA-expand (8 -> 16).
  4. Triton kernel fuses V GQA-expand (8 -> 16).
  5. Triton kernel builds the causal + padding mask.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128


@triton.jit
def q_rmsnorm_rope_kernel(
    q_in_ptr,
    q_out_ptr,
    w_q_norm_ptr,
    cos_ptr,
    sin_ptr,
    B,
    S,
    NUM_Q_HEADS,
    D,
    q_in_stride_s,
    q_in_stride_h,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_s,
    cos_stride_s,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_per_batch = S * NUM_Q_HEADS
    b = pid // num_blocks_per_batch
    rem = pid % num_blocks_per_batch
    s = rem // NUM_Q_HEADS
    h = rem % NUM_Q_HEADS

    idx1 = tl.arange(0, BLOCK_D // 2)
    idx2 = tl.arange(BLOCK_D // 2, BLOCK_D)
    in_offset = (b * S + s) * q_in_stride_s + h * q_in_stride_h

    q1 = tl.load(q_in_ptr + in_offset + idx1).to(tl.float32)
    q2 = tl.load(q_in_ptr + in_offset + idx2).to(tl.float32)

    # RMSNorm in fp32
    var = (tl.sum(q1 * q1, axis=0) + tl.sum(q2 * q2, axis=0)) / D
    inv = tl.rsqrt(var + eps)
    w1 = tl.load(w_q_norm_ptr + idx1).to(tl.float32)
    w2 = tl.load(w_q_norm_ptr + idx2).to(tl.float32)
    q1 = q1 * inv * w1
    q2 = q2 * inv * w2

    # RoPE
    cos_offset = (b * S + s) * cos_stride_s
    sin_offset = cos_offset
    cos1 = tl.load(cos_ptr + cos_offset + idx1).to(tl.float32)
    cos2 = tl.load(cos_ptr + cos_offset + idx2).to(tl.float32)
    sin1 = tl.load(sin_ptr + sin_offset + idx1).to(tl.float32)
    sin2 = tl.load(sin_ptr + sin_offset + idx2).to(tl.float32)

    out1 = q1 * cos1 - q2 * sin1
    out2 = q2 * cos2 + q1 * sin2

    out_offset = b * q_out_stride_b + h * q_out_stride_h + s * q_out_stride_s
    tl.store(q_out_ptr + out_offset + idx1, out1.to(tl.bfloat16))
    tl.store(q_out_ptr + out_offset + idx2, out2.to(tl.bfloat16))


@triton.jit
def k_rmsnorm_rope_expand_kernel(
    k_in_ptr,
    k_out_ptr,
    w_k_norm_ptr,
    cos_ptr,
    sin_ptr,
    B,
    S,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    D,
    k_in_stride_s,
    k_in_stride_h,
    k_out_stride_b,
    k_out_stride_h,
    k_out_stride_s,
    cos_stride_s,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_per_batch = S * NUM_Q_HEADS
    b = pid // num_blocks_per_batch
    rem = pid % num_blocks_per_batch
    s = rem // NUM_Q_HEADS
    h_out = rem % NUM_Q_HEADS
    h_in = h_out // 2

    idx1 = tl.arange(0, BLOCK_D // 2)
    idx2 = tl.arange(BLOCK_D // 2, BLOCK_D)
    in_offset = (b * S + s) * k_in_stride_s + h_in * k_in_stride_h

    k1 = tl.load(k_in_ptr + in_offset + idx1).to(tl.float32)
    k2 = tl.load(k_in_ptr + in_offset + idx2).to(tl.float32)

    # RMSNorm in fp32
    var = (tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)) / D
    inv = tl.rsqrt(var + eps)
    w1 = tl.load(w_k_norm_ptr + idx1).to(tl.float32)
    w2 = tl.load(w_k_norm_ptr + idx2).to(tl.float32)
    k1 = k1 * inv * w1
    k2 = k2 * inv * w2

    # RoPE
    cos_offset = (b * S + s) * cos_stride_s
    sin_offset = cos_offset
    cos1 = tl.load(cos_ptr + cos_offset + idx1).to(tl.float32)
    cos2 = tl.load(cos_ptr + cos_offset + idx2).to(tl.float32)
    sin1 = tl.load(sin_ptr + sin_offset + idx1).to(tl.float32)
    sin2 = tl.load(sin_ptr + sin_offset + idx2).to(tl.float32)

    out1 = k1 * cos1 - k2 * sin1
    out2 = k2 * cos2 + k1 * sin2

    out_offset = b * k_out_stride_b + h_out * k_out_stride_h + s * k_out_stride_s
    tl.store(k_out_ptr + out_offset + idx1, out1.to(tl.bfloat16))
    tl.store(k_out_ptr + out_offset + idx2, out2.to(tl.bfloat16))


@triton.jit
def v_expand_kernel(
    v_in_ptr,
    v_out_ptr,
    B,
    S,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    D,
    v_in_stride_s,
    v_in_stride_h,
    v_out_stride_b,
    v_out_stride_h,
    v_out_stride_s,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_per_batch = S * NUM_Q_HEADS
    b = pid // num_blocks_per_batch
    rem = pid % num_blocks_per_batch
    s = rem // NUM_Q_HEADS
    h_out = rem % NUM_Q_HEADS
    h_in = h_out // 2

    idx = tl.arange(0, BLOCK_D)
    in_offset = (b * S + s) * v_in_stride_s + h_in * v_in_stride_h
    v_vals = tl.load(v_in_ptr + in_offset + idx)

    out_offset = b * v_out_stride_b + h_out * v_out_stride_h + s * v_out_stride_s
    tl.store(v_out_ptr + out_offset + idx, v_vals)


@triton.jit
def causal_mask_kernel(
    attention_mask_ptr,
    mask_out_ptr,
    B,
    S,
    mask_out_stride_b,
    mask_out_stride_q,
    mask_out_stride_k,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    q = pid % S

    idx = tl.arange(0, BLOCK_S)
    am = tl.load(attention_mask_ptr + b * S + idx)

    k_vals = idx
    causal = k_vals <= q
    keep = causal & (am != 0)

    mask_vals = tl.where(
        keep,
        tl.full((BLOCK_S,), 0.0, tl.float32),
        tl.full((BLOCK_S,), -float("inf"), tl.float32),
    )

    out_offset = b * mask_out_stride_b + q * mask_out_stride_q
    tl.store(mask_out_ptr + out_offset + idx, mask_vals.to(tl.bfloat16))


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
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, hidden = hidden_states.shape
    device = hidden_states.device

    # --- Single stacked QKV projection GEMM ---
    hidden_flat = hidden_states.reshape(B * S, hidden)
    w_qkv = torch.cat([w_q, w_k, w_v], dim=0)  # (4096, 2048)
    w_qkv_T = torch.as_strided(w_qkv, (hidden, NUM_Q_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM), (1, hidden))
    qkv_flat = torch.mm(hidden_flat, w_qkv_T)  # (B*S, 4096)

    q_flat = qkv_flat[:, : NUM_Q_HEADS * HEAD_DIM]
    k_flat = qkv_flat[:, NUM_Q_HEADS * HEAD_DIM : NUM_Q_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM]
    v_flat = qkv_flat[:, NUM_Q_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM :]

    # --- RoPE tables (B, S, D) -> flattened (B*S, D) ---
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_flat = emb.cos().reshape(B * S, HEAD_DIM)
    sin_flat = emb.sin().reshape(B * S, HEAD_DIM)

    # --- Allocate outputs ---
    q = torch.empty((B, NUM_Q_HEADS, S, HEAD_DIM), dtype=torch.bfloat16, device=device)
    k = torch.empty((B, NUM_Q_HEADS, S, HEAD_DIM), dtype=torch.bfloat16, device=device)
    v = torch.empty((B, NUM_Q_HEADS, S, HEAD_DIM), dtype=torch.bfloat16, device=device)
    mask = torch.empty((B, 1, S, S), dtype=torch.bfloat16, device=device)

    # --- Launch fused Triton kernels ---
    grid_q = B * S * NUM_Q_HEADS
    grid_k = B * S * NUM_Q_HEADS
    grid_v = B * S * NUM_Q_HEADS
    grid_mask = B * S

    q_rmsnorm_rope_kernel[(grid_q,)](
        q_flat,
        q,
        w_q_norm,
        cos_flat,
        sin_flat,
        B,
        S,
        NUM_Q_HEADS,
        HEAD_DIM,
        q_flat.stride(0),
        HEAD_DIM,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cos_flat.stride(0),
        eps,
        BLOCK_D=HEAD_DIM,
    )

    k_rmsnorm_rope_expand_kernel[(grid_k,)](
        k_flat,
        k,
        w_k_norm,
        cos_flat,
        sin_flat,
        B,
        S,
        NUM_KV_HEADS,
        NUM_Q_HEADS,
        HEAD_DIM,
        k_flat.stride(0),
        HEAD_DIM,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos_flat.stride(0),
        eps,
        BLOCK_D=HEAD_DIM,
    )

    v_expand_kernel[(grid_v,)](
        v_flat,
        v,
        B,
        S,
        NUM_KV_HEADS,
        NUM_Q_HEADS,
        HEAD_DIM,
        v_flat.stride(0),
        HEAD_DIM,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        BLOCK_D=HEAD_DIM,
    )

    causal_mask_kernel[(grid_mask,)](
        attention_mask,
        mask,
        B,
        S,
        mask.stride(0),
        mask.stride(2),
        mask.stride(3),
        BLOCK_S=S,
    )

    return q, k, v, mask
