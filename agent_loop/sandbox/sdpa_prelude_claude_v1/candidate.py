"""Fused SDPA prelude: Q/K/V projections + RMSNorm + RoPE + GQA expand + mask.

Approach:
  * QKV projections stay as cuBLAS mm (with the as_strided trick).
  * One Triton kernel for Q (RMSNorm + RoPE, layout SHD -> HSD).
  * One Triton kernel for K (RMSNorm + RoPE + GQA expand 8->16, layout SHD -> HSD).
  * One Triton kernel for V (GQA expand 8->16, layout SHD -> HSD).
  * One Triton kernel for the (S, S) causal/padding mask.

cos/sin tables for RoPE are computed inline inside the Q/K kernels from
position_ids + inv_freq, so we never materialise them.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF = HEAD_DIM // 2  # 64
GQA = NUM_Q_HEADS // NUM_KV_HEADS  # 2


@triton.jit
def _rmsnorm_rope_q_kernel(
    in_ptr, out_ptr, w_norm_ptr, position_ids_ptr, inv_freq_ptr,
    eps,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    HALF_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # in_ptr layout: (B, S, H, D)  -- flattened cuBLAS output reshaped.
    # out_ptr layout: (B, H, S, D)
    # Each program: one batch element b, one head h, BLOCK_S rows of S.
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    h = pid_bh % H
    b = pid_bh // H

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    half_off = tl.arange(0, HALF_C)

    in_base = b * S * H * D + s_off[:, None] * (H * D) + h * D
    x_lo = tl.load(in_ptr + in_base + half_off[None, :],
                   mask=s_mask[:, None], other=0.0).to(tl.float32)
    x_hi = tl.load(in_ptr + in_base + half_off[None, :] + HALF_C,
                   mask=s_mask[:, None], other=0.0).to(tl.float32)

    # RMSNorm over D (=128). All math fp32.
    sum_sq = tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)
    inv = tl.rsqrt(sum_sq / D + eps)  # (BLOCK_S,)

    w_lo = tl.load(w_norm_ptr + half_off).to(tl.float32)
    w_hi = tl.load(w_norm_ptr + half_off + HALF_C).to(tl.float32)
    n_lo = x_lo * inv[:, None] * w_lo[None, :]
    n_hi = x_hi * inv[:, None] * w_hi[None, :]

    # RoPE: build cos/sin from position_ids + inv_freq.
    pos = tl.load(position_ids_ptr + b * S + s_off,
                  mask=s_mask, other=0).to(tl.float32)
    inv_f = tl.load(inv_freq_ptr + half_off)  # fp32
    angle = pos[:, None] * inv_f[None, :]
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    # x * cos + rotate_half(x) * sin, where rotate_half([lo,hi]) = [-hi, lo].
    out_lo = n_lo * cos - n_hi * sin
    out_hi = n_hi * cos + n_lo * sin

    out_base = b * H * S * D + h * (S * D) + s_off[:, None] * D
    tl.store(out_ptr + out_base + half_off[None, :],
             out_lo.to(tl.bfloat16), mask=s_mask[:, None])
    tl.store(out_ptr + out_base + half_off[None, :] + HALF_C,
             out_hi.to(tl.bfloat16), mask=s_mask[:, None])


@triton.jit
def _rmsnorm_rope_k_expand_kernel(
    in_ptr, out_ptr, w_norm_ptr, position_ids_ptr, inv_freq_ptr,
    eps,
    S: tl.constexpr, QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
    HALF_C: tl.constexpr, BLOCK_S: tl.constexpr, GQA_C: tl.constexpr,
):
    # in_ptr layout: (B, S, KH, D)
    # out_ptr layout: (B, QH, S, D) where qh -> kh = qh // GQA
    pid_bkh = tl.program_id(0)
    pid_s = tl.program_id(1)
    kh = pid_bkh % KH
    b = pid_bkh // KH

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    half_off = tl.arange(0, HALF_C)

    in_base = b * S * KH * D + s_off[:, None] * (KH * D) + kh * D
    x_lo = tl.load(in_ptr + in_base + half_off[None, :],
                   mask=s_mask[:, None], other=0.0).to(tl.float32)
    x_hi = tl.load(in_ptr + in_base + half_off[None, :] + HALF_C,
                   mask=s_mask[:, None], other=0.0).to(tl.float32)

    sum_sq = tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)
    inv = tl.rsqrt(sum_sq / D + eps)

    w_lo = tl.load(w_norm_ptr + half_off).to(tl.float32)
    w_hi = tl.load(w_norm_ptr + half_off + HALF_C).to(tl.float32)
    n_lo = x_lo * inv[:, None] * w_lo[None, :]
    n_hi = x_hi * inv[:, None] * w_hi[None, :]

    pos = tl.load(position_ids_ptr + b * S + s_off,
                  mask=s_mask, other=0).to(tl.float32)
    inv_f = tl.load(inv_freq_ptr + half_off)
    angle = pos[:, None] * inv_f[None, :]
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    out_lo = (n_lo * cos - n_hi * sin).to(tl.bfloat16)
    out_hi = (n_hi * cos + n_lo * sin).to(tl.bfloat16)

    # Write to GQA_C consecutive q-head positions: qh = kh*GQA + i for i in [0, GQA).
    for i in tl.static_range(GQA_C):
        qh = kh * GQA_C + i
        out_base = b * QH * S * D + qh * (S * D) + s_off[:, None] * D
        tl.store(out_ptr + out_base + half_off[None, :],
                 out_lo, mask=s_mask[:, None])
        tl.store(out_ptr + out_base + half_off[None, :] + HALF_C,
                 out_hi, mask=s_mask[:, None])


@triton.jit
def _v_expand_kernel(
    in_ptr, out_ptr,
    S: tl.constexpr, QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, GQA_C: tl.constexpr,
):
    # in_ptr layout: (B, S, KH, D).  out: (B, QH, S, D), GQA-expanded.
    pid_bkh = tl.program_id(0)
    pid_s = tl.program_id(1)
    kh = pid_bkh % KH
    b = pid_bkh // KH

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    d_off = tl.arange(0, D)

    in_base = b * S * KH * D + s_off[:, None] * (KH * D) + kh * D
    x = tl.load(in_ptr + in_base + d_off[None, :],
                mask=s_mask[:, None], other=0)

    for i in tl.static_range(GQA_C):
        qh = kh * GQA_C + i
        out_base = b * QH * S * D + qh * (S * D) + s_off[:, None] * D
        tl.store(out_ptr + out_base + d_off[None, :], x, mask=s_mask[:, None])


@triton.jit
def _causal_mask_kernel(
    attn_mask_ptr, out_ptr,
    S: tl.constexpr, BLOCK: tl.constexpr,
):
    # out shape (B, 1, S, S) bf16. attn_mask shape (B, S) int64.
    pid = tl.program_id(0)
    q = pid % S
    b = pid // S

    k_off = tl.arange(0, BLOCK)
    k_mask = k_off < S

    causal = k_off <= q
    am = tl.load(attn_mask_ptr + b * S + k_off, mask=k_mask, other=0)
    keep = (am != 0) & causal & k_mask

    NEG_INF: tl.constexpr = float("-inf")
    val = tl.where(keep, 0.0, NEG_INF)

    base = b * S * S + q * S
    tl.store(out_ptr + base + k_off, val.to(tl.bfloat16), mask=k_mask)


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
    B, S, H = hidden_states.shape
    QH = NUM_Q_HEADS
    KH = NUM_KV_HEADS
    D = HEAD_DIM
    device = hidden_states.device
    dtype = torch.bfloat16

    # --- Q/K/V projections (cuBLAS, with as_strided fast-path trick). ---
    hidden_flat = hidden_states.reshape(B * S, H)
    w_q_T = torch.as_strided(w_q, (H, QH * D), (1, H))
    w_k_T = torch.as_strided(w_k, (H, KH * D), (1, H))
    w_v_T = torch.as_strided(w_v, (H, KH * D), (1, H))
    q_flat = torch.mm(hidden_flat, w_q_T)  # (B*S, QH*D)
    k_flat = torch.mm(hidden_flat, w_k_T)  # (B*S, KH*D)
    v_flat = torch.mm(hidden_flat, w_v_T)  # (B*S, KH*D)

    # --- Allocate outputs. ---
    q_out = torch.empty((B, QH, S, D), dtype=dtype, device=device)
    k_out = torch.empty((B, QH, S, D), dtype=dtype, device=device)
    v_out = torch.empty((B, QH, S, D), dtype=dtype, device=device)
    mask_out = torch.empty((B, 1, S, S), dtype=dtype, device=device)

    BLOCK_S = 16  # rows per program
    grid_q = (B * QH, triton.cdiv(S, BLOCK_S))
    grid_k = (B * KH, triton.cdiv(S, BLOCK_S))
    grid_v = (B * KH, triton.cdiv(S, BLOCK_S))

    _rmsnorm_rope_q_kernel[grid_q](
        q_flat, q_out, w_q_norm, position_ids, inv_freq,
        eps,
        S, QH, D, HALF, BLOCK_S,
        num_warps=4,
    )
    _rmsnorm_rope_k_expand_kernel[grid_k](
        k_flat, k_out, w_k_norm, position_ids, inv_freq,
        eps,
        S, QH, KH, D, HALF, BLOCK_S, GQA,
        num_warps=4,
    )
    _v_expand_kernel[grid_v](
        v_flat, v_out,
        S, QH, KH, D, BLOCK_S, GQA,
        num_warps=4,
    )

    # Mask
    MASK_BLOCK = triton.next_power_of_2(S)  # 512
    _causal_mask_kernel[(B * S,)](
        attention_mask, mask_out,
        S, MASK_BLOCK,
        num_warps=4,
    )

    return q_out, k_out, v_out, mask_out
