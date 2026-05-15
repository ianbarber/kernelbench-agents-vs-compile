from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


SEQ = 512
HIDDEN = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
GQA_GROUP = 2


@triton.jit
def _q_norm_rope_inplace(
    q_flat,
    w_norm,
    inv_freq,
    position_ids,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # pid = s * 16 + q_head
    offs = tl.arange(0, BLOCK)
    base = pid * BLOCK

    x = tl.load(q_flat + base + offs).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(ss * 0.0078125 + eps)

    other = tl.where(offs < 64, offs + 64, offs - 64)
    other_x = tl.load(q_flat + base + other).to(tl.float32)

    w = tl.load(w_norm + offs).to(tl.float32)
    other_w = tl.load(w_norm + other).to(tl.float32)
    x_norm = x * inv_rms * w
    rot_norm = other_x * inv_rms * other_w
    rot_norm = tl.where(offs < 64, -rot_norm, rot_norm)

    seq = pid // 16
    pos = tl.load(position_ids + seq).to(tl.float32)
    angle = pos * tl.load(inv_freq + (offs & 63)).to(tl.float32)
    out = x_norm * tl.cos(angle) + rot_norm * tl.sin(angle)
    tl.store(q_flat + base + offs, out)


@triton.jit
def _kv_norm_rope_expand(
    k_flat,
    v_flat,
    k_out,
    v_out,
    w_norm,
    inv_freq,
    position_ids,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # pid = s * 8 + kv_head
    offs = tl.arange(0, BLOCK)
    src_base = pid * BLOCK

    x = tl.load(k_flat + src_base + offs).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(ss * 0.0078125 + eps)

    other = tl.where(offs < 64, offs + 64, offs - 64)
    other_x = tl.load(k_flat + src_base + other).to(tl.float32)

    w = tl.load(w_norm + offs).to(tl.float32)
    other_w = tl.load(w_norm + other).to(tl.float32)
    x_norm = x * inv_rms * w
    rot_norm = other_x * inv_rms * other_w
    rot_norm = tl.where(offs < 64, -rot_norm, rot_norm)

    seq = pid // 8
    kv_head = pid - seq * 8
    pos = tl.load(position_ids + seq).to(tl.float32)
    angle = pos * tl.load(inv_freq + (offs & 63)).to(tl.float32)
    k_val = x_norm * tl.cos(angle) + rot_norm * tl.sin(angle)

    q_head0 = kv_head * 2
    dst0 = (q_head0 * 512 + seq) * 128 + offs
    dst1 = dst0 + 512 * 128
    tl.store(k_out + dst0, k_val)
    tl.store(k_out + dst1, k_val)

    v_val = tl.load(v_flat + src_base + offs)
    tl.store(v_out + dst0, v_val)
    tl.store(v_out + dst1, v_val)


@triton.jit
def _mask_kernel(attention_mask, out, BLOCK: tl.constexpr):
    q = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    keep = (k <= q) & (tl.load(attention_mask + k) != 0)
    val = tl.where(keep, 0.0, -float("inf"))
    tl.store(out + q * 512 + k, val)


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
    hidden_flat = hidden_states.reshape(SEQ, HIDDEN)

    w_q_t = torch.as_strided(w_q, (HIDDEN, HIDDEN), (1, HIDDEN))
    w_k_t = torch.as_strided(w_k, (HIDDEN, NUM_KV_HEADS * HEAD_DIM), (1, HIDDEN))
    w_v_t = torch.as_strided(w_v, (HIDDEN, NUM_KV_HEADS * HEAD_DIM), (1, HIDDEN))

    q_flat = torch.empty((SEQ, HIDDEN), device=hidden_states.device, dtype=torch.bfloat16)
    k_flat = torch.empty((SEQ, NUM_KV_HEADS * HEAD_DIM), device=hidden_states.device, dtype=torch.bfloat16)
    v_flat = torch.empty((SEQ, NUM_KV_HEADS * HEAD_DIM), device=hidden_states.device, dtype=torch.bfloat16)

    torch.mm(hidden_flat, w_q_t, out=q_flat)
    torch.mm(hidden_flat, w_k_t, out=k_flat)
    torch.mm(hidden_flat, w_v_t, out=v_flat)

    k = torch.empty((1, NUM_Q_HEADS, SEQ, HEAD_DIM), device=hidden_states.device, dtype=torch.bfloat16)
    v = torch.empty((1, NUM_Q_HEADS, SEQ, HEAD_DIM), device=hidden_states.device, dtype=torch.bfloat16)
    mask = torch.empty((1, 1, SEQ, SEQ), device=hidden_states.device, dtype=torch.bfloat16)

    _q_norm_rope_inplace[(SEQ * NUM_Q_HEADS,)](
        q_flat, w_q_norm, inv_freq, position_ids, eps, BLOCK=HEAD_DIM, num_warps=4
    )
    _kv_norm_rope_expand[(SEQ * NUM_KV_HEADS,)](
        k_flat, v_flat, k, v, w_k_norm, inv_freq, position_ids, eps, BLOCK=HEAD_DIM, num_warps=4
    )
    _mask_kernel[(SEQ,)](attention_mask, mask, BLOCK=SEQ, num_warps=8)

    q = q_flat.view(1, SEQ, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    return q, k, v, mask
