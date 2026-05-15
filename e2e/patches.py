"""Monkey-patches for installing agent-produced kernels into HF Qwen3.

This module rewires `Qwen3MLP.forward` and `Qwen3RMSNorm.forward` (and
optionally `Qwen3DecoderLayer.forward`) to call our extracted Triton kernels
instead of the eager PyTorch implementations.

The patches are *class-level* (we replace `Qwen3MLP.forward`, etc.), and we
keep idempotent install/uninstall so we can flip configurations within a
single Python process.

Pure vs fused RMSNorm:
  - install_rmsnorm_claude_pure: drop-in replacement for `Qwen3RMSNorm.forward`
    that calls claude's kernel with `residual = zeros_like(x)`. This wastes
    the fusion benefit but tests whether agent-RMSNorm at non-fused use is
    competitive.
  - install_rmsnorm_claude_fused: refactors `Qwen3DecoderLayer.forward` so
    that the residual-add + post-norm pair is performed in a single kernel
    call. Implemented for both the input_layernorm (pre-attn) and the
    post_attention_layernorm (pre-mlp) pairs.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3 import modeling_qwen3 as M

from .kernels.swiglu_kimi import run as swiglu_run
from .kernels.rmsnorm_claude import run as rmsnorm_run
from .kernels.sdpa_prelude_kimi import run_no_mask as sdpa_prelude_run_no_mask


# Marker attribute on the class so we know whether a patch has been installed.
_SWIGLU_MARK = "_kbench_swiglu_patched"
_RMSNORM_PURE_MARK = "_kbench_rmsnorm_pure_patched"
_RMSNORM_FUSED_MARK = "_kbench_rmsnorm_fused_patched"
_SDPA_PRELUDE_MARK = "_kbench_sdpa_prelude_patched"

# Cached originals (set at first install).
_ORIG_MLP_FORWARD = M.Qwen3MLP.forward
_ORIG_RMSNORM_FORWARD = M.Qwen3RMSNorm.forward
_ORIG_DECODER_FORWARD = M.Qwen3DecoderLayer.forward
_ORIG_ATTN_FORWARD = M.Qwen3Attention.forward


# ---------------- SwiGLU ----------------


def _mlp_forward_kimi(self, x):
    """Replacement for Qwen3MLP.forward using kimi's swiglu kernel.

    The kernel requires bf16 contiguous inputs of equal shape and only
    handles n_elements % BLOCK_SIZE == 0 (BLOCK_SIZE=256). For Qwen3-1.7B
    intermediate_size=6144, so any batch*seq*6144 satisfies that.
    """
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    # Triton kernel needs contiguous bf16 same-shape tensors.
    if not gate.is_contiguous():
        gate = gate.contiguous()
    if not up.is_contiguous():
        up = up.contiguous()
    n = gate.numel()
    # Fall back to eager if shape isn't divisible (shouldn't happen for Qwen3
    # at standard shapes, but be safe — better correct than crash).
    if n % 256 != 0 or gate.dtype != torch.bfloat16:
        return self.down_proj(self.act_fn(gate) * up)
    fused = swiglu_run(gate, up)
    return self.down_proj(fused)


def install_swiglu_kimi(model: Any) -> None:
    cls = M.Qwen3MLP
    if getattr(cls, _SWIGLU_MARK, False):
        return
    cls.forward = _mlp_forward_kimi
    setattr(cls, _SWIGLU_MARK, True)


def uninstall_swiglu(model: Any) -> None:
    cls = M.Qwen3MLP
    if not getattr(cls, _SWIGLU_MARK, False):
        return
    cls.forward = _ORIG_MLP_FORWARD
    setattr(cls, _SWIGLU_MARK, False)


# ---------------- RMSNorm (pure) ----------------

# A cache of zero-residual tensors keyed by (shape, device, dtype) to avoid
# reallocating inside forward. This is critical for decode-step performance.
_ZERO_CACHE: dict = {}


def _get_zero_like(x: torch.Tensor) -> torch.Tensor:
    key = (tuple(x.shape), x.device, x.dtype)
    z = _ZERO_CACHE.get(key)
    if z is None or z.shape != x.shape:
        z = torch.zeros_like(x)
        _ZERO_CACHE[key] = z
    return z


def _rmsnorm_forward_claude_pure(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Replacement for Qwen3RMSNorm.forward using claude's kernel + zero residual."""
    x = hidden_states
    # Kernel assumes bf16 + cuda + last dim is the norm dim.
    if x.dtype != torch.bfloat16 or not x.is_cuda or self.weight.dtype != torch.bfloat16:
        # Fall back to eager.
        return _ORIG_RMSNORM_FORWARD(self, hidden_states)
    if not x.is_contiguous():
        x = x.contiguous()
    zero = _get_zero_like(x)
    return rmsnorm_run(x, zero, self.weight, self.variance_epsilon)


def install_rmsnorm_claude_pure(model: Any) -> None:
    cls = M.Qwen3RMSNorm
    if getattr(cls, _RMSNORM_PURE_MARK, False):
        return
    cls.forward = _rmsnorm_forward_claude_pure
    setattr(cls, _RMSNORM_PURE_MARK, True)


def uninstall_rmsnorm_pure(model: Any) -> None:
    cls = M.Qwen3RMSNorm
    if not getattr(cls, _RMSNORM_PURE_MARK, False):
        return
    cls.forward = _ORIG_RMSNORM_FORWARD
    setattr(cls, _RMSNORM_PURE_MARK, False)
    _ZERO_CACHE.clear()


# ---------------- RMSNorm (fused decoder-layer refactor) ----------------


def _decoder_forward_fused(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache: bool = False,
    position_embeddings=None,
    **kwargs,
):
    """Refactor of Qwen3DecoderLayer.forward that fuses each (residual_add + RMSNorm).

    Original (eager) pattern:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states, _ = self_attn(...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states

    Our refactor fuses the "residual + norm" pair on the second pass — and
    on the first pass too, with the previous block's residual=0 absorbed
    into the input_layernorm call. To keep it clean and HF-compatible we
    only fuse where there is a real residual to add:

      Layer entry — no prior residual to fuse, just call our kernel with
      residual=zeros (degrades to pure-norm). This matches install_pure
      behaviour for the first norm.

      Between attn output and post_attention_layernorm — we have a real
      residual. We call the kernel with (attn_out, residual) and produce
      the normalized tensor directly. The post-MLP residual-add is a plain
      add.

    Net: one of two norms per layer becomes a fused residual+norm. Compared
    to pure (where both norms have residual=0 wasted), we save a memory
    pass per layer for the post_attention_layernorm.
    """
    # --- pre-attn norm: just call rmsnorm_run with zero residual ---
    if hidden_states.dtype == torch.bfloat16 and hidden_states.is_cuda:
        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        zero = _get_zero_like(x)
        normed = rmsnorm_run(
            x,
            zero,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )
    else:
        normed = self.input_layernorm(hidden_states)

    residual = hidden_states
    # Self Attention
    attn_out, _ = self.self_attn(
        hidden_states=normed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    # Fused residual + post_attention_layernorm:
    #   pre_mlp_residual = residual + attn_out   (this is the value to add back later)
    #   pre_mlp_norm     = rmsnorm(pre_mlp_residual) * post_attention_layernorm.weight
    # Our kernel computes rmsnorm(x + residual) * weight in one shot. We pass
    # x=attn_out, residual=residual. We also need pre_mlp_residual separately
    # for the trailing add — kernel doesn't materialize it, so we compute it
    # with a plain add (still saves one kernel pass for the norm phase).
    if attn_out.dtype == torch.bfloat16 and attn_out.is_cuda:
        a = attn_out if attn_out.is_contiguous() else attn_out.contiguous()
        r = residual if residual.is_contiguous() else residual.contiguous()
        pre_mlp_norm = rmsnorm_run(
            a,
            r,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        # We still need the un-normed sum for the trailing residual.
        pre_mlp_residual = a + r
    else:
        pre_mlp_residual = residual + attn_out
        pre_mlp_norm = self.post_attention_layernorm(pre_mlp_residual)

    hidden_states = self.mlp(pre_mlp_norm)
    hidden_states = pre_mlp_residual + hidden_states
    return hidden_states


def install_rmsnorm_claude_fused(model: Any) -> None:
    cls = M.Qwen3DecoderLayer
    if getattr(cls, _RMSNORM_FUSED_MARK, False):
        return
    cls.forward = _decoder_forward_fused
    setattr(cls, _RMSNORM_FUSED_MARK, True)


def uninstall_rmsnorm_fused(model: Any) -> None:
    cls = M.Qwen3DecoderLayer
    if not getattr(cls, _RMSNORM_FUSED_MARK, False):
        return
    cls.forward = _ORIG_DECODER_FORWARD
    setattr(cls, _RMSNORM_FUSED_MARK, False)
    _ZERO_CACHE.clear()


# ---------------- SDPA prelude (kimi) ----------------
#
# Fuses Q/K/V projections (stacked GEMM) + per-head Q/K RMSNorm + RoPE +
# GQA-expand of K and V into 4 Triton kernels. We REPLACE everything in
# Qwen3Attention.forward up through `repeat_kv` (kimi's K/V kernels do the
# GQA expand themselves, so the SDPA we call here is plain MHA shapes).
#
# Scope:
#   - Prefill only (sequence length > 1 and past_key_values is None).
#   - For decode (S == 1) or cached prefill (use_cache=True with past KV),
#     fall back to the original eager forward. The kimi kernel fuses the
#     GQA expand which is incompatible with HF's cache.update() — the cache
#     stores PRE-expansion KV (8 heads), and our kernel only emits the
#     post-expansion (16-head) tensors. Untangling would require splitting
#     the K kernel into two passes; for the headline prefill speedup this
#     is unnecessary.


def _attn_forward_sdpa_prelude(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """Replacement for Qwen3Attention.forward.

    For prefill with no cache, route through kimi's fused SDPA-prelude
    kernel. Otherwise fall back to the original HF forward.
    """
    B, S, hidden = hidden_states.shape
    use_kimi = (
        S > 1
        and past_key_values is None
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        # Kimi's kernels were written for Qwen3-1.7B head_dim=128.
        and self.head_dim == 128
        and self.config.num_attention_heads == 16
        and self.config.num_key_value_heads == 8
        # Mask must be a real additive 4D mask (B, 1, S, S) bf16/fp32.
        # If HF skipped mask creation (None), eager SDPA still works — but
        # we'd need to build one. Skip fast path in that case.
        and attention_mask is not None
        and attention_mask.dim() == 4
    )

    if not use_kimi:
        return _ORIG_ATTN_FORWARD(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    cos, sin = position_embeddings  # (B, S, head_dim) bf16
    # Kimi's Triton kernels index cos/sin as flat (B*S, head_dim).
    head_dim = self.head_dim
    cos_flat = cos.reshape(B * S, head_dim).contiguous()
    sin_flat = sin.reshape(B * S, head_dim).contiguous()

    hidden = hidden_states.contiguous() if not hidden_states.is_contiguous() else hidden_states

    q, k_expanded, v_expanded = sdpa_prelude_run_no_mask(
        hidden,
        self.q_proj.weight,
        self.k_proj.weight,
        self.v_proj.weight,
        self.q_norm.weight,
        self.k_norm.weight,
        cos_flat,
        sin_flat,
        self.q_norm.variance_epsilon,
    )

    # Inline eager SDPA (we already did the GQA expand, so we MUST NOT call
    # eager_attention_forward which would repeat_kv on top of our 16-head
    # tensors). Use F.scaled_dot_product_attention with the prebuilt 4D mask;
    # this routes to the cuBLAS/efficient SDPA backend.
    # Cast attention_mask to bf16 if necessary so shapes/dtypes match.
    attn_mask = attention_mask
    if attn_mask.dtype != q.dtype:
        attn_mask = attn_mask.to(q.dtype)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k_expanded,
        v_expanded,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,  # mask is already causal — don't double-apply
        scale=self.scaling,
    )
    # (B, H, S, D) -> (B, S, H*D)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_sdpa_prelude_kimi(model: Any) -> None:
    cls = M.Qwen3Attention
    if getattr(cls, _SDPA_PRELUDE_MARK, False):
        return
    cls.forward = _attn_forward_sdpa_prelude
    setattr(cls, _SDPA_PRELUDE_MARK, True)


def uninstall_sdpa_prelude(model: Any) -> None:
    cls = M.Qwen3Attention
    if not getattr(cls, _SDPA_PRELUDE_MARK, False):
        return
    cls.forward = _ORIG_ATTN_FORWARD
    setattr(cls, _SDPA_PRELUDE_MARK, False)


# ---------------- combined uninstall ----------------


def uninstall(model: Any) -> None:
    """Revert every patch installed by this module."""
    uninstall_swiglu(model)
    uninstall_rmsnorm_pure(model)
    uninstall_rmsnorm_fused(model)
    uninstall_sdpa_prelude(model)
