"""Qwen3-1.7B loader and prefill/decode harnesses.

Uses the HuggingFace `DynamicCache` API for KV caching. See
`transformers/cache_utils.py::DynamicCache` and
`transformers/models/qwen3/modeling_qwen3.py::Qwen3ForCausalLM.forward`.

Decode step is exercised by passing the cache back in along with a single new
token; the model uses `past_key_values.get_seq_length()` to derive
`position_ids`, so the per-step forward only does compute on the new token.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

MODEL_ID = "Qwen/Qwen3-1.7B"


def load_model(dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    """Load Qwen3-1.7B and tokenizer, put on `device` in `dtype`, eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def prefill_fn(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single forward pass; no KV cache reuse, no sampling. Returns logits."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return out.logits


@torch.no_grad()
def decode_fn(
    model,
    past_key_values,
    last_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """One decode step using `past_key_values`. Returns (next_logits, updated_kv).

    `last_token_ids` must be shape (batch, 1). `attention_mask` should be the
    full mask covering past_seen_tokens + 1 positions (HF requires the full
    mask, not just the new token's slot, so the causal mask builder can compute
    correctly).
    """
    out = model(
        input_ids=last_token_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return out.logits, out.past_key_values


@torch.no_grad()
def build_kv_cache(model, prompt_ids: torch.Tensor) -> Tuple[object, torch.Tensor, torch.Tensor]:
    """Prefill once to build a KV cache for `prompt_ids`.

    Returns (past_key_values, last_token_ids, attention_mask_for_first_decode).
    The returned attention mask covers the full prompt (length = prompt_ids.shape[1]);
    callers should extend it by one slot per decode step.
    """
    from transformers import DynamicCache

    device = prompt_ids.device
    batch_size, seq_len = prompt_ids.shape
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    past_key_values = DynamicCache(config=model.config)
    out = model(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # Last token from the prefill: take argmax of last position's logits as a
    # deterministic, sampling-free "next token". We don't actually care about
    # token quality — we only need a real id for the decode-step measurement.
    last_token_ids = out.logits[:, -1:, :].argmax(dim=-1)
    return out.past_key_values, last_token_ids, attention_mask
