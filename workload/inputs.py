"""Canonical workload input batches for Qwen3-1.7B benchmarking.

All RNG is seeded so the same workload name always produces the same tensors.
Token ids are random integers in `[0, vocab_size)`; we don't care that they're
nonsense text — we want repeatable shapes for performance measurement.

Workloads:
  - prefill_{512,2048}_b1                : prefill at seq_len, batch=1
  - decode_ctx{512,2048}_b{1,8}          : single decode step, cache prebuilt
                                            to the named context length

Decode workloads expose a `kv_cache_builder(model)` callable which runs the
prefill on the model and returns `(past_key_values, last_token_ids,
attention_mask)`. The attention mask returned covers the full context (length =
ctx_len); the caller must extend it by 1 (to ctx_len + 1) before passing to
`decode_fn`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

# Qwen3-1.7B uses the Qwen2/Qwen3 tokenizer with vocab_size = 151936.
# We hardcode this so `inputs.py` can be imported without the model weights.
# Sanity-checked against the published config on HF.
QWEN3_VOCAB_SIZE = 151936

# Master seed; per-workload we derive a sub-seed by hashing the name so two
# workloads of the same shape still differ.
MASTER_SEED = 0xC0FFEE


def _seeded_generator(name: str) -> torch.Generator:
    g = torch.Generator(device="cpu")
    # hashlib for cross-process stability — built-in hash() is randomized per
    # Python process (PYTHONHASHSEED), which would produce different tensors
    # in run_eager.py vs run_compile.py and silently break correctness checks.
    name_hash = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    sub = (MASTER_SEED ^ name_hash) & 0x7FFFFFFF
    g.manual_seed(sub)
    return g


def _make_input_ids(batch_size: int, seq_len: int, name: str) -> torch.Tensor:
    g = _seeded_generator(name)
    return torch.randint(
        low=0,
        high=QWEN3_VOCAB_SIZE,
        size=(batch_size, seq_len),
        generator=g,
        dtype=torch.long,
    )


def _make_attention_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.ones((batch_size, seq_len), dtype=torch.long)


def _prefill_workload(name: str, seq_len: int, batch_size: int) -> Dict:
    input_ids = _make_input_ids(batch_size, seq_len, name)
    attention_mask = _make_attention_mask(batch_size, seq_len)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mode": "prefill",
        "seq_len": seq_len,
        "batch_size": batch_size,
        "name": name,
        "kv_cache_builder": None,
    }


def _decode_workload(name: str, ctx_len: int, batch_size: int) -> Dict:
    # The prompt tokens used to prefill the KV cache.
    prompt_ids = _make_input_ids(batch_size, ctx_len, name + ":prompt")
    # The "current" decode-step token (shape [B, 1]). In practice the cache
    # builder will return a real argmax token from the prefill, but we also
    # include a fixed-seed fallback so `input_ids` is meaningful on its own.
    next_ids = _make_input_ids(batch_size, 1, name + ":next")
    full_attn_mask = _make_attention_mask(batch_size, ctx_len + 1)

    def kv_cache_builder(model):
        # Lazy import to avoid pulling torch/transformers at module-import time
        # in code paths that only want shapes.
        from workload.model import build_kv_cache

        device = next(model.parameters()).device
        past_key_values, last_token_ids, attn = build_kv_cache(
            model, prompt_ids.to(device)
        )
        # Extend the attention mask by 1 to cover the upcoming decode token.
        decode_mask = torch.ones(
            (batch_size, ctx_len + 1), dtype=torch.long, device=device
        )
        return {
            "past_key_values": past_key_values,
            "last_token_ids": last_token_ids,
            "attention_mask": decode_mask,
        }

    return {
        # For decode mode, `input_ids` is the single decode-step token.
        "input_ids": next_ids,
        "attention_mask": full_attn_mask,
        "mode": "decode",
        "seq_len": ctx_len,  # context length already in cache
        "batch_size": batch_size,
        "name": name,
        "prompt_ids": prompt_ids,
        "kv_cache_builder": kv_cache_builder,
    }


_REGISTRY: Dict[str, Callable[[], Dict]] = {
    "prefill_512_b1": lambda: _prefill_workload("prefill_512_b1", 512, 1),
    "prefill_2048_b1": lambda: _prefill_workload("prefill_2048_b1", 2048, 1),
    "decode_ctx512_b1": lambda: _decode_workload("decode_ctx512_b1", 512, 1),
    "decode_ctx512_b8": lambda: _decode_workload("decode_ctx512_b8", 512, 8),
    "decode_ctx2048_b1": lambda: _decode_workload("decode_ctx2048_b1", 2048, 1),
    "decode_ctx2048_b8": lambda: _decode_workload("decode_ctx2048_b8", 2048, 8),
}


def list_workloads():
    return sorted(_REGISTRY.keys())


def get_workload(name: str) -> Dict:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown workload '{name}'. Available: {list_workloads()}"
        )
    return _REGISTRY[name]()
