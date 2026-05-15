"""Minimal reproducer: eager vs torch.compile on a single prefill of Qwen3-1.7B.

We do this cleanly — one model load, one input, run eager, then wrap with
compile and run again. No do_bench loop, no kv cache, no multiple workloads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workload.model import load_model  # noqa: E402


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    return float(torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b)))


def l1_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    return float((a - b).abs().mean() / (a.abs().mean() + 1e-3))


def main():
    torch.manual_seed(0)
    print("Loading Qwen3-1.7B...")
    model, _ = load_model(dtype=torch.bfloat16, device="cuda")
    print("Loaded.")

    batch, seq = 1, 512
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch, seq), device="cuda", dtype=torch.long)
    attn = torch.ones((batch, seq), device="cuda", dtype=torch.long)

    # 1) Eager pass.
    with torch.no_grad():
        out_eager = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        torch.cuda.synchronize()
    print(f"eager logits: shape={tuple(out_eager.shape)} dtype={out_eager.dtype} "
          f"nan={int(torch.isnan(out_eager).sum())} inf={int(torch.isinf(out_eager).sum())}")
    print(f"  range: min={out_eager.min().item():.3f} max={out_eager.max().item():.3f}")

    # 2) Compile pass.
    import torch._inductor.config as ic
    ic.triton.cudagraphs = False

    compiled = torch.compile(model, mode="default", dynamic=False)
    with torch.no_grad():
        out_compile = compiled(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        torch.cuda.synchronize()
    print(f"compile logits: shape={tuple(out_compile.shape)} dtype={out_compile.dtype} "
          f"nan={int(torch.isnan(out_compile).sum())} inf={int(torch.isinf(out_compile).sum())}")
    print(f"  range: min={out_compile.min().item():.3f} max={out_compile.max().item():.3f}")

    # 3) Compare.
    cs = cos_sim(out_eager, out_compile)
    l1 = l1_rel(out_eager, out_compile)
    max_abs = (out_eager.to(torch.float32) - out_compile.to(torch.float32)).abs().max().item()
    print(f"\ncos_sim={cs:.6f}  l1_rel={l1:.6f}  max_abs_diff={max_abs:.4f}")

    # Argmax agreement — most relevant for downstream sampling.
    eager_top = out_eager.argmax(dim=-1)
    compile_top = out_compile.argmax(dim=-1)
    agree = (eager_top == compile_top).float().mean().item()
    print(f"argmax agreement: {agree*100:.2f}%")

    # 4) Run it twice more to see if results are stable.
    with torch.no_grad():
        out2 = compiled(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        out3 = compiled(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        torch.cuda.synchronize()
    print(f"\nstability across compile calls:")
    print(f"  call1 vs call2 max_abs={float((out_compile - out2).abs().max()):.6f}")
    print(f"  call1 vs call3 max_abs={float((out_compile - out3).abs().max()):.6f}")


if __name__ == "__main__":
    main()
