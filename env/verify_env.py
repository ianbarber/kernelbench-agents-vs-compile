#!/usr/bin/env python3
"""Environment verification for the LLM-kernel vs torch.compile experiment.

Idempotent: safe to run repeatedly. Prints versions, runs a tiny
torch.compile smoke test, and a tiny manual Triton vector-add kernel.
Exits non-zero on any failure.
"""
from __future__ import annotations

import sys
import traceback
from typing import Tuple


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def report_versions() -> Tuple[object, object]:
    section("Versions")
    import torch
    print(f"torch              : {torch.__version__}")
    print(f"torch.version.cuda : {torch.version.cuda}")
    print(f"cuda available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        print(f"device name        : {name}")
        print(f"device capability  : sm_{cap[0]}{cap[1]} ({cap})")
        print(f"device count       : {torch.cuda.device_count()}")
    else:
        print("device capability  : <no cuda>")

    try:
        import triton
        print(f"triton             : {triton.__version__}")
    except Exception as e:  # pragma: no cover
        print(f"triton             : IMPORT FAILED ({e})")
        triton = None  # type: ignore

    try:
        import transformers
        print(f"transformers       : {transformers.__version__}")
    except Exception as e:
        print(f"transformers       : IMPORT FAILED ({e})")

    try:
        import accelerate
        print(f"accelerate         : {accelerate.__version__}")
    except Exception as e:
        print(f"accelerate         : IMPORT FAILED ({e})")

    return torch, triton


def smoke_torch_compile(torch) -> bool:
    section("torch.compile smoke test")
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return False
    try:
        def fn(x):
            return (x * x + x).sum()

        x = torch.randn(1024, device="cuda", dtype=torch.float32)
        eager = fn(x)
        compiled = torch.compile(fn, mode="reduce-overhead")
        out = compiled(x)
        # Warm-up may trigger recompilation; call twice.
        out = compiled(x)
        torch.cuda.synchronize()
        match = torch.allclose(eager, out, rtol=1e-4, atol=1e-4)
        print(f"eager   : {eager.item():.6f}")
        print(f"compiled: {out.item():.6f}")
        print(f"match   : {match}")
        if not match:
            print("FAIL: torch.compile output mismatch")
            return False
        print("PASS: torch.compile")
        return True
    except Exception:
        print("FAIL: torch.compile raised")
        traceback.print_exc()
        return False


def smoke_triton(torch, triton) -> bool:
    section("Triton vector-add smoke test")
    if triton is None:
        print("FAIL: triton not importable")
        return False
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return False
    try:
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        n = 4096
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        BLOCK = 256
        grid = ((n + BLOCK - 1) // BLOCK,)
        add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
        torch.cuda.synchronize()
        ref = x + y
        match = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
        max_err = (out - ref).abs().max().item()
        print(f"n={n}, BLOCK={BLOCK}, max_abs_err={max_err:.3e}, match={match}")
        if not match:
            print("FAIL: Triton kernel output mismatch")
            return False
        print("PASS: Triton vector-add")
        return True
    except Exception:
        print("FAIL: Triton kernel raised")
        traceback.print_exc()
        return False


def main() -> int:
    try:
        torch, triton = report_versions()
    except Exception:
        print("FAIL: could not import torch")
        traceback.print_exc()
        return 2

    ok_compile = smoke_torch_compile(torch)
    ok_triton = smoke_triton(torch, triton)

    section("Summary")
    print(f"torch.compile : {'PASS' if ok_compile else 'FAIL'}")
    print(f"triton kernel : {'PASS' if ok_triton else 'FAIL'}")
    overall = ok_compile and ok_triton
    print(f"overall       : {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
