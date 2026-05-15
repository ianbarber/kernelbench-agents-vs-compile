"""Eager PyTorch reference for the SwiGLU elementwise op.

out = silu(x) * y, where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
"""
import torch


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) * y
