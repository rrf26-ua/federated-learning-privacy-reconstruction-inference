"""Metrics used to evaluate image reconstruction attacks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean squared error between two image tensors in [0, 1]."""
    return torch.mean((original.detach().cpu() - reconstructed.detach().cpu()) ** 2).item()


def psnr(original: torch.Tensor, reconstructed: torch.Tensor, max_value: float = 1.0) -> float:
    """Peak signal-to-noise ratio between two image tensors in [0, 1]."""
    value = mse(original, reconstructed)
    if value == 0:
        return float("inf")
    return 20 * math.log10(max_value) - 10 * math.log10(value)


def save_json(path: str | Path, data: dict) -> None:
    """Save metrics as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
