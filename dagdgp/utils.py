"""Common utilities for DAG-DGPs."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def inv_softplus(x: Tensor) -> Tensor:
    """Numerically stable inverse of ``softplus`` for positive inputs."""
    if torch.any(x <= 0):
        raise ValueError("inv_softplus expects strictly positive inputs.")

    threshold = torch.log(torch.tensor(torch.finfo(x.dtype).max, dtype=x.dtype, device=x.device))
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


def softplus_positive(raw: Tensor, minimum: float = 1e-6) -> Tensor:
    """Map an unconstrained tensor to a strictly positive tensor."""
    return torch.nn.functional.softplus(raw) + float(minimum)


def validate_positive_finite(value: Any, name: str) -> None:
    """Validate that ``value`` is finite and strictly positive."""
    try:
        scalar = float(value)
    except Exception as exc:  
        raise TypeError(f"{name} must be a float-like value.") from exc

    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}.")
    if scalar <= 0:
        raise ValueError(f"{name} must be > 0, got {scalar}.")
