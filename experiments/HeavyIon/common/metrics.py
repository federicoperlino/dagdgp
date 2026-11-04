"""Marginal predictive scores for HeavyIon."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def gaussian_crps(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    y_t = torch.from_numpy(np.asarray(y_true, dtype=np.float64))
    mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float64))
    std_t = torch.sqrt(
        torch.clamp(torch.from_numpy(np.asarray(var, dtype=np.float64)), min=1e-12)
    )
    z = (y_t - mean_t) / std_t
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = std_t * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return float(crps.mean().item())


def compute_metrics(
    y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, baseline_mean: float
) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((mean - y_true) ** 2)))
    rmse_base = float(np.sqrt(np.mean((baseline_mean - y_true) ** 2)))
    return {
        "rmse": rmse,
        "n_rmse": 1.0 - rmse / rmse_base,
        "crps": gaussian_crps(y_true, mean, var),
    }


def summarise_scalar_samples(samples: Tensor) -> dict[str, Tensor]:
    if samples.dim() != 2:
        raise ValueError("samples must have shape (num_samples, num_points).")

    quantiles = torch.quantile(
        samples,
        torch.tensor([0.05, 0.5, 0.95], device=samples.device, dtype=samples.dtype),
        dim=0,
    )
    return {
        "mean": samples.mean(dim=0),
        "variance": samples.var(dim=0, unbiased=False),
        "std": samples.std(dim=0, unbiased=False),
        "q05": quantiles[0],
        "q50": quantiles[1],
        "q95": quantiles[2],
    }
