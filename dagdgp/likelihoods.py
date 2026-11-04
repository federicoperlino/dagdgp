"""Likelihood (observational distributions) for DAG-DGPs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .utils import inv_softplus, softplus_positive, validate_positive_finite


_LOG_2PI = math.log(2.0 * math.pi)


class GaussianLikelihood(nn.Module):
    """Gaussian likelihood for scalar regression."""

    def __init__(self, init_variance: float = 1e-2, eps: float = 1e-6) -> None:
        super().__init__()
        validate_positive_finite(init_variance, "init_variance")
        self.eps = float(eps)

        init = torch.tensor([max(float(init_variance), self.eps) - self.eps]).clamp_min(1e-12)
        self.raw_variance = nn.Parameter(inv_softplus(init))

    @property
    def variance(self) -> Tensor:
        """Observation noise variance."""
        return softplus_positive(self.raw_variance, minimum=self.eps)

    def set_variance(self, variance: Tensor | float) -> None:
        """Set the observation noise variance."""
        variance_t = torch.as_tensor(
            variance,
            dtype=self.raw_variance.dtype,
            device=self.raw_variance.device,
        )
        if variance_t.numel() != 1:
            raise ValueError("variance must be scalar.")

        validate_positive_finite(float(variance_t.item()), "variance")
        target = variance_t.reshape_as(self.raw_variance).clamp_min(self.eps) - self.eps
        target = target.clamp_min(1e-12)
        with torch.no_grad():
            self.raw_variance.copy_(inv_softplus(target))

    def log_prob(self, y: Tensor, f: Tensor) -> Tensor:
        """Compute ``log p(y | f)``."""
        variance = self.variance
        return -0.5 * (torch.log(variance) + _LOG_2PI + (y - f) ** 2 / variance)

    def expected_log_prob(self, y: Tensor, f_mean: Tensor, f_var: Tensor) -> Tensor:
        """Analytic ``E_q[log p(y | f)]`` for ``q(f)=N(f_mean, f_var)``."""
        if y.shape != f_mean.shape or y.shape != f_var.shape:
            raise ValueError("y, f_mean, f_var must have the same shape.")

        variance = self.variance
        return -0.5 * (torch.log(variance) + _LOG_2PI + ((y - f_mean) ** 2 + f_var) / variance)
