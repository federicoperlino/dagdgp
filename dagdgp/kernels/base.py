"""Base kernels and shared kernel types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import inv_softplus, softplus_positive, validate_positive_finite


class Kernel(Protocol):
    """Minimal protocol for kernels used by the models."""

    input_dim: int

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        """Return the kernel matrix ``K(X1, X2)``."""

    def Kdiag(self, X: Tensor) -> Tensor:  
        """Return the diagonal of ``K(X, X)``."""


@dataclass(frozen=True)
class Block:
    """A named contiguous block in a concatenated input vector."""

    name: str
    start: int
    end: int

    @property
    def dim(self) -> int:
        return int(self.end - self.start)


class MaternKernel(nn.Module):
    """Matern kernel with optional ARD and closed-form ``nu`` options."""

    def __init__(
        self,
        input_dim: int,
        *,
        ard: bool = True,
        nu: float = 2.5,
        init_lengthscale: float = 1.0,
        init_variance: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if float(nu) not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be one of {0.5, 1.5, 2.5}.")
        validate_positive_finite(init_lengthscale, "init_lengthscale")
        validate_positive_finite(init_variance, "init_variance")

        self.input_dim = int(input_dim)
        self.ard = bool(ard)
        self.nu = float(nu)
        self.eps = float(eps)

        ls_shape = (self.input_dim,) if self.ard else (1,)
        init_ls = torch.full(ls_shape, float(init_lengthscale))
        init_var = torch.tensor(float(init_variance))

        self.raw_lengthscale = nn.Parameter(inv_softplus(init_ls))
        self.raw_variance = nn.Parameter(0.5 * torch.log(init_var))

    @property
    def lengthscale(self) -> Tensor:
        return softplus_positive(self.raw_lengthscale, minimum=self.eps)

    @property
    def variance(self) -> Tensor:
        return torch.exp(2.0 * self.raw_variance)

    def set_lengthscale(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_lengthscale)
        if target.shape != self.raw_lengthscale.shape:
            raise ValueError(f"lengthscale must have shape {tuple(self.raw_lengthscale.shape)}.")
        target = torch.clamp(target - self.eps, min=1e-12)
        with torch.no_grad():
            self.raw_lengthscale.copy_(inv_softplus(target))

    def set_variance(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_variance.dtype, device=self.raw_variance.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_variance)
        if target.shape != self.raw_variance.shape:
            raise ValueError(f"variance must have shape {tuple(self.raw_variance.shape)}.")
        with torch.no_grad():
            self.raw_variance.copy_(0.5 * torch.log(target.clamp_min(1e-12)))

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.dim() != 2 or X2.dim() != 2:
            raise ValueError("X1 and X2 must be rank-2 tensors.")
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError(
                f"MaternKernel expected input_dim={self.input_dim}, got {X1.shape[1]} and {X2.shape[1]}."
            )

        ell = self.lengthscale
        X1s = X1 / ell
        X2s = X2 / ell
        X1_sq = (X1s * X1s).sum(dim=-1, keepdim=True)
        X2_sq = (X2s * X2s).sum(dim=-1).unsqueeze(0)
        r2 = torch.clamp(X1_sq + X2_sq - 2.0 * (X1s @ X2s.transpose(0, 1)), min=0.0)
        r = torch.sqrt(torch.where(r2 > 0, r2, torch.ones_like(r2)))
        r = torch.where(r2 > 0, r, torch.zeros_like(r))

        if self.nu == 0.5:
            return self.variance * torch.exp(-r)
        if self.nu == 1.5:
            s3r = math.sqrt(3.0) * r
            return self.variance * (1.0 + s3r) * torch.exp(-s3r)

        s5r = math.sqrt(5.0) * r
        return self.variance * (1.0 + s5r + (5.0 / 3.0) * r2) * torch.exp(-s5r)

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"MaternKernel expected X shape (N,{self.input_dim}).")
        return self.variance.expand(X.shape[0])


class RBFKernel(nn.Module):
    """Squared exponential kernel with optional ARD."""

    def __init__(
        self,
        input_dim: int,
        *,
        ard: bool = True,
        init_lengthscale: float = 1.0,
        init_variance: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        validate_positive_finite(init_lengthscale, "init_lengthscale")
        validate_positive_finite(init_variance, "init_variance")

        self.input_dim = int(input_dim)
        self.ard = bool(ard)
        self.eps = float(eps)

        ls_shape = (self.input_dim,) if self.ard else (1,)
        init_ls = torch.full(ls_shape, float(init_lengthscale))
        init_var = torch.tensor(float(init_variance))

        self.raw_lengthscale = nn.Parameter(inv_softplus(init_ls))
        self.raw_variance = nn.Parameter(0.5 * torch.log(init_var))

    @property
    def lengthscale(self) -> Tensor:
        return softplus_positive(self.raw_lengthscale, minimum=self.eps)

    @property
    def variance(self) -> Tensor:
        return torch.exp(2.0 * self.raw_variance)

    def set_lengthscale(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_lengthscale)
        if target.shape != self.raw_lengthscale.shape:
            raise ValueError(f"lengthscale must have shape {tuple(self.raw_lengthscale.shape)}.")
        target = torch.clamp(target - self.eps, min=1e-12)
        with torch.no_grad():
            self.raw_lengthscale.copy_(inv_softplus(target))

    def set_variance(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_variance.dtype, device=self.raw_variance.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_variance)
        if target.shape != self.raw_variance.shape:
            raise ValueError(f"variance must have shape {tuple(self.raw_variance.shape)}.")
        with torch.no_grad():
            self.raw_variance.copy_(0.5 * torch.log(target.clamp_min(1e-12)))

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.dim() != 2 or X2.dim() != 2:
            raise ValueError("X1 and X2 must be rank-2 tensors.")
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError(
                f"RBFKernel expected input_dim={self.input_dim}, got {X1.shape[1]} and {X2.shape[1]}."
            )

        ell = self.lengthscale
        X1s = X1 / ell
        X2s = X2 / ell
        X1_sq = (X1s * X1s).sum(dim=-1, keepdim=True)
        X2_sq = (X2s * X2s).sum(dim=-1).unsqueeze(0)
        r2 = X1_sq + X2_sq - 2.0 * (X1s @ X2s.transpose(0, 1))
        r2 = torch.clamp(r2, min=0.0)
        return self.variance * torch.exp(-0.5 * r2)

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"RBFKernel expected X shape (N,{self.input_dim}).")
        return self.variance.expand(X.shape[0])


class PeriodicKernel(nn.Module):
    """Periodic kernel with optional ARD."""

    def __init__(
        self,
        input_dim: int,
        *,
        ard: bool = True,
        init_lengthscale: float = 1.0,
        init_period: float = 1.0,
        init_variance: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        validate_positive_finite(init_lengthscale, "init_lengthscale")
        validate_positive_finite(init_period, "init_period")
        validate_positive_finite(init_variance, "init_variance")

        self.input_dim = int(input_dim)
        self.ard = bool(ard)
        self.eps = float(eps)

        param_shape = (self.input_dim,) if self.ard else (1,)
        init_ls = torch.full(param_shape, float(init_lengthscale))
        init_period_t = torch.full(param_shape, float(init_period))
        init_var = torch.tensor(float(init_variance))

        self.raw_lengthscale = nn.Parameter(inv_softplus(init_ls))
        self.raw_period = nn.Parameter(inv_softplus(init_period_t))
        self.raw_variance = nn.Parameter(0.5 * torch.log(init_var))

    @property
    def lengthscale(self) -> Tensor:
        return softplus_positive(self.raw_lengthscale, minimum=self.eps)

    @property
    def period(self) -> Tensor:
        return softplus_positive(self.raw_period, minimum=self.eps)

    @property
    def variance(self) -> Tensor:
        return torch.exp(2.0 * self.raw_variance)

    def set_lengthscale(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_lengthscale)
        if target.shape != self.raw_lengthscale.shape:
            raise ValueError(f"lengthscale must have shape {tuple(self.raw_lengthscale.shape)}.")
        target = torch.clamp(target - self.eps, min=1e-12)
        with torch.no_grad():
            self.raw_lengthscale.copy_(inv_softplus(target))

    def set_period(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_period.dtype, device=self.raw_period.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_period)
        if target.shape != self.raw_period.shape:
            raise ValueError(f"period must have shape {tuple(self.raw_period.shape)}.")
        target = torch.clamp(target - self.eps, min=1e-12)
        with torch.no_grad():
            self.raw_period.copy_(inv_softplus(target))

    def set_variance(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_variance.dtype, device=self.raw_variance.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_variance)
        if target.shape != self.raw_variance.shape:
            raise ValueError(f"variance must have shape {tuple(self.raw_variance.shape)}.")
        with torch.no_grad():
            self.raw_variance.copy_(0.5 * torch.log(target.clamp_min(1e-12)))

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.dim() != 2 or X2.dim() != 2:
            raise ValueError("X1 and X2 must be rank-2 tensors.")
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError(
                f"PeriodicKernel expected input_dim={self.input_dim}, got {X1.shape[1]} and {X2.shape[1]}."
            )

        period = self.period
        lengthscale = self.lengthscale
        diff = X1[:, None, :] - X2[None, :, :]
        scaled = torch.pi * diff / period
        sin_sq = torch.sin(scaled).square()
        exponent = -2.0 * (sin_sq / lengthscale.square()).sum(dim=-1)
        return self.variance * torch.exp(exponent)

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"PeriodicKernel expected X shape (N,{self.input_dim}).")
        variance = self.variance.to(device=X.device, dtype=X.dtype)
        return variance.expand(X.shape[0])


class LinearKernel(nn.Module):
    """Linear kernel: ``k(x, x') = variance * x^T x'``."""

    def __init__(self, input_dim: int, *, init_variance: float = 1.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        validate_positive_finite(init_variance, "init_variance")

        self.input_dim = int(input_dim)
        init_var = torch.tensor(float(init_variance))
        self.raw_variance = nn.Parameter(0.5 * torch.log(init_var))

    @property
    def variance(self) -> Tensor:
        return torch.exp(2.0 * self.raw_variance)

    def set_variance(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_variance.dtype, device=self.raw_variance.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_variance)
        if target.shape != self.raw_variance.shape:
            raise ValueError(f"variance must have shape {tuple(self.raw_variance.shape)}.")
        with torch.no_grad():
            self.raw_variance.copy_(0.5 * torch.log(target.clamp_min(1e-12)))

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.dim() != 2 or X2.dim() != 2:
            raise ValueError("X1 and X2 must be rank-2 tensors.")
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError(
                f"LinearKernel expected input_dim={self.input_dim}, got {X1.shape[1]} and {X2.shape[1]}."
            )
        return self.variance * (X1 @ X2.transpose(0, 1))

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"LinearKernel expected X shape (N,{self.input_dim}).")
        return self.variance * (X.square().sum(dim=-1))


class WhiteKernel(nn.Module):
    """White noise kernel: ``k(x_n, x_m) = variance * 1[n=m]``."""

    def __init__(self, input_dim: int, *, init_variance: float = 1.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        validate_positive_finite(init_variance, "init_variance")

        self.input_dim = int(input_dim)
        init_var = torch.tensor(float(init_variance))
        self.raw_variance = nn.Parameter(0.5 * torch.log(init_var))

    @property
    def variance(self) -> Tensor:
        return torch.exp(2.0 * self.raw_variance)

    def set_variance(self, value: float | Tensor) -> None:
        target = torch.as_tensor(value, dtype=self.raw_variance.dtype, device=self.raw_variance.device)
        if target.ndim == 0:
            target = target.expand_as(self.raw_variance)
        if target.shape != self.raw_variance.shape:
            raise ValueError(f"variance must have shape {tuple(self.raw_variance.shape)}.")
        with torch.no_grad():
            self.raw_variance.copy_(0.5 * torch.log(target.clamp_min(1e-12)))

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.dim() != 2 or X2.dim() != 2:
            raise ValueError("X1 and X2 must be rank-2 tensors.")
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError(
                f"WhiteKernel expected input_dim={self.input_dim}, got {X1.shape[1]} and {X2.shape[1]}."
            )

        if X1 is X2:
            return self.variance * torch.eye(X1.shape[0], dtype=X1.dtype, device=X1.device)
        return X1.new_zeros((X1.shape[0], X2.shape[0]))

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"WhiteKernel expected X shape (N,{self.input_dim}).")
        return self.variance.to(device=X.device, dtype=X.dtype).expand(X.shape[0])
