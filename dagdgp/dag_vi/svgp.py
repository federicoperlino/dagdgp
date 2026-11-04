"""Whitened mean-field sparse variational GP module."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..gaussian import (
    kl_from_gaussian_q,
    predict_f_from_gaussian_q,
    predict_f_full_cov_from_gaussian_q,
    stable_cholesky,
)
from ..kernels.base import Kernel
from ..utils import inv_softplus, softplus_positive, validate_positive_finite


def _canonical_q_root_parametrization(name: str) -> str:
    """Normalise supported variational-root parametrisations."""
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "direct": "direct",
        "standard": "direct",
        "gpytorch": "direct",
        "gpytorch-like": "direct",
        "softplus": "softplus-diag",
        "softplus-diag": "softplus-diag",
    }
    if key not in aliases:
        raise ValueError("q_root_parametrization must be one of {'direct', 'softplus-diag'}.")
    return aliases[key]


def _lower_triangular_root(raw_root: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert an unconstrained square matrix into a valid Cholesky factor."""
    if raw_root.dim() != 2 or raw_root.shape[0] != raw_root.shape[1]:
        raise ValueError("raw_root must be a square matrix.")

    root = torch.tril(raw_root)
    diag = torch.diagonal(root)
    positive_diag = softplus_positive(diag, minimum=eps)
    root = root - torch.diag(diag) + torch.diag(positive_diag)
    return root


class MeanFieldSVGP(nn.Module):
    """Whitened sparse variational GP with scalar outputs."""

    def __init__(
        self,
        kernel: Kernel,
        inducing_inputs: Tensor,
        *,
        jitter: float = 1e-6,
        mean_offset: float = 0.0,
        init_q_mean: float = 0.0,
        init_q_root_scale: float = 1.0,
        q_root_parametrization: str = "direct",
        whiten: bool = True,
    ) -> None:
        super().__init__()

        if inducing_inputs.dim() != 2:
            raise ValueError("inducing_inputs must have shape (M, D).")
        if inducing_inputs.shape[1] != getattr(kernel, "input_dim", None):
            raise ValueError(
                f"Kernel input_dim={getattr(kernel, 'input_dim', None)} does not match "
                f"inducing_inputs.shape[1]={inducing_inputs.shape[1]}."
            )
        validate_positive_finite(init_q_root_scale, "init_q_root_scale")

        self.kernel = kernel
        self.Z = nn.Parameter(inducing_inputs.clone())
        self.jitter = float(jitter)
        self.register_buffer(
            "mean_offset",
            torch.as_tensor(
                float(mean_offset),
                dtype=inducing_inputs.dtype,
                device=inducing_inputs.device,
            ),
        )
        self.q_root_parametrization = _canonical_q_root_parametrization(q_root_parametrization)

        num_inducing = int(inducing_inputs.shape[0])
        device = inducing_inputs.device
        dtype = inducing_inputs.dtype

        self.q_mean = nn.Parameter(torch.full((num_inducing,), float(init_q_mean), dtype=dtype, device=device))

        init_root = torch.zeros((num_inducing, num_inducing), dtype=dtype, device=device)
        init_diag = torch.full((num_inducing,), float(init_q_root_scale), dtype=dtype, device=device)
        if self.q_root_parametrization == "softplus-diag":
            init_diag = inv_softplus(init_diag)
        init_root = init_root + torch.diag(init_diag)
        self.q_raw_root = nn.Parameter(init_root)
        self.whiten = bool(whiten)

    @property
    def num_inducing(self) -> int:
        return int(self.Z.shape[0])

    def q_root(self) -> Tensor:
        """Return the lower-triangular variational root."""
        if self.q_root_parametrization == "softplus-diag":
            return _lower_triangular_root(self.q_raw_root, eps=1e-6)
        return torch.tril(self.q_raw_root)

    def kl(self) -> Tensor:
        """Return ``KL[q(v) || p(v)]`` in whitened coordinates."""
        return kl_from_gaussian_q(
            self.q_mean,
            self.q_root(),
            whiten=self.whiten,
            kernel=self.kernel,
            inducing_inputs=self.Z,
            diag_shift=self.jitter,
        )

    def predict_f(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return predictive mean and marginal variance at ``X``."""
        mean, variance = predict_f_from_gaussian_q(
            self.kernel,
            self.Z,
            X,
            self.q_mean,
            self.q_root(),
            whiten=self.whiten,
            jitter=self.jitter,
        )
        return mean + self.mean_offset, variance

    def predict_f_full_cov(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return predictive mean and full covariance at ``X``."""
        mean, covariance = predict_f_full_cov_from_gaussian_q(
            self.kernel,
            self.Z,
            X,
            self.q_mean,
            self.q_root(),
            whiten=self.whiten,
            jitter=self.jitter,
        )
        return mean + self.mean_offset, covariance

    def sample_f(self, X: Tensor, num_samples: int) -> Tensor:
        """Draw pointwise predictive samples at ``X``."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        mean, var = self.predict_f(X)
        eps = torch.randn((num_samples, X.shape[0]), device=X.device, dtype=X.dtype)
        return mean.unsqueeze(0) + torch.sqrt(var).unsqueeze(0) * eps

    def sample_f_joint(self, X: Tensor, num_samples: int) -> Tensor:
        """Draw joint predictive samples at ``X``."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        mean, cov = self.predict_f_full_cov(X)
        chol = stable_cholesky(cov, jitter=self.jitter)
        eps = torch.randn((num_samples, X.shape[0]), device=X.device, dtype=X.dtype)
        return mean.unsqueeze(0) + eps @ chol.transpose(0, 1)


class ParentMeanLinkedMeanFieldSVGP(MeanFieldSVGP):
    """Mean-field SVGP whose inducing inputs append parent predictive means."""

    def __init__(
        self,
        kernel: Kernel,
        inducing_inputs: Tensor,
        *,
        base_input_dim: int,
        latent_tail_dim: int = 0,
        parent_mean_fn=None,
        jitter: float = 1e-6,
        init_q_mean: float = 0.0,
        init_q_root_scale: float = 1.0,
        q_root_parametrization: str = "direct",
        whiten: bool = True,
    ) -> None:
        super().__init__(
            kernel,
            inducing_inputs,
            jitter=jitter,
            init_q_mean=init_q_mean,
            init_q_root_scale=init_q_root_scale,
            q_root_parametrization=q_root_parametrization,
            whiten=whiten,
        )
        base_dim = int(base_input_dim)
        if base_dim < 0 or base_dim > inducing_inputs.shape[1]:
            raise ValueError("base_input_dim must satisfy 0 <= base_input_dim <= inducing_inputs.shape[1].")
        z_base = inducing_inputs[:, :base_dim].clone()
        del self._parameters["Z"]
        self.Z_base = nn.Parameter(z_base)
        self.base_input_dim = base_dim
        self.latent_tail_dim = int(latent_tail_dim)
        if self.latent_tail_dim < 0 or self.latent_tail_dim > self.base_input_dim:
            raise ValueError("latent_tail_dim must satisfy 0 <= latent_tail_dim <= base_input_dim.")
        self._parent_mean_fn = parent_mean_fn

    def set_parent_mean_fn(self, parent_mean_fn) -> None:
        self._parent_mean_fn = parent_mean_fn

    @property
    def num_inducing(self) -> int:
        return int(self.Z_base.shape[0])

    @property
    def Z(self) -> Tensor:
        if self._parent_mean_fn is None:
            raise RuntimeError("Parent mean function is not set for ParentMeanLinkedMeanFieldSVGP.")
        parent_means = self._parent_mean_fn(self.Z_base)
        if isinstance(parent_means, Tensor):
            parent_means = (parent_means,)
        if len(parent_means) == 0:
            return self.Z_base
        parent_cols = [m.reshape(-1, 1) for m in parent_means]
        if self.latent_tail_dim == 0:
            return torch.cat([self.Z_base, *parent_cols], dim=-1)
        base_core = self.Z_base[:, :-self.latent_tail_dim]
        latent_tail = self.Z_base[:, -self.latent_tail_dim :]
        _Z = torch.cat([base_core, *parent_cols, latent_tail], dim=-1)
        return _Z
