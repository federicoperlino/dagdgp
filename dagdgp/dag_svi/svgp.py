"""Local sparse GP modules used by DAG-SVI."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..gaussian import (
    predict_f_from_whitened_marginal,
    sample_f_conditional_from_whitened_xi,
    sample_f_conditional_joint_from_whitened_xi,
    whitened_conditional_interpolation,
)
from ..kernels.base import Kernel


class StructuredNodeSVGP(nn.Module):
    """Local inducing-point module with a global structured posterior."""

    def __init__(
        self,
        kernel: Kernel,
        inducing_inputs: Tensor,
        *,
        jitter: float = 1e-6,
        mean_offset: float = 0.0,
    ) -> None:
        super().__init__()

        if inducing_inputs.dim() != 2:
            raise ValueError("inducing_inputs must have shape (M, D).")
        if inducing_inputs.shape[1] != getattr(kernel, "input_dim", None):
            raise ValueError(
                f"Kernel input_dim={getattr(kernel, 'input_dim', None)} does not match "
                f"inducing_inputs.shape[1]={inducing_inputs.shape[1]}."
            )

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

    @property
    def num_inducing(self) -> int:
        return int(self.Z.shape[0])

    def predict_f(self, X: Tensor, *, q_mean: Tensor, q_cov: Tensor) -> tuple[Tensor, Tensor]:
        """Return predictive mean and marginal variance under the nodewise whitened marginal."""
        mean, variance = predict_f_from_whitened_marginal(
            self.kernel,
            self.Z,
            X,
            q_mean,
            q_cov,
            jitter=self.jitter,
        )
        return mean + self.mean_offset, variance

    def sample_f(self, X: Tensor, *, xi_sample: Tensor) -> Tensor:
        """Sample pointwise from ``p(f(X) | xi_w)``."""
        return self.mean_offset + sample_f_conditional_from_whitened_xi(
            self.kernel,
            self.Z,
            X,
            xi_sample,
            jitter=self.jitter,
        )

    def conditional_statistics(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return interpolation weights and residual variances for whitened conditionals.

        The returned interpolation weights have shape ``(N, M)`` and satisfy
        ``E[f(X) | xi] = interp @ xi`` for a whitened inducing vector ``xi`` of shape ``(M,)``.
        The residual variances correspond to the conditional sparse-GP variance ``K_ff - Q_ff``
        on the diagonal.
        """
        return whitened_conditional_interpolation(
            self.kernel,
            self.Z,
            X,
            jitter=self.jitter,
        )

    def sample_f_joint(self, X: Tensor, *, xi_sample: Tensor) -> Tensor:
        """Sample jointly from ``p(f(X) | xi_w)``."""
        return self.mean_offset + sample_f_conditional_joint_from_whitened_xi(
            self.kernel,
            self.Z,
            X,
            xi_sample,
            jitter=self.jitter,
        )


class ParentMeanLinkedStructuredNodeSVGP(StructuredNodeSVGP):
    """Structured node whose inducing inputs append parent predictive means."""

    def __init__(
        self,
        kernel: Kernel,
        inducing_inputs: Tensor,
        *,
        base_input_dim: int,
        latent_tail_dim: int = 0,
        parent_mean_fn=None,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(kernel, inducing_inputs, jitter=jitter)
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
            raise RuntimeError("Parent mean function is not set for ParentMeanLinkedStructuredNodeSVGP.")
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
        return torch.cat([base_core, *parent_cols, latent_tail], dim=-1)
