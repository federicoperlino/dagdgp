"""KL utilities for Gaussian posteriors over inducing variables."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .conditionals import _kernel_cholesky_with_diag_shift
from .linalg import solve_lower_triangular


class KernelLike(Protocol):
    """Minimal kernel protocol needed by the Gaussian KL helpers."""

    input_dim: int

    def K(self, X1: Tensor, X2: Tensor) -> Tensor: ...

    def Kdiag(self, X: Tensor) -> Tensor: ...


def _validate_lower_triangular_root(root: Tensor, *, atol: float = 1e-10) -> None:
    """Validate that a root matrix is lower triangular."""
    if hasattr(torch._C, "_functorch") and torch._C._functorch.is_batchedtensor(root):
        return
    if not torch.allclose(root, torch.tril(root), atol=atol, rtol=0.0):
        raise ValueError("q_L must be lower triangular.")


def _root_logdet(root: Tensor, eps: float = 1e-12) -> Tensor:
    """Compute ``log det(root root^T)`` from a lower-triangular root."""
    _validate_lower_triangular_root(root, atol=max(float(eps), 1e-10))
    diag = torch.diagonal(root)
    return 2.0 * torch.log(diag.abs().clamp_min(eps)).sum()


def kl_from_gaussian_q(
    q_mean: Tensor,
    q_L: Tensor,
    *,
    whiten: bool,
    Luu: Tensor | None = None,
    kernel: KernelLike | None = None,
    inducing_inputs: Tensor | None = None,
    jitter: float = 1e-6,
    diag_shift: float = 0.0,
) -> Tensor:
    """Compute ``KL[q || p]`` for a Gaussian posterior over inducing variables.

    ``q_L`` is a lower-triangular root such that ``S = q_L q_L^T``.
    """

    if q_mean.dim() != 1:
        raise ValueError("q_mean must have shape (M,)")
    if q_L.dim() != 2 or q_L.shape[0] != q_L.shape[1] or q_L.shape[0] != q_mean.shape[0]:
        raise ValueError("q_L must have shape (M, M) and match q_mean")

    M = int(q_mean.numel())
    if whiten:
        trS = (q_L**2).sum()
        logdetS = _root_logdet(q_L)
        return 0.5 * (trS + (q_mean @ q_mean) - M - logdetS)

    if Luu is None:
        if kernel is None or inducing_inputs is None:
            raise ValueError("kernel and inducing_inputs are required when Luu is not provided")
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    A = solve_lower_triangular(Luu, q_L)
    tr_term = (A**2).sum()
    alpha = solve_lower_triangular(Luu, q_mean.unsqueeze(-1)).squeeze(-1)
    quad_term = (alpha**2).sum()
    logdet_kuu = 2.0 * torch.log(torch.diagonal(Luu)).sum()
    logdet_s = _root_logdet(q_L)
    return 0.5 * (tr_term + quad_term - M + logdet_kuu - logdet_s)


__all__ = ["kl_from_gaussian_q"]
