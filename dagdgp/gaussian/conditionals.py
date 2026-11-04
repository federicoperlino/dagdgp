"""Shared Gaussian conditional and predictive utilities for sparse GPs."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .linalg import solve_lower_triangular, solve_upper_triangular, stable_cholesky


class KernelLike(Protocol):
    """Minimal kernel protocol needed by the Gaussian conditional helpers."""

    input_dim: int

    def K(self, X1: Tensor, X2: Tensor) -> Tensor: ...

    def Kdiag(self, X: Tensor) -> Tensor: ...


class ComputePredictiveUpdates(torch.autograd.Function):
    """Custom backward for whitened diagonal predictive updates."""

    @staticmethod
    def forward(
        ctx,
        chol: Tensor,
        induc_data_covar: Tensor,
        middle: Tensor,
        inducing_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        interp_term = solve_lower_triangular(chol, induc_data_covar)
        mean_update = (interp_term.transpose(0, 1) @ inducing_values.unsqueeze(-1)).squeeze(-1)
        variance_update = torch.sum(interp_term.transpose(0, 1) * (interp_term.transpose(0, 1) @ middle), dim=-1)
        ctx.save_for_backward(chol, interp_term, middle, inducing_values)
        return mean_update, variance_update

    @staticmethod
    def backward(
        ctx,
        d_mean: Tensor,
        d_variance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chol, interp_term, middle, inducing_values = ctx.saved_tensors

        interp_term_times_dmean = interp_term @ d_mean.unsqueeze(-1)
        interp_term_scaled_dvariance = interp_term * d_variance.unsqueeze(0)

        inv_chol_times_middle = solve_upper_triangular(chol.transpose(0, 1), middle)
        inv_chol_times_inducing_values = solve_upper_triangular(chol.transpose(0, 1), inducing_values.unsqueeze(-1))

        d_middle = interp_term_scaled_dvariance @ interp_term.transpose(0, 1)
        d_inducing_values = interp_term_times_dmean.squeeze(-1)
        d_induc_data_covar = 2.0 * inv_chol_times_middle @ interp_term_scaled_dvariance
        d_induc_data_covar = d_induc_data_covar + inv_chol_times_inducing_values @ d_mean.unsqueeze(0)
        d_chol = -2.0 * inv_chol_times_middle @ d_middle
        d_chol = d_chol - inv_chol_times_inducing_values @ interp_term_times_dmean.transpose(0, 1)
        d_chol = torch.tril(d_chol)
        return d_chol, d_induc_data_covar, d_middle, d_inducing_values


def _matches_inducing_inputs(X: Tensor, inducing_inputs: Tensor, *, jitter: float) -> bool:
    """Detect the exact-conditioning case ``X == Z``.

    If the query inputs coincide with the inducing inputs, conditioning on
    ``u = f(Z)`` should reproduce those values exactly with zero conditional
    variance. Numerical jitter used for Cholesky factorisation must not leak
    into this semantic identity.
    """

    if X.shape != inducing_inputs.shape:
        return False
    atol = max(float(jitter) * 10.0, 1e-12)
    return torch.allclose(X, inducing_inputs, atol=atol, rtol=0.0)


def _validate_inputs(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    q_mean: Tensor | None = None,
    q_L: Tensor | None = None,
    u_sample: Tensor | None = None,
) -> None:
    if X.dim() != 2:
        raise ValueError("X must have shape (N, D)")
    if inducing_inputs.dim() != 2:
        raise ValueError("inducing_inputs must have shape (M, D)")
    if X.shape[1] != getattr(kernel, "input_dim", None):
        raise ValueError("Input dimension mismatch.")
    if inducing_inputs.shape[1] != getattr(kernel, "input_dim", None):
        raise ValueError("Inducing input dimension mismatch.")
    if q_mean is not None and q_mean.dim() != 1:
        raise ValueError("q_mean must have shape (M,)")
    if q_L is not None:
        if q_L.dim() != 2 or q_L.shape[0] != q_L.shape[1]:
            raise ValueError("q_L must have shape (M, M)")
        if q_mean is None or q_L.shape[0] != q_mean.shape[0]:
            raise ValueError("q_L must match q_mean")
    if u_sample is not None:
        if u_sample.dim() != 1 or u_sample.shape[0] != inducing_inputs.shape[0]:
            raise ValueError("u_sample must have shape (M,)")


def _kernel_cholesky_with_diag_shift(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    *,
    jitter: float,
    diag_shift: float,
) -> tuple[Tensor, Tensor]:
    """Build ``Kuu`` and its Cholesky factor for the shifted GP prior."""

    Kuu = kernel.K(inducing_inputs, inducing_inputs)
    Kuu = 0.5 * (Kuu + Kuu.transpose(0, 1))
    if float(diag_shift) != 0.0:
        eye = torch.eye(Kuu.shape[0], device=Kuu.device, dtype=Kuu.dtype)
        Kuu = Kuu + float(diag_shift) * eye

    Luu = stable_cholesky(Kuu, jitter=jitter)
    return Kuu, Luu


def predict_f_from_gaussian_q(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    q_mean: Tensor,
    q_L: Tensor,
    *,
    whiten: bool,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Predict marginal mean and variance of ``f(X)`` under a Gaussian ``q(u)``."""

    _validate_inputs(kernel, inducing_inputs, X, q_mean=q_mean, q_L=q_L)

    if not whiten and _matches_inducing_inputs(X, inducing_inputs, jitter=jitter):
        q_cov_diag = torch.diagonal(q_L @ q_L.transpose(0, 1))
        return q_mean, torch.clamp(q_cov_diag, min=1e-9)

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)

    if whiten:
        if A.shape[0] < A.shape[1]:
            q_cov = q_L @ q_L.transpose(0, 1)
            eye = torch.eye(q_cov.shape[0], dtype=q_cov.dtype, device=q_cov.device)
            middle = q_cov - eye
            mean_update, var_update = ComputePredictiveUpdates.apply(Luu, Kuf, middle, q_mean)
            mean = mean_update
            var = kernel.Kdiag(X) + float(diag_shift) + var_update
            var = torch.clamp(var, min=1e-9)
            return mean, var
        mean = A.transpose(0, 1) @ q_mean
        diag_q = ((q_L.transpose(0, 1) @ A) ** 2).sum(dim=0)
        diag_p = (A**2).sum(dim=0)
    else:
        A2 = solve_upper_triangular(Luu.transpose(0, 1), A)
        mean = A2.transpose(0, 1) @ q_mean
        diag_q = ((q_L.transpose(0, 1) @ A2) ** 2).sum(dim=0)
        diag_p = (A**2).sum(dim=0)

    var = kernel.Kdiag(X) + float(diag_shift) + diag_q - diag_p
    var = torch.clamp(var, min=1e-9)
    return mean, var


def predict_f_full_cov_from_gaussian_q(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    q_mean: Tensor,
    q_L: Tensor,
    *,
    whiten: bool,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Predict mean and full covariance of ``f(X)`` under a Gaussian ``q(u)``."""

    _validate_inputs(kernel, inducing_inputs, X, q_mean=q_mean, q_L=q_L)

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    Kff = kernel.K(X, X)
    Kff = 0.5 * (Kff + Kff.transpose(0, 1))
    if float(diag_shift) != 0.0:
        eye = torch.eye(Kff.shape[0], device=Kff.device, dtype=Kff.dtype)
        Kff = Kff + float(diag_shift) * eye

    A1 = solve_lower_triangular(Luu, Kuf)

    if whiten:
        mean = A1.transpose(0, 1) @ q_mean
        q_cov = q_L @ q_L.transpose(0, 1)
        eye = torch.eye(q_cov.shape[0], device=q_cov.device, dtype=q_cov.dtype)
        cov = Kff + A1.transpose(0, 1) @ (q_cov - eye) @ A1
    else:
        A2 = solve_upper_triangular(Luu.transpose(0, 1), A1)
        mean = A2.transpose(0, 1) @ q_mean
        cov = Kff - A1.transpose(0, 1) @ A1
        q_factor = q_L.transpose(0, 1) @ A2
        cov = cov + q_factor.transpose(0, 1) @ q_factor

    cov = 0.5 * (cov + cov.transpose(0, 1))
    return mean, cov


def whitened_conditional_interpolation(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    *,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return interpolation weights and residual variances for whitened conditionals.

    For each input ``X[i]``, the conditional distribution is
    ``f_i | xi ~ N(interp[i] @ xi, residual[i])``.

    The interpolation weights have shape ``(N, M)``. The residual variances
    are the diagonal entries of ``K_ff - Q_ff``.
    """

    _validate_inputs(kernel, inducing_inputs, X)

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)
    residual = kernel.Kdiag(X) + float(diag_shift) - (A**2).sum(dim=0)
    residual = torch.clamp(residual, min=1e-9)
    return A.transpose(0, 1), residual


def predict_f_from_whitened_marginal(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    q_mean: Tensor,
    q_cov: Tensor,
    *,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Predict ``f(X)`` when the whitened inducing marginal is ``N(q_mean, q_cov)``."""

    _validate_inputs(kernel, inducing_inputs, X, q_mean=q_mean)
    if q_cov.dim() != 2 or q_cov.shape[0] != q_cov.shape[1] or q_cov.shape[0] != q_mean.shape[0]:
        raise ValueError("q_cov must have shape (M, M) and match q_mean.")

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)
    mean = A.transpose(0, 1) @ q_mean
    eye = torch.eye(q_cov.shape[0], device=q_cov.device, dtype=q_cov.dtype)
    middle = q_cov - eye
    var = kernel.Kdiag(X) + float(diag_shift) + torch.sum(A * (middle @ A), dim=0)
    var = torch.clamp(var, min=1e-9)
    return mean, var


def sample_f_conditional_from_u(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    u_sample: Tensor,
    *,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
    eps: Tensor | None = None,
) -> Tensor:
    """Sample from the sparse conditional ``p(f(X) | u)`` with diagonal covariance."""

    _validate_inputs(kernel, inducing_inputs, X, u_sample=u_sample)

    if _matches_inducing_inputs(X, inducing_inputs, jitter=jitter):
        return u_sample

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)
    A2 = solve_upper_triangular(Luu.transpose(0, 1), A)
    mean = A2.transpose(0, 1) @ u_sample
    var = kernel.Kdiag(X) + float(diag_shift) - (A**2).sum(dim=0)
    var = torch.clamp(var, min=1e-9)

    if eps is None:
        eps = torch.randn_like(mean)
    else:
        if eps.shape != mean.shape:
            raise ValueError("eps must have shape (N,)")
        eps = eps.to(device=mean.device, dtype=mean.dtype)
    return mean + torch.sqrt(var) * eps


def sample_f_conditional_from_whitened_xi(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    xi_sample: Tensor,
    *,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
    eps: Tensor | None = None,
) -> Tensor:
    """Sample from ``p(f(X) | xi)`` in whitened coordinates with diagonal covariance."""

    _validate_inputs(kernel, inducing_inputs, X, u_sample=xi_sample)

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)
    mean = A.transpose(0, 1) @ xi_sample
    var = kernel.Kdiag(X) + float(diag_shift) - (A**2).sum(dim=0)
    var = torch.clamp(var, min=1e-9)

    if eps is None:
        eps = torch.randn_like(mean)
    else:
        if eps.shape != mean.shape:
            raise ValueError("eps must have shape (N,)")
        eps = eps.to(device=mean.device, dtype=mean.dtype)
    return mean + torch.sqrt(var) * eps


def sample_f_conditional_joint_from_u(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    u_sample: Tensor,
    *,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
    eps: Tensor | None = None,
) -> Tensor:
    """Sample jointly from the full conditional ``p(f(X) | u)``."""

    _validate_inputs(kernel, inducing_inputs, X, u_sample=u_sample)

    if _matches_inducing_inputs(X, inducing_inputs, jitter=jitter):
        return u_sample

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)
    A2 = solve_upper_triangular(Luu.transpose(0, 1), A)
    mean = A2.transpose(0, 1) @ u_sample

    Kff = kernel.K(X, X)
    Kff = 0.5 * (Kff + Kff.transpose(0, 1))
    if float(diag_shift) != 0.0:
        eye = torch.eye(Kff.shape[0], device=Kff.device, dtype=Kff.dtype)
        Kff = Kff + float(diag_shift) * eye
    cov = Kff - A.transpose(0, 1) @ A
    cov = 0.5 * (cov + cov.transpose(0, 1))
    chol = stable_cholesky(cov, jitter=jitter)

    if eps is None:
        eps = torch.randn((X.shape[0],), device=mean.device, dtype=mean.dtype)
    else:
        if eps.shape != mean.shape:
            raise ValueError("eps must have shape (N,)")
        eps = eps.to(device=mean.device, dtype=mean.dtype)
    return mean + chol @ eps


def sample_f_conditional_joint_from_whitened_xi(
    kernel: KernelLike,
    inducing_inputs: Tensor,
    X: Tensor,
    xi_sample: Tensor,
    *,
    jitter: float,
    Luu: Tensor | None = None,
    diag_shift: float = 0.0,
    eps: Tensor | None = None,
) -> Tensor:
    """Sample jointly from ``p(f(X) | xi)`` in whitened coordinates."""

    _validate_inputs(kernel, inducing_inputs, X, u_sample=xi_sample)

    if Luu is None:
        _, Luu = _kernel_cholesky_with_diag_shift(
            kernel,
            inducing_inputs,
            jitter=jitter,
            diag_shift=diag_shift,
        )

    Kuf = kernel.K(inducing_inputs, X)
    A = solve_lower_triangular(Luu, Kuf)
    mean = A.transpose(0, 1) @ xi_sample

    Kff = kernel.K(X, X)
    Kff = 0.5 * (Kff + Kff.transpose(0, 1))
    if float(diag_shift) != 0.0:
        eye = torch.eye(Kff.shape[0], device=Kff.device, dtype=Kff.dtype)
        Kff = Kff + float(diag_shift) * eye
    cov = Kff - A.transpose(0, 1) @ A
    cov = 0.5 * (cov + cov.transpose(0, 1))
    chol = stable_cholesky(cov, jitter=jitter)

    if eps is None:
        eps = torch.randn((X.shape[0],), device=mean.device, dtype=mean.dtype)
    else:
        if eps.shape != mean.shape:
            raise ValueError("eps must have shape (N,)")
        eps = eps.to(device=mean.device, dtype=mean.dtype)
    return mean + chol @ eps


__all__ = [
    "predict_f_from_gaussian_q",
    "predict_f_full_cov_from_gaussian_q",
    "predict_f_from_whitened_marginal",
    "sample_f_conditional_from_u",
    "sample_f_conditional_from_whitened_xi",
    "sample_f_conditional_joint_from_u",
    "sample_f_conditional_joint_from_whitened_xi",
    "whitened_conditional_interpolation",
]
