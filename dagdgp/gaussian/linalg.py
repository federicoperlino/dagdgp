"""Shared Gaussian linear-algebra primitives."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor


_LARGE_CHOLESKY_JITTER_WARNING_THRESHOLD = 1e-3


def solve_lower_triangular(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``matrix @ x = rhs`` for lower-triangular ``matrix``."""
    return torch.linalg.solve_triangular(matrix, rhs, upper=False)


def solve_upper_triangular(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``matrix @ x = rhs`` for upper-triangular ``matrix``."""
    return torch.linalg.solve_triangular(matrix, rhs, upper=True)


def solve_right_lower_triangular(rhs: Tensor, matrix: Tensor) -> Tensor:
    """Solve ``x @ matrix = rhs`` for lower-triangular ``matrix``."""
    return solve_upper_triangular(matrix.transpose(-1, -2), rhs.transpose(-1, -2)).transpose(-1, -2)


def safe_cholesky_solve(rhs: Tensor, chol: Tensor) -> Tensor:
    """Solve ``(chol chol^T) x = rhs`` robustly across CPU and CUDA."""
    if chol.device.type == "cuda":
        solved = solve_lower_triangular(chol, rhs)
        return solve_upper_triangular(chol.transpose(-1, -2), solved)
    return torch.cholesky_solve(rhs, chol)


def stable_cholesky(matrix: Tensor, jitter: float) -> Tensor:
    """Return a Cholesky factor, escalating diagonal jitter if needed."""
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    base_jitter = max(float(jitter), 0.0)

    if base_jitter == 0.0:
        base_jitter = 1e-9

    jitter_schedule = [base_jitter * (10.0**k) for k in range(6)]


    warned_large_jitter = False
    for diag_jitter in jitter_schedule:
        if diag_jitter >= _LARGE_CHOLESKY_JITTER_WARNING_THRESHOLD and not warned_large_jitter:
            warnings.warn(
                (
                    "stable_cholesky escalated diagonal jitter to "
                    f"{diag_jitter:.1e} (base={base_jitter:.1e}); this often indicates an "
                    "ill-conditioned kernel/covariance matrix and can materially alter the posterior."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            warned_large_jitter = True
        candidate = matrix + diag_jitter * eye
        try:
            return torch.linalg.cholesky(candidate)
        except RuntimeError:
            continue
    raise RuntimeError("Cholesky factorisation failed even after jitter escalation.")
