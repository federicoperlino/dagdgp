"""Shared Gaussian algebra utilities."""

from .conditionals import (
    predict_f_from_gaussian_q,
    predict_f_full_cov_from_gaussian_q,
    predict_f_from_whitened_marginal,
    whitened_conditional_interpolation,
    sample_f_conditional_from_u,
    sample_f_conditional_from_whitened_xi,
    sample_f_conditional_joint_from_u,
    sample_f_conditional_joint_from_whitened_xi,
)
from .kl import kl_from_gaussian_q
from .linalg import (
    safe_cholesky_solve,
    solve_lower_triangular,
    solve_right_lower_triangular,
    solve_upper_triangular,
    stable_cholesky,
)

__all__ = [
    "kl_from_gaussian_q",
    "predict_f_from_gaussian_q",
    "predict_f_full_cov_from_gaussian_q",
    "predict_f_from_whitened_marginal",
    "whitened_conditional_interpolation",
    "sample_f_conditional_from_u",
    "sample_f_conditional_from_whitened_xi",
    "sample_f_conditional_joint_from_u",
    "sample_f_conditional_joint_from_whitened_xi",
    "safe_cholesky_solve",
    "solve_lower_triangular",
    "solve_right_lower_triangular",
    "solve_upper_triangular",
    "stable_cholesky",
]
