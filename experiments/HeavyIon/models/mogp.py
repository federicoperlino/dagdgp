"""LMC-SVGP baseline for the HeavyIon experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import torch

from experiments.HeavyIon.common.metrics import compute_metrics
from experiments.HeavyIon.common.real_config import set_all_seeds

TASKS = ("l1", "l2", "h")


@dataclass(frozen=True)
class MOGPRunSpec:
    seed: int
    num_latents: int = 3
    num_inducing: int = 64
    steps: int = 12000
    learning_rate: float = 0.01
    init_lengthscale: float = 1.0
    init_outputscale: float = 1.0
    init_noise: float = 0.000101
    jitter: float = 1e-05
    learn_inducing: bool = True


class LMCMultitaskGP(gpytorch.models.ApproximateGP):
    """LMC-SVGP baseline jointly modelling L1, L2, and H."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        *,
        num_tasks: int,
        init_lengthscale: float,
        init_outputscale: float,
        learn_inducing: bool,
    ) -> None:
        num_latents = int(inducing_points.shape[0])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.shape[-2], batch_shape=torch.Size([num_latents])
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
        )
        strategy = gpytorch.variational.LMCVariationalStrategy(
            base_strategy, num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1
        )
        super().__init__(strategy)
        batch_shape = torch.Size([num_latents])
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=inducing_points.shape[-1], batch_shape=batch_shape
            ),
            batch_shape=batch_shape,
        )
        self.covar_module.base_kernel.lengthscale = init_lengthscale
        self.covar_module.outputscale = init_outputscale

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _prepare_mogp_tensors(
    data: Any, *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.from_numpy(data.x_union).to(device=device, dtype=dtype)
    columns = []
    for task in TASKS:
        values = data.train_data.y[task].to(device=device, dtype=dtype).clone()
        mask = data.train_data.mask[task].to(device=device)
        values[~mask] = torch.nan
        columns.append(values)
    train_y = torch.stack(columns, dim=-1)
    test_x = torch.from_numpy(data.x_test).to(device=device, dtype=dtype)
    return (train_x, train_y, test_x)


def _build_mogp(
    train_x: torch.Tensor, spec: MOGPRunSpec
) -> tuple[LMCMultitaskGP, gpytorch.likelihoods.MultitaskGaussianLikelihood]:
    if spec.num_inducing > len(train_x):
        raise ValueError(
            "mogp_num_inducing exceeds the number of distinct training inputs."
        )
    indices = torch.randperm(len(train_x), device=train_x.device)[: spec.num_inducing]
    base = train_x.index_select(0, indices)
    inducing = base.unsqueeze(0).expand(spec.num_latents, -1, -1).clone()
    model = LMCMultitaskGP(
        inducing,
        num_tasks=len(TASKS),
        init_lengthscale=spec.init_lengthscale,
        init_outputscale=spec.init_outputscale,
        learn_inducing=spec.learn_inducing,
    ).to(device=train_x.device, dtype=train_x.dtype)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=len(TASKS), rank=0, has_global_noise=False, has_task_noise=True
    ).to(device=train_x.device, dtype=train_x.dtype)
    likelihood.task_noises = torch.full(
        (len(TASKS),), spec.init_noise, device=train_x.device, dtype=train_x.dtype
    )
    return (model, likelihood)


def _evaluate_mogp(
    *,
    data: Any,
    model: LMCMultitaskGP,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    jitter: float,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score Gaussian predictions, including noise when likelihood is supplied."""
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
        train_predictive = model(train_x)
        test_predictive = model(test_x)
        test_latent_var = test_predictive.variance.detach().cpu().numpy()
        if likelihood is not None:
            train_predictive = likelihood(train_predictive)
            test_predictive = likelihood(test_predictive)
    train_mean = train_predictive.mean.detach().cpu().numpy()
    train_var = train_predictive.variance.detach().cpu().numpy()
    test_mean = test_predictive.mean.detach().cpu().numpy()
    test_var = test_predictive.variance.detach().cpu().numpy()
    train_metrics = {}
    for task_index, task in enumerate(TASKS):
        mask = data.train_data.mask[task].cpu().numpy().astype(bool)
        truth = data.train_data.y[task].cpu().numpy()[mask]
        train_metrics[task] = compute_metrics(
            truth,
            train_mean[mask, task_index],
            train_var[mask, task_index],
            data.baseline_mean,
        )
    test_h = compute_metrics(
        data.y_test, test_mean[:, 2], test_var[:, 2], data.baseline_mean
    )
    predictions = {
        "test_h_mean": test_mean[:, 2],
        "test_h_variance": test_var[:, 2],
        "test_h_truth": data.y_test,
        "test_all_mean": test_mean,
        "test_all_variance": test_var,
        "test_h_latent_variance": test_latent_var[:, 2],
    }
    test_h_latent = compute_metrics(
        data.y_test, test_mean[:, 2], test_latent_var[:, 2], data.baseline_mean
    )
    return (
        {"test_h": test_h, "test_h_latent": test_h_latent, "train_observed": train_metrics},
        predictions,
    )


def fit_mogp(
    data: Any,
    *,
    spec: MOGPRunSpec,
    device: torch.device,
    dtype: torch.dtype,
    log_every: int = 500,
    label: str = "MO-GP (LMC-SVGP)",
):
    """Train the LMC-SVGP using Adam."""
    set_all_seeds(spec.seed)
    train_x, train_y, test_x = _prepare_mogp_tensors(data, device=device, dtype=dtype)
    model, likelihood = _build_mogp(train_x, spec)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [*model.parameters(), *likelihood.parameters()], lr=spec.learning_rate
    )
    objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_x))
    history: list[float] = []
    started = time.perf_counter()
    with (
        gpytorch.settings.cholesky_jitter(spec.jitter),
        gpytorch.settings.observation_nan_policy("mask"),
    ):
        for step in range(1, spec.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = -objective(model(train_x), train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite MOGP loss at step {step}.")
            loss.backward()
            optimizer.step()
            elbo = float((-loss).detach().cpu())
            history.append(elbo)
            if step == 1 or step % log_every == 0 or step == spec.steps:
                print(
                    f"[{label} seed={spec.seed}] step={step}/{spec.steps} elbo={elbo:.8f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    return (model, likelihood, optimizer, history, elapsed, train_x, test_x)
