"""Local training loops for the HeavyIon experiment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from dagdgp.data import DAGDataset
from dagdgp.dag_vi.dag_model import DAGVIModel
from dagdgp.dag_svi.dag_model import DAGSVIModel


@dataclass(frozen=True)
class DAGVIFitConfig:
    """Joint optimisation defaults for DAG-VI."""

    steps: int = 5000
    joint_lr: float = 0.01
    batch_size: int = 256
    num_mc_samples: int = 10
    likelihood_lr_scale: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class DAGSVIFitConfig:
    """Joint optimisation defaults for DAG-SVI."""

    steps: int = 5000
    joint_lr: float = 0.01
    posterior_lr_scale: float = 1.0
    batch_size: int = 256
    num_mc_samples: int = 10
    likelihood_lr_scale: float = 0.5
    seed: int = 0


def fit_dag_vi(
    model: DAGVIModel,
    dataset: DAGDataset,
    config: DAGVIFitConfig,
    *,
    callback: Callable[[dict[str, float | int | str]], None] | None = None,
) -> list[float]:
    """Fit a DAG-VI model with a simple Adam loop."""
    torch.manual_seed(int(config.seed))
    reference = next(model.parameters())
    data = dataset.to(device=reference.device, dtype=reference.dtype)
    likelihood_params = [
        param
        for likelihood in model.likelihoods.values()
        for param in likelihood.parameters()
        if param.requires_grad
    ]
    likelihood_ids = {id(param) for param in likelihood_params}
    other_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in likelihood_ids
    ]
    param_groups: list[dict[str, object]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": float(config.joint_lr)})
    likelihood_lr = float(config.joint_lr) * max(0.0, float(config.likelihood_lr_scale))
    if likelihood_params and likelihood_lr > 0.0:
        param_groups.append({"params": likelihood_params, "lr": likelihood_lr})
    optimizer = torch.optim.Adam(param_groups)
    num_data = data.num_data
    batch_size = min(int(config.batch_size), num_data)
    history: list[float] = []
    for step in range(int(config.steps)):
        indices = (
            None
            if batch_size >= num_data
            else torch.randperm(num_data, device=reference.device)[:batch_size]
        )
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(
            data, indices=indices, num_mc_samples=int(config.num_mc_samples)
        )
        elbo_value = float(elbo.detach().cpu())
        if callback is not None:
            callback({"phase": "joint", "step": step + 1, "elbo": float(elbo.detach())})
        (-elbo).backward()
        optimizer.step()
        history.append(elbo_value)
    model.eval()
    return history


def fit_dag_svi(
    model: DAGSVIModel,
    dataset: DAGDataset,
    config: DAGSVIFitConfig,
    *,
    callback: Callable[[dict[str, float | int | str]], None] | None = None,
) -> list[float]:
    """Fit a DAG-SVI DAG model with a simple Adam loop."""
    torch.manual_seed(int(config.seed))
    reference = next(model.parameters())
    data = dataset.to(device=reference.device, dtype=reference.dtype)
    posterior_params = [
        param for param in model.posterior.parameters() if param.requires_grad
    ]
    posterior_ids = {id(param) for param in posterior_params}
    likelihood_params = [
        param
        for likelihood in model.likelihoods.values()
        for param in likelihood.parameters()
        if param.requires_grad
    ]
    likelihood_ids = {id(param) for param in likelihood_params}
    other_params = [
        param
        for param in model.parameters()
        if param.requires_grad
        and id(param) not in likelihood_ids
        and (id(param) not in posterior_ids)
    ]
    param_groups: list[dict[str, object]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": float(config.joint_lr)})
    posterior_lr = float(config.joint_lr) * max(0.0, float(config.posterior_lr_scale))
    if posterior_params and posterior_lr > 0.0:
        param_groups.append({"params": posterior_params, "lr": posterior_lr})
    likelihood_lr = float(config.joint_lr) * max(0.0, float(config.likelihood_lr_scale))
    if likelihood_params and likelihood_lr > 0.0:
        param_groups.append({"params": likelihood_params, "lr": likelihood_lr})
    optimizer = torch.optim.Adam(param_groups)
    num_data = data.num_data
    batch_size = min(int(config.batch_size), num_data)
    history: list[float] = []
    for step in range(int(config.steps)):
        indices = (
            None
            if batch_size >= num_data
            else torch.randperm(num_data, device=reference.device)[:batch_size]
        )
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(
            data, indices=indices, num_mc_samples=int(config.num_mc_samples)
        )
        elbo_value = float(elbo.detach().cpu())
        if callback is not None:
            callback({"phase": "joint", "step": step + 1, "elbo": float(elbo.detach())})
        (-elbo).backward()
        optimizer.step()
        history.append(elbo_value)
    model.eval()
    return history
