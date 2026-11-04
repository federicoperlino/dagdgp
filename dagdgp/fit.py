"""Unified joint ELBO training for DAG-DGP models."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .data import DAGDataset


@dataclass(frozen=True)
class FitConfig:
    """Joint optimisation defaults for DAG-VI and DAG-SVI."""

    steps: int = 5000
    joint_lr: float = 0.01
    posterior_lr_scale: float = 1.0
    inducing_unfreeze_step: int | None = None
    likelihood_unfreeze_step: int | None = None
    freeze_until_ratio: float = 0.0
    batch_size: int = 256
    num_mc_samples: int = 10
    likelihood_lr_scale: float = 0.5
    optimizer: Literal["adam", "adamw"] = "adam"
    lr_schedule: Literal["none", "cosine"] = "none"
    cosine_min_lr_ratio: float = 0.0
    seed: int = 0


def fit(
    model: nn.Module,
    dataset: DAGDataset,
    config: FitConfig,
    *,
    callback: Callable[[dict[str, float | int | str]], None] | None = None,
    start_step: int = 0,
    history: list[float] | None = None,
    optimizer_state_dict: dict[str, object] | None = None,
    optimizer_state_by_name: dict[str, dict[str, object]] | None = None,
) -> list[float]:
    """Fit a DAG-DGP model with configurable optimizer, LR schedule, and staged unfreezing."""
    torch.manual_seed(int(config.seed))

    reference = next(model.parameters())
    data = dataset.to(device=reference.device, dtype=reference.dtype)

    total_steps = int(config.steps)
    start = int(start_step)
    freeze_until_ratio = min(1.0, max(0.0, float(config.freeze_until_ratio)))
    default_unfreeze_step = int(math.ceil(float(total_steps) * freeze_until_ratio))
    inducing_unfreeze_step = (
        default_unfreeze_step
        if config.inducing_unfreeze_step is None
        else max(0, int(config.inducing_unfreeze_step))
    )
    likelihood_unfreeze_step = (
        default_unfreeze_step
        if config.likelihood_unfreeze_step is None
        else max(0, int(config.likelihood_unfreeze_step))
    )
    inducing_freeze_active = start < inducing_unfreeze_step
    likelihood_freeze_active = start < likelihood_unfreeze_step

    gp_modules = getattr(model, "gp_modules", None)
    if gp_modules is None:
        raise AttributeError("Model must expose a 'gp_modules' attribute.")
    likelihoods = getattr(model, "likelihoods", None)
    if likelihoods is None:
        raise AttributeError("Model must expose a 'likelihoods' attribute.")

    z_params: list[torch.nn.Parameter] = []
    for node in gp_modules.values():
        for name in ("Z", "Z_base"):
            param = node._parameters.get(name, None)
            if isinstance(param, torch.nn.Parameter):
                z_params.append(param)
    # Keep deterministic order but avoid accidental duplicates.
    z_params = list(dict.fromkeys(z_params))
    likelihood_params = [param for likelihood in likelihoods.values() for param in likelihood.parameters()]
    for param in z_params:
        param.requires_grad = not inducing_freeze_active
    for param in likelihood_params:
        param.requires_grad = not likelihood_freeze_active

    posterior = getattr(model, "posterior", None)
    posterior_params = [] if posterior is None else [param for param in posterior.parameters() if param.requires_grad]

    posterior_ids = {id(param) for param in posterior_params}
    z_ids = {id(param) for param in z_params}
    likelihood_ids = {id(param) for param in likelihood_params}
    other_params = [
        param
        for param in model.parameters()
        if param.requires_grad
        and id(param) not in likelihood_ids
        and id(param) not in posterior_ids
        and id(param) not in z_ids
    ]

    param_groups: list[dict[str, object]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": float(config.joint_lr)})
    if z_params:
        param_groups.append({"params": z_params, "lr": float(config.joint_lr)})

    posterior_lr = float(config.joint_lr) * max(0.0, float(config.posterior_lr_scale))
    if posterior_params and posterior_lr > 0.0:
        param_groups.append({"params": posterior_params, "lr": posterior_lr})

    likelihood_lr = float(config.joint_lr) * max(0.0, float(config.likelihood_lr_scale))
    if likelihood_params and likelihood_lr > 0.0:
        param_groups.append({"params": likelihood_params, "lr": likelihood_lr})

    optimizer_name = str(config.optimizer).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unsupported optimizer '{config.optimizer}'. Expected 'adam' or 'adamw'.")

    base_lrs = [float(group["lr"]) for group in param_groups]
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    num_data = data.num_data
    batch_size = min(int(config.batch_size), num_data)
    history_out: list[float] = [] if history is None else list(history)
    if start >= total_steps:
        model.eval()
        return history_out

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    lr_schedule = str(config.lr_schedule).lower()
    if lr_schedule == "cosine":
        min_ratio = min(1.0, max(0.0, float(config.cosine_min_lr_ratio)))
        t_max = max(total_steps, 1)

        def _lr_lambda(step: int) -> float:
            progress = min(step, t_max) / float(t_max)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        for opt_group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            opt_group["initial_lr"] = base_lr
            opt_group["lr"] = base_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        if start > 0:
            scheduler.step(start)
    elif lr_schedule != "none":
        raise ValueError(f"Unsupported lr_schedule '{config.lr_schedule}'. Expected 'none' or 'cosine'.")

    for step in tqdm(range(start, total_steps), total=total_steps, initial=start, desc="joint"):
        if inducing_freeze_active and step >= inducing_unfreeze_step:
            inducing_freeze_active = False
            for param in z_params:
                param.requires_grad = True
        if likelihood_freeze_active and step >= likelihood_unfreeze_step:
            likelihood_freeze_active = False
            for param in likelihood_params:
                param.requires_grad = True

        indices = None if batch_size >= num_data else torch.randperm(num_data, device=reference.device)[:batch_size]
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(data, indices=indices, num_mc_samples=int(config.num_mc_samples))
        elbo_value = float(elbo.detach().cpu())
        (-elbo).backward()

        if callback is not None:
            callback(
                {
                    "phase": "joint",
                    "step": step + 1,
                    "elbo": elbo_value,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                }
            )
            
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history_out.append(elbo_value)

    model.eval()
    return history_out
