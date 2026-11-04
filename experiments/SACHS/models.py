"""DAG-DGP and multi-output GP models for Sachs."""

from __future__ import annotations

import random

import gpytorch
import numpy as np
import torch

from dagdgp import (
    DAG,
    DAGDataset,
    GaussianLikelihood,
    DAGVIModel,
    MeanFieldSVGP,
    DAGSVIModel,
    StructuredNodeSVGP,
)
from dagdgp.kernels import RBFKernel
from dagdgp.dag_vi.inference import _node_likelihood_term_mean_field
from dagdgp.utils import inv_softplus
from config import Config, MODEL_SEED, NODES, PARENTS, ROOTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inducing_grid(source: torch.Tensor, count: int) -> torch.Tensor:
    resolution = int(np.ceil(count ** (1.0 / source.shape[1])))
    levels = [
        torch.linspace(
            source[:, dim].min(),
            source[:, dim].max(),
            resolution,
            dtype=source.dtype,
            device=source.device,
        )
        for dim in range(source.shape[1])
    ]
    flat = torch.arange(count, device=source.device)
    coordinates = torch.empty(
        (count, source.shape[1]), dtype=torch.long, device=source.device
    )
    cursor = flat
    for dim in range(source.shape[1] - 1, -1, -1):
        coordinates[:, dim] = cursor % resolution
        cursor = cursor // resolution
    return torch.stack(
        [levels[dim][coordinates[:, dim]] for dim in range(source.shape[1])],
        dim=-1,
    )


def initialise_dag_svi_from_dag_vi(
    model: DAGSVIModel, mf_state: dict[str, torch.Tensor]
) -> None:
    loaded = model.load_state_dict(mf_state, strict=False)
    if any(not key.endswith((".q_mean", ".q_raw_root")) for key in loaded.unexpected_keys):
        raise RuntimeError(f"Unexpected MF state keys: {loaded.unexpected_keys}")
    if any(not key.startswith("posterior.") for key in loaded.missing_keys):
        raise RuntimeError(f"Unexpected missing structured keys: {loaded.missing_keys}")

    posterior = model.posterior
    means, precision_roots = [], []
    for node in posterior.structure.order:
        q_mean = mf_state[f"gp_modules.{node}.q_mean"].to(
            device=posterior.q_mean.device, dtype=posterior.q_mean.dtype
        )
        raw_root = mf_state[f"gp_modules.{node}.q_raw_root"].to(
            device=posterior.q_mean.device, dtype=posterior.q_mean.dtype
        )
        q_root = torch.tril(raw_root)
        covariance = q_root @ q_root.transpose(-1, -2)
        precision = torch.linalg.inv(
            0.5 * (covariance + covariance.transpose(-1, -2))
        )
        precision_roots.append(
            torch.linalg.cholesky(
                0.5 * (precision + precision.transpose(-1, -2))
            )
        )
        means.append(q_mean.reshape(-1))

    with torch.no_grad():
        posterior.q_mean.copy_(torch.cat(means))
        for index, root in enumerate(precision_roots):
            raw = torch.tril(root)
            if posterior.diag_parametrization == "softplus-diag":
                diagonal = torch.diagonal(raw)
                raw = raw - torch.diag(diagonal) + torch.diag(
                    inv_softplus(diagonal)
                )
            posterior.raw_diag_blocks[index].copy_(raw)
        for key in posterior.raw_offdiag_blocks:
            posterior.raw_offdiag_blocks[key].zero_()


class LMCMultitaskGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing: torch.Tensor) -> None:
        num_latents = inducing.shape[0]
        q = gpytorch.variational.CholeskyVariationalDistribution(
            inducing.shape[-2], batch_shape=torch.Size([num_latents])
        )
        base = gpytorch.variational.VariationalStrategy(
            self, inducing, q, learn_inducing_locations=True
        )
        strategy = gpytorch.variational.LMCVariationalStrategy(
            base,
            num_tasks=len(NODES),
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(strategy)
        batch = torch.Size([num_latents])
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2, batch_shape=batch),
            batch_shape=batch,
        )
        self.covar_module.base_kernel.lengthscale = 0.1
        self.covar_module.outputscale = 1.0

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def build_dag_model(
    family: str, dataset: DAGDataset, train_mask: np.ndarray,
    cfg: Config, device: torch.device,
) -> DAGVIModel | DAGSVIModel:
    set_seed(MODEL_SEED)
    dag = DAG(PARENTS)
    latent_dims = {"PKA": 1, "Raf": 1}
    initialise_from = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    modules = {}
    for node in dag.non_roots:
        columns = []
        for parent in dag.parents_of(node):
            if parent in dataset.roots:
                values = dataset.roots[parent].squeeze(-1)
                if cfg.protocol == "scalability":
                    values = values[initialise_from]
            else:
                values, observed = dataset.get_observations(parent)
                if cfg.protocol == "scalability":
                    observed = observed & initialise_from
                values = values[observed]
            if cfg.protocol == "extrapolation":
                column = torch.linspace(values.min(), values.max(), cfg.num_inducing)
            else:
                column = torch.linspace(
                    values.min(), values.max(), cfg.num_inducing,
                    dtype=values.dtype, device=values.device,
                )
            columns.append(column.unsqueeze(-1))
        inducing = inducing_grid(torch.cat(columns, dim=-1), cfg.num_inducing)
        latent_dim = latent_dims.get(node, 0)
        if latent_dim:
            inducing = torch.cat([
                inducing,
                torch.zeros(cfg.num_inducing, latent_dim, dtype=inducing.dtype, device=inducing.device),
            ], dim=-1)
        kernel = RBFKernel(
            inducing.shape[1], ard=True, init_lengthscale=0.1,
            init_variance=1.0, eps=1e-9,
        )
        if family == "dag_vi":
            modules[node] = MeanFieldSVGP(kernel, inducing, jitter=1e-5)
        elif family == "dag_svi":
            modules[node] = StructuredNodeSVGP(kernel, inducing, jitter=1e-5)
        else:
            raise ValueError(f"Unknown DAG model: {family}")
    common = {
        "dag": dag,
        "root_dims": {root: 1 for root in ROOTS},
        "gp_modules": modules,
        "likelihoods": {
            node: GaussianLikelihood(init_variance=1e-2, eps=1e-9)
            for node in dag.non_roots
        },
        "latent_variable_dims": latent_dims,
        "num_data": dataset.num_data,
        "latent_prior_variance": 1.0,
        "latent_init_std": 0.01,
    }
    if family == "dag_vi":
        model = DAGVIModel(**common)
    else:
        model = DAGSVIModel(
            **common, local_conditioning_jitter=1e-5, marginal_backend="dense",
        )
    return model.to(device=device, dtype=torch.float64)


def nodewise_pretrain(model: DAGVIModel, dataset: DAGDataset, cfg: Config) -> None:
    torch.manual_seed(MODEL_SEED)
    original_requires_grad = {p: p.requires_grad for p in model.parameters()}
    for node in model.dag.topo:
        if node not in model.gp_modules:
            continue
        parameters = [
            *model.gp_modules[node].parameters(),
            *model.likelihoods[node].parameters(),
        ]
        if node in model.latent_layers:
            parameters.extend(model.latent_layers[node].parameters())
        parameters = list(dict.fromkeys(parameters))
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in parameters:
            parameter.requires_grad = True
        optimizer = torch.optim.Adam(parameters, lr=cfg.joint_lr)
        _, observed = dataset.get_observations(node)
        observed_indices = torch.nonzero(observed, as_tuple=False).squeeze(-1)
        for _ in range(cfg.pretrain_steps):
            batch_dataset, batch_indices, scale = dataset, None, 1.0
            if cfg.pretrain_batch_size is not None:
                permutation = torch.randperm(len(observed_indices), device=observed_indices.device)
                batch_indices = observed_indices[
                    permutation[:min(cfg.pretrain_batch_size, len(observed_indices))]
                ]
                batch_mask = torch.zeros_like(observed, dtype=torch.bool)
                batch_mask[batch_indices] = True
                batch_dataset = DAGDataset(
                    roots=dataset.roots, y=dataset.y,
                    mask={**dataset.mask, node: batch_mask},
                )
                scale = float(len(observed_indices)) / float(len(batch_indices))
            optimizer.zero_grad(set_to_none=True)
            ell = _node_likelihood_term_mean_field(
                model, batch_dataset, node=node, num_mc_samples=cfg.num_mc_samples,
            ) * scale
            kl = model.gp_modules[node].kl()
            if node in model.latent_layers:
                if batch_indices is None:
                    kl = kl + model.latent_layers[node].kl()
                else:
                    kl = kl + model.latent_layers[node].kl(indices=batch_indices) * scale
            elbo = ell - kl
            if not torch.isfinite(elbo):
                raise FloatingPointError(f"Non-finite pretraining ELBO for {node}.")
            (-elbo).backward()
            optimizer.step()
    for parameter, requires_grad in original_requires_grad.items():
        parameter.requires_grad = requires_grad
    model.eval()


def build_mogp(
    root_values: torch.Tensor, train_mask: np.ndarray, cfg: Config,
) -> tuple[LMCMultitaskGP, gpytorch.likelihoods.MultitaskGaussianLikelihood]:
    if cfg.protocol == "scalability":
        source = root_values[
            torch.as_tensor(train_mask, dtype=torch.bool, device=root_values.device)
        ]
        base = inducing_grid(source, cfg.num_inducing)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(MODEL_SEED)
        selected = torch.randperm(len(root_values), generator=generator)[:cfg.num_inducing]
        base = root_values.index_select(0, selected.to(root_values.device))
    inducing = base.unsqueeze(0).expand(len(NODES), -1, -1).clone()
    model = LMCMultitaskGP(inducing).to(device=root_values.device, dtype=torch.float64)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=len(NODES), rank=0, has_global_noise=False, has_task_noise=True,
    ).to(device=root_values.device, dtype=torch.float64)
    likelihood.task_noises = torch.full((len(NODES),), 1e-2, device=root_values.device)
    return model, likelihood
