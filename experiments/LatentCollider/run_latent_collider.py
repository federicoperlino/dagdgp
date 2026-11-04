#!/usr/bin/env python3
"""Run the latent collider experiment and write the plot data.

The experiment uses the collider DAG

    X1 -> A -> C <- B <- X2

and writes `summary.json` together with `figure_seed2_plot_data.pt` into this
directory.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




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


OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
PLOT_DATA_PATH = OUTPUT_DIR / "figure_seed2_plot_data.pt"

SUMMARY_SEEDS = 3
FIGURE_SEED = 2
STEPS = 3000
POSTERIOR_SAMPLES_EVAL = 200
POSTERIOR_SAMPLES_FIG = 1500
CHILD_LENGTHSCALE = 0.60
NOISE_STD = 0.015
M_ROOT = 15
CHILD_GRID = 7
NUM_MC_SAMPLES = 4
DEVICE = torch.device(os.environ.get("DAGDGP_DEVICE", "cpu"))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class SumProjectionKernel(nn.Module):
    """RBF kernel on the fused latent input s = a + b."""

    def __init__(self, lengthscale: float = 0.8, variance: float = 1.0) -> None:
        super().__init__()
        self.input_dim = 2
        self.base = RBFKernel(1, ard=False, init_lengthscale=lengthscale, init_variance=variance)

    def _u(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=1, keepdim=True)

    def K(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:  
        return self.base.K(self._u(X1), self._u(X2))

    def Kdiag(self, X: torch.Tensor) -> torch.Tensor:  
        return self.base.Kdiag(self._u(X))


@dataclass(frozen=True)
class ExperimentConfig:
    n1: int = 15
    n2: int = 11
    x1_min: float = -1.0
    x1_max: float = 1.0
    x2_min: float = -1.2
    x2_max: float = 1.2
    x1_dense: int = 181
    x2_dense: int = 141
    root_lengthscale: float = 0.4
    root_variance: float = 1.0
    child_lengthscale: float = 0.60
    child_variance: float = 1.0
    noise_std: float = 0.015
    m_root: int = 15
    child_grid_m1: int = 7
    child_grid_m2: int = 7
    child_latent_min: float = -2.5
    child_latent_max: float = 2.5
    steps: int = 3000
    learning_rate: float = 0.01
    num_mc_samples: int = 4
    posterior_samples: int = 1500
    num_show_curves: int = 24
    num_anchor_cases: int = 8
    min_anchor_manhattan: int = 3
    figure_seed: int = 2
    fit_seed: int = 2
    child_input_connection: bool = False
    child_kernel_mode: str = "sum"


@dataclass(frozen=True)
class TruthData:
    x1_train: torch.Tensor
    x2_train: torch.Tensor
    x1_dense: torch.Tensor
    x2_dense: torch.Tensor
    a_train_unique: torch.Tensor
    b_train_unique: torch.Tensor
    a_dense_unique: torch.Tensor
    b_dense_unique: torch.Tensor
    a_train: torch.Tensor
    b_train: torch.Tensor
    c_train: torch.Tensor
    y_train: torch.Tensor
    c_dense_truth_mean: torch.Tensor


@dataclass(frozen=True)
class FitLoopConfig:
    steps: int
    joint_lr: float
    batch_size: int
    num_mc_samples: int
    likelihood_lr_scale: float
    posterior_lr_scale: float = 1.0
    seed: int = 0


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def sample_gp(kernel: nn.Module, X: torch.Tensor, *, jitter: float = 1e-6) -> torch.Tensor:
    K = kernel.K(X, X)
    K = 0.5 * (K + K.transpose(0, 1))
    K = K + jitter * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
    L = torch.linalg.cholesky(K)
    return (L @ torch.randn(X.shape[0], dtype=X.dtype, device=X.device)).detach()


def mesh(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.meshgrid(x1, x2, indexing="ij")


def joint_sample_on_train_and_dense(
    kernel: nn.Module,
    x_train: torch.Tensor,
    x_dense: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_all = torch.cat([x_train.unsqueeze(-1), x_dense.unsqueeze(-1)], dim=0)
    draw = sample_gp(kernel, X_all)
    n_train = x_train.shape[0]
    return draw[:n_train].detach(), draw[n_train:].detach()


def kernel_ctor_from_config(config: ExperimentConfig) -> Callable[[], nn.Module]:
    return lambda: SumProjectionKernel(lengthscale=config.child_lengthscale, variance=config.child_variance)


def generate_dataset(
    config: ExperimentConfig,
    *,
    seed: int,
    kernel_ctor: Callable[[], nn.Module],
) -> tuple[DAGDataset, TruthData]:
    torch.manual_seed(seed)

    x1_train = torch.linspace(config.x1_min, config.x1_max, config.n1)
    x2_train = torch.linspace(config.x2_min, config.x2_max, config.n2)
    x1_dense = torch.linspace(config.x1_min, config.x1_max, config.x1_dense)
    x2_dense = torch.linspace(config.x2_min, config.x2_max, config.x2_dense)

    root_kernel = RBFKernel(1, ard=False, init_lengthscale=config.root_lengthscale, init_variance=config.root_variance)
    freeze_module(root_kernel)

    a_train_unique, a_dense_unique = joint_sample_on_train_and_dense(root_kernel, x1_train, x1_dense)
    b_train_unique, b_dense_unique = joint_sample_on_train_and_dense(root_kernel, x2_train, x2_dense)

    A_train = a_train_unique[:, None].expand(config.n1, config.n2).reshape(-1).detach()
    B_train = b_train_unique[None, :].expand(config.n1, config.n2).reshape(-1).detach()
    X1g, X2g = mesh(x1_train, x2_train)

    child_train_inputs = torch.stack([A_train, B_train], dim=-1)
    child_kernel = kernel_ctor()
    freeze_module(child_kernel)
    c_train = sample_gp(child_kernel, child_train_inputs).detach()
    y_train = (c_train + config.noise_std * torch.randn_like(c_train)).detach()

    dataset = DAGDataset(
        roots={"X1": X1g.reshape(-1, 1), "X2": X2g.reshape(-1, 1)},
        y={"C": y_train},
    )

    A_dense = a_dense_unique[:, None].expand(config.x1_dense, config.x2_dense).reshape(-1)
    B_dense = b_dense_unique[None, :].expand(config.x1_dense, config.x2_dense).reshape(-1)
    child_dense_inputs = torch.stack([A_dense, B_dense], dim=-1)
    Ktt = child_kernel.K(child_train_inputs, child_train_inputs)
    Ktt = 0.5 * (Ktt + Ktt.transpose(0, 1)) + 1e-6 * torch.eye(Ktt.shape[0], dtype=Ktt.dtype)
    Kdt = child_kernel.K(child_dense_inputs, child_train_inputs)
    alpha = torch.linalg.solve(Ktt, c_train)
    c_dense_truth_mean = (Kdt @ alpha).reshape(config.x1_dense, config.x2_dense).detach()

    truth = TruthData(
        x1_train=x1_train.detach(),
        x2_train=x2_train.detach(),
        x1_dense=x1_dense.detach(),
        x2_dense=x2_dense.detach(),
        a_train_unique=a_train_unique.detach(),
        b_train_unique=b_train_unique.detach(),
        a_dense_unique=a_dense_unique.detach(),
        b_dense_unique=b_dense_unique.detach(),
        a_train=A_train.detach(),
        b_train=B_train.detach(),
        c_train=c_train.detach(),
        y_train=y_train.detach(),
        c_dense_truth_mean=c_dense_truth_mean,
    )
    return dataset, truth


def build_models(
    config: ExperimentConfig,
    *,
    kernel_ctor: Callable[[], nn.Module],
) -> tuple[DAGVIModel, DAGSVIModel]:
    dag = DAG({"X1": [], "X2": [], "A": ["X1"], "B": ["X2"], "C": ["A", "B"]})

    Z_A = torch.linspace(config.x1_min, config.x1_max, config.m_root).unsqueeze(-1)
    Z_B = torch.linspace(config.x2_min, config.x2_max, config.m_root).unsqueeze(-1)
    z1, z2 = torch.meshgrid(
        torch.linspace(config.child_latent_min, config.child_latent_max, config.child_grid_m1),
        torch.linspace(config.child_latent_min, config.child_latent_max, config.child_grid_m2),
        indexing="ij",
    )
    Z_C = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=-1)

    mf = DAGVIModel(
        dag=dag,
        root_dims={"X1": 1, "X2": 1},
        gp_modules={
            "A": MeanFieldSVGP(
                RBFKernel(1, ard=False, init_lengthscale=config.root_lengthscale, init_variance=config.root_variance),
                Z_A,
            ),
            "B": MeanFieldSVGP(
                RBFKernel(1, ard=False, init_lengthscale=config.root_lengthscale, init_variance=config.root_variance),
                Z_B,
            ),
            "C": MeanFieldSVGP(kernel_ctor(), Z_C),
        },
        likelihoods={"C": GaussianLikelihood(init_variance=config.noise_std**2)},
    )

    structured = DAGSVIModel(
        dag=dag,
        root_dims={"X1": 1, "X2": 1},
        gp_modules={
            "A": StructuredNodeSVGP(
                RBFKernel(1, ard=False, init_lengthscale=config.root_lengthscale, init_variance=config.root_variance),
                Z_A,
            ),
            "B": StructuredNodeSVGP(
                RBFKernel(1, ard=False, init_lengthscale=config.root_lengthscale, init_variance=config.root_variance),
                Z_B,
            ),
            "C": StructuredNodeSVGP(kernel_ctor(), Z_C),
        },
        likelihoods={"C": GaussianLikelihood(init_variance=config.noise_std**2)},
        observed_nodes=("C",),
    )

    for model in (mf, structured):
        for gp in model.gp_modules.values():
            freeze_module(gp.kernel)
            gp.Z.requires_grad_(False)
        for lk in model.likelihoods.values():
            lk.set_variance(config.noise_std**2)
            freeze_module(lk)
        model.to(device=DEVICE, dtype=torch.float64)

    return mf, structured


def fit_dag_vi_loop(model: DAGVIModel, dataset: DAGDataset, config: FitLoopConfig) -> list[float]:
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
    other_params = [param for param in model.parameters() if param.requires_grad and id(param) not in likelihood_ids]

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

    for _ in range(int(config.steps)):
        indices = None if batch_size >= num_data else torch.randperm(num_data, device=reference.device)[:batch_size]
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(data, indices=indices, num_mc_samples=int(config.num_mc_samples))
        history.append(float(elbo.detach().cpu()))
        (-elbo).backward()
        optimizer.step()

    model.eval()
    return history


def fit_dag_svi_loop(model: DAGSVIModel, dataset: DAGDataset, config: FitLoopConfig) -> list[float]:
    torch.manual_seed(int(config.seed))
    reference = next(model.parameters())
    data = dataset.to(device=reference.device, dtype=reference.dtype)

    posterior_params = [param for param in model.posterior.parameters() if param.requires_grad]
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
        if param.requires_grad and id(param) not in likelihood_ids and id(param) not in posterior_ids
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

    for _ in range(int(config.steps)):
        indices = None if batch_size >= num_data else torch.randperm(num_data, device=reference.device)[:batch_size]
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(data, indices=indices, num_mc_samples=int(config.num_mc_samples))
        history.append(float(elbo.detach().cpu()))
        (-elbo).backward()
        optimizer.step()

    model.eval()
    return history


def fit_both(
    config: ExperimentConfig,
    dataset: DAGDataset,
    *,
    kernel_ctor: Callable[[], nn.Module],
) -> tuple[DAGVIModel, DAGSVIModel]:
    mf, structured = build_models(config, kernel_ctor=kernel_ctor)
    fit_config = FitLoopConfig(
        steps=config.steps,
        joint_lr=config.learning_rate,
        batch_size=dataset.num_data,
        num_mc_samples=config.num_mc_samples,
        likelihood_lr_scale=0.0,
        posterior_lr_scale=1.0,
        seed=config.fit_seed,
    )
    fit_dag_vi_loop(mf, dataset, fit_config)
    fit_dag_svi_loop(structured, dataset, fit_config)
    # Move both models to CPU for evaluation.
    mf.to(device="cpu")
    structured.to(device="cpu")
    return mf, structured


@torch.no_grad()
def sample_all_latents(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    *,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    return model.sample_latents(dataset, num_samples=num_samples)


@torch.no_grad()
def line_samples_for_A(
    model: DAGVIModel | DAGSVIModel,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = torch.linspace(config.x1_min, config.x1_max, config.x1_dense)
    x2 = torch.zeros_like(x1)
    data = DAGDataset(roots={"X1": x1.unsqueeze(-1), "X2": x2.unsqueeze(-1)}, y={})
    samples = sample_all_latents(model, data, num_samples=config.posterior_samples)
    return x1.detach().cpu().numpy(), samples["A"].squeeze(-1).detach().cpu().numpy()


@torch.no_grad()
def line_samples_for_B(
    model: DAGVIModel | DAGSVIModel,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x2 = torch.linspace(config.x2_min, config.x2_max, config.x2_dense)
    x1 = torch.zeros_like(x2)
    data = DAGDataset(roots={"X1": x1.unsqueeze(-1), "X2": x2.unsqueeze(-1)}, y={})
    samples = sample_all_latents(model, data, num_samples=config.posterior_samples)
    return x2.detach().cpu().numpy(), samples["B"].squeeze(-1).detach().cpu().numpy()


def global_metrics(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    truth: TruthData,
    *,
    num_samples: int,
    noise_std: float,
) -> dict[str, float]:
    samples = sample_all_latents(model, dataset, num_samples=num_samples)
    C = samples["C"].squeeze(-1)
    A = samples["A"].squeeze(-1)
    B = samples["B"].squeeze(-1)

    c_mean = C.mean(dim=0)
    rmse = float(torch.mean((c_mean - truth.c_train) ** 2).sqrt())

    sigma2 = noise_std**2
    log_probs = -0.5 * math.log(2.0 * math.pi * sigma2) - 0.5 * (truth.y_train.unsqueeze(0) - C) ** 2 / sigma2
    nll = float(-(torch.logsumexp(log_probs, dim=0) - math.log(C.shape[0])).mean())

    A0 = A - A.mean(dim=0, keepdim=True)
    B0 = B - B.mean(dim=0, keepdim=True)
    corr = (A0 * B0).mean(dim=0) / (A0.std(dim=0, unbiased=False) * B0.std(dim=0, unbiased=False) + 1e-12)

    return {
        "rmse_c": rmse,
        "nll_c": nll,
        "avg_abs_corr_ab": float(corr.abs().mean()),
        "var_a": float(A.var(dim=0, unbiased=False).mean()),
        "var_b": float(B.var(dim=0, unbiased=False).mean()),
    }


def make_mild_test_mask(config: ExperimentConfig) -> torch.Tensor:
    """Select interior grid points for interpolation testing using a fixed random seed."""
    mask = torch.zeros(config.n1 * config.n2, dtype=torch.bool)
    interior = [i * config.n2 + j for i in range(1, config.n1 - 1) for j in range(1, config.n2 - 1)]
    generator = torch.Generator().manual_seed(123)
    choice = torch.randperm(len(interior), generator=generator)[:29]
    chosen = torch.tensor(interior, dtype=torch.long).index_select(0, choice)
    mask[chosen] = True
    return mask


def subset_dataset(dataset: DAGDataset, indices: torch.Tensor) -> DAGDataset:
    return DAGDataset(
        roots={name: value.index_select(0, indices) for name, value in dataset.roots.items()},
        y={name: value.index_select(0, indices) for name, value in dataset.y.items()},
    )


def empirical_crps(samples: torch.Tensor, targets: torch.Tensor) -> float:
    term1 = (samples - targets.unsqueeze(0)).abs().mean(dim=0)
    pairwise = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean(dim=(0, 1))
    return float((term1 - 0.5 * pairwise).mean())


def heldout_metrics(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    truth: TruthData,
    test_mask: torch.Tensor,
    *,
    num_samples: int,
) -> dict[str, float]:
    test_indices = torch.nonzero(test_mask, as_tuple=False).squeeze(-1)
    test_dataset = subset_dataset(dataset, test_indices)
    samples = sample_all_latents(model, test_dataset, num_samples=num_samples)
    c_samples = samples["C"].squeeze(-1)
    c_mean = c_samples.mean(dim=0)
    y_test = truth.y_train.index_select(0, test_indices)
    c_test = truth.c_train.index_select(0, test_indices)
    return {
        "num_test_points": int(test_indices.numel()),
        "rmse_y_test": float(torch.mean((c_mean - y_test) ** 2).sqrt()),
        "rmse_c_test": float(torch.mean((c_mean - c_test) ** 2).sqrt()),
        "crps_y_test": empirical_crps(c_samples, y_test),
    }


def select_anchor_cases_from_observed_c(truth: TruthData, config: ExperimentConfig) -> tuple[list[tuple[int, int]], np.ndarray]:
    Y = truth.y_train.reshape(config.n1, config.n2).detach().cpu().numpy()
    gx = np.zeros_like(Y)
    gy = np.zeros_like(Y)
    gx[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / 2.0
    gy[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / 2.0
    score = np.minimum(np.abs(gx), np.abs(gy)) * np.sqrt(gx**2 + gy**2)
    score[[0, -1], :] = -np.inf
    score[:, [0, -1]] = -np.inf

    coords: list[tuple[int, int]] = []
    for flat_idx in np.argsort(score.reshape(-1))[::-1]:
        i, j = divmod(int(flat_idx), config.n2)
        if not np.isfinite(score[i, j]):
            continue
        if all(abs(i - u) + abs(j - v) >= config.min_anchor_manhattan for u, v in coords):
            coords.append((i, j))
            if len(coords) >= config.num_anchor_cases:
                break
    return coords, score


def selected_case_corr(
    samples: dict[str, torch.Tensor],
    coords: list[tuple[int, int]],
    config: ExperimentConfig,
) -> list[float]:
    A = samples["A"].squeeze(-1).detach().cpu().numpy()
    B = samples["B"].squeeze(-1).detach().cpu().numpy()
    corrs: list[float] = []
    for i, j in coords:
        idx = i * config.n2 + j
        corrs.append(float(np.corrcoef(A[:, idx], B[:, idx])[0, 1]))
    return corrs


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    metrics = ["rmse_c", "nll_c", "avg_abs_corr_ab", "var_a", "var_b"]
    out: dict[str, object] = {"num_seeds": len(records)}
    for method in ("mf", "structured"):
        out[method] = {}
        for metric in metrics:
            vals = np.asarray([record[method][metric] for record in records], dtype=float)
            out[method][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        out[method]["heldout_test"] = {}
        for metric in ("rmse_y_test", "rmse_c_test", "crps_y_test"):
            vals = np.asarray([record[method]["heldout_test"][metric] for record in records], dtype=float)
            out[method]["heldout_test"][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
    out["advantages"] = {
        "structured_minus_mf_rmse_c": out["structured"]["rmse_c"]["mean"] - out["mf"]["rmse_c"]["mean"],
        "structured_minus_mf_nll_c": out["structured"]["nll_c"]["mean"] - out["mf"]["nll_c"]["mean"],
        "structured_minus_mf_avg_abs_corr_ab": out["structured"]["avg_abs_corr_ab"]["mean"] - out["mf"]["avg_abs_corr_ab"]["mean"],
        "structured_minus_mf_rmse_y_test": out["structured"]["heldout_test"]["rmse_y_test"]["mean"]
        - out["mf"]["heldout_test"]["rmse_y_test"]["mean"],
        "structured_minus_mf_crps_y_test": out["structured"]["heldout_test"]["crps_y_test"]["mean"]
        - out["mf"]["heldout_test"]["crps_y_test"]["mean"],
    }
    return out


def run_one_seed(seed: int, *, keep_plot_data: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    config = ExperimentConfig(
        steps=STEPS,
        figure_seed=seed,
        fit_seed=seed,
        child_lengthscale=CHILD_LENGTHSCALE,
        noise_std=NOISE_STD,
        m_root=M_ROOT,
        child_grid_m1=CHILD_GRID,
        child_grid_m2=CHILD_GRID,
        num_mc_samples=NUM_MC_SAMPLES,
        posterior_samples=POSTERIOR_SAMPLES_FIG,
        child_kernel_mode="sum",
        child_input_connection=False,
    )
    kernel_ctor = kernel_ctor_from_config(config)
    dataset, truth = generate_dataset(config, seed=seed, kernel_ctor=kernel_ctor)
    test_mask = make_mild_test_mask(config)
    train_indices = torch.nonzero(~test_mask, as_tuple=False).squeeze(-1)
    train_dataset = subset_dataset(dataset, train_indices)
    mf_model, structured_model = fit_both(config, train_dataset, kernel_ctor=kernel_ctor)

    mf_metrics = global_metrics(
        mf_model,
        dataset,
        truth,
        num_samples=POSTERIOR_SAMPLES_EVAL,
        noise_std=config.noise_std,
    )
    structured_metrics = global_metrics(
        structured_model,
        dataset,
        truth,
        num_samples=POSTERIOR_SAMPLES_EVAL,
        noise_std=config.noise_std,
    )
    mf_heldout = heldout_metrics(
        mf_model,
        dataset,
        truth,
        test_mask,
        num_samples=POSTERIOR_SAMPLES_EVAL,
    )
    structured_heldout = heldout_metrics(
        structured_model,
        dataset,
        truth,
        test_mask,
        num_samples=POSTERIOR_SAMPLES_EVAL,
    )

    record = {
        "seed": seed,
        "mf": {**mf_metrics, "heldout_test": mf_heldout},
        "structured": {**structured_metrics, "heldout_test": structured_heldout},
    }
    if not keep_plot_data:
        return record, None

    mf_samples_eval = sample_all_latents(mf_model, dataset, num_samples=POSTERIOR_SAMPLES_EVAL)
    structured_samples_eval = sample_all_latents(structured_model, dataset, num_samples=POSTERIOR_SAMPLES_EVAL)
    mf_samples_fig = sample_all_latents(mf_model, dataset, num_samples=POSTERIOR_SAMPLES_FIG)
    structured_samples_fig = sample_all_latents(structured_model, dataset, num_samples=POSTERIOR_SAMPLES_FIG)
    x1_dense, A_mf = line_samples_for_A(mf_model, config)
    _, A_structured = line_samples_for_A(structured_model, config)
    x2_dense, B_mf = line_samples_for_B(mf_model, config)
    _, B_structured = line_samples_for_B(structured_model, config)

    plot_data = {
        "config": config,
        "truth": truth,
        "mf_metrics": mf_metrics,
        "structured_metrics": structured_metrics,
        "mf_heldout_metrics": mf_heldout,
        "structured_heldout_metrics": structured_heldout,
        "mf_samples_eval": mf_samples_eval,
        "structured_samples_eval": structured_samples_eval,
        "mf_samples_fig": mf_samples_fig,
        "structured_samples_fig": structured_samples_fig,
        "test_mask": test_mask,
        "line_draws": {
            "x1_dense": x1_dense,
            "x2_dense": x2_dense,
            "A_mf": A_mf,
            "A_structured": A_structured,
            "B_mf": B_mf,
            "B_structured": B_structured,
        },
    }
    return record, plot_data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training on {DEVICE}")

    records: list[dict[str, object]] = []
    figure_plot_data = None
    for seed in range(SUMMARY_SEEDS):
        record, plot_data = run_one_seed(seed, keep_plot_data=(seed == FIGURE_SEED))
        records.append(record)
        if plot_data is not None:
            figure_plot_data = plot_data

    if figure_plot_data is None:
        raise RuntimeError("Failed to build figure plot data.")

    coords, _ = select_anchor_cases_from_observed_c(figure_plot_data["truth"], figure_plot_data["config"])
    figure_seed_metrics = next(record for record in records if int(record["seed"]) == FIGURE_SEED)
    figure_seed_mf_corr = selected_case_corr(figure_plot_data["mf_samples_fig"], coords, figure_plot_data["config"])
    figure_seed_structured_corr = selected_case_corr(
        figure_plot_data["structured_samples_fig"],
        coords,
        figure_plot_data["config"],
    )

    summary = {
        "config": asdict(figure_plot_data["config"]),
        "observation_regime": "Mild interior hold-out test set for C; full grid retained for visualisation.",
        "per_seed": records,
        "aggregate": aggregate(records),
        "figure_seed": FIGURE_SEED,
        "figure_seed_metrics": {
            "mf": figure_seed_metrics["mf"],
            "structured": figure_seed_metrics["structured"],
        },
        "figure_seed_coords": [list(item) for item in coords],
        "figure_seed_mf_per_case_corr": figure_seed_mf_corr,
        "figure_seed_structured_per_case_corr": figure_seed_structured_corr,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(figure_plot_data, PLOT_DATA_PATH)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {PLOT_DATA_PATH}")


if __name__ == "__main__":
    main()
