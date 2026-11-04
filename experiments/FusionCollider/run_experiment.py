#!/usr/bin/env python3
"""Compare DAG-VI and DAG-SVI fusion kernels on Ishigami collider mechanisms."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dagdgp import (  
    DAG,
    DAGDataset,
    GaussianLikelihood,
    DAGVIModel,
    MeanFieldSVGP,
    DAGSVIModel,
    StructuredNodeSVGP,
)
from dagdgp.kernels import AdditiveFusionKernel, ProductFusionKernel, RBFKernel  
from dagdgp.utils import inv_softplus  


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SCENARIOS = ("context", "independent")
FUSIONS = ("additive", "product")
INFERENCES = ("dag_vi", "dag_svi")
METRICS = ("test_crps", "test_r2")


@dataclass(frozen=True)
class Config:
    n_a: int = 10
    n_b: int = 10
    parent_noise_std: float = 0.025
    child_noise_std: float = 0.05
    parent_lengthscale: float = 0.42
    parent_residual_scale: float = 0.35
    child_inducing_per_axis: int = 8
    steps: int = 10_000
    learning_rate: float = 5e-3
    kernel_learning_rate: float = 2.5e-3
    num_mc_samples: int = 5
    eval_samples: int = 400


def freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def set_positive_raw(raw: torch.Tensor, value: float, minimum: float = 1e-6) -> None:
    target = torch.as_tensor(
        value - minimum,
        dtype=raw.dtype,
        device=raw.device,
    ).clamp_min(1e-12)
    with torch.no_grad():
        raw.copy_(inv_softplus(target).expand_as(raw))


def sample_gp(kernel: nn.Module, inputs: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    covariance = kernel.K(inputs, inputs)
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + 1e-7 * torch.eye(inputs.shape[0], dtype=inputs.dtype)
    return (torch.linalg.cholesky(covariance) @ torch.randn(inputs.shape[0])).detach()


def odd_gp_residual(inputs: torch.Tensor, cfg: Config, seed: int) -> torch.Tensor:
    kernel = RBFKernel(
        1,
        ard=False,
        init_lengthscale=cfg.parent_lengthscale,
        init_variance=1.0,
    )
    freeze(kernel)
    draw = sample_gp(kernel, inputs[:, None], seed)
    return ((draw - draw.flip(0)) / math.sqrt(2.0)).detach()


def to_ishigami_domain(values: torch.Tensor) -> torch.Tensor:
    return -math.pi + 2.0 * math.pi * (
        (values - values.min()) / (values.max() - values.min())
    )


def empirical_crps(samples: torch.Tensor, targets: torch.Tensor) -> float:
    first_term = (samples - targets[None, :]).abs().mean(0)
    ordered = samples.sort(0).values
    ranks = torch.arange(
        1,
        samples.shape[0] + 1,
        dtype=samples.dtype,
        device=samples.device,
    )[:, None]
    second_term = (
        (2.0 * ranks - samples.shape[0] - 1.0) * ordered
    ).sum(0) / float(samples.shape[0] ** 2)
    return float((first_term - second_term).mean().cpu())


def generate_data(
    cfg: Config,
    scenario: str,
    data_seed: int,
) -> tuple[DAGDataset, dict[str, torch.Tensor]]:
    x_a = torch.linspace(-1.0, 1.0, cfg.n_a)
    x_b = torch.linspace(-1.2, 1.2, cfg.n_b)
    x_a_grid, x_b_grid = torch.meshgrid(x_a, x_b, indexing="ij")

    a = x_a + cfg.parent_residual_scale * odd_gp_residual(
        x_a,
        cfg,
        10_000 + 2 * data_seed,
    )
    b = x_b + cfg.parent_residual_scale * odd_gp_residual(
        x_b,
        cfg,
        10_001 + 2 * data_seed,
    )
    a_ishigami = to_ishigami_domain(a)
    b_ishigami = to_ishigami_domain(b)

    if scenario == "context":
        c = 0.1 * b_ishigami[None, :].pow(4) * torch.sin(a_ishigami)[:, None]
    elif scenario == "independent":
        c = (
            torch.sin(a_ishigami)[:, None]
            + 7.0 * torch.sin(b_ishigami)[None, :].square()
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}.")

    a_grid = a[:, None].expand(cfg.n_a, cfg.n_b).reshape(-1)
    b_grid = b[None, :].expand(cfg.n_a, cfg.n_b).reshape(-1)

    torch.manual_seed(30_000 + data_seed)
    y_a = a_grid + cfg.parent_noise_std * torch.randn(cfg.n_a * cfg.n_b)
    y_b = b_grid + cfg.parent_noise_std * torch.randn(cfg.n_a * cfg.n_b)
    y_c = c.reshape(-1) + cfg.child_noise_std * torch.randn(cfg.n_a * cfg.n_b)

    mask_a = torch.zeros(cfg.n_a * cfg.n_b, dtype=torch.bool)
    mask_a[torch.arange(cfg.n_a) * cfg.n_b] = True
    mask_b = torch.zeros(cfg.n_a * cfg.n_b, dtype=torch.bool)
    mask_b[torch.arange(cfg.n_b)] = True

    if scenario == "context":
        heldout_a = torch.argsort(a)[torch.tensor([2, 7])]
        mask_c_grid = torch.ones((cfg.n_a, cfg.n_b), dtype=torch.bool)
        mask_c_grid[heldout_a, :] = False
    else:
        reference_a = torch.topk(a.abs(), 2, largest=False).indices
        reference_b = torch.topk(b.abs(), 2, largest=False).indices
        mask_c_grid = torch.zeros((cfg.n_a, cfg.n_b), dtype=torch.bool)
        mask_c_grid[reference_a, :] = True
        mask_c_grid[:, reference_b] = True

    dataset = DAGDataset(
        roots={
            "X1": x_a_grid.reshape(-1, 1),
            "X2": x_b_grid.reshape(-1, 1),
        },
        y={"A": y_a, "B": y_b, "C": y_c},
        mask={"A": mask_a, "B": mask_b, "C": mask_c_grid.reshape(-1)},
    )
    truth = {
        "x_a": x_a,
        "x_b": x_b,
        "A": a,
        "B": b,
        "C": c,
        "y_C": y_c.reshape(cfg.n_a, cfg.n_b),
        "mask_C": mask_c_grid,
    }
    return dataset, truth


def make_child_kernel(fusion: str, truth: dict[str, torch.Tensor]) -> nn.Module:
    initial_lengthscales = (
        max(0.2, 0.75 * float(truth["A"].std())),
        max(0.2, 0.75 * float(truth["B"].std())),
    )
    if fusion == "additive":
        kernel = AdditiveFusionKernel(
            ["A", "B"],
            [1, 1],
            base_kernel="rbf",
            ard=False,
            trainable_scales=True,
        )
        components = [component.base_kernel for component in kernel.kernels]
    elif fusion == "product":
        kernel = ProductFusionKernel(
            ["A", "B"],
            [1, 1],
            base_kernel="rbf",
            ard=False,
            trainable_scale=True,
        )
        components = list(kernel.kernels)
    else:
        raise ValueError(f"Unknown fusion: {fusion}.")

    for component, lengthscale in zip(components, initial_lengthscales):
        component.set_lengthscale(lengthscale)
        component.set_variance(1.0)
        component.raw_variance.requires_grad_(False)
    return kernel


def build_model(
    cfg: Config,
    inference: str,
    fusion: str,
    truth: dict[str, torch.Tensor],
    dataset: DAGDataset,
    device: str,
) -> DAGVIModel | DAGSVIModel:
    if inference not in INFERENCES:
        raise ValueError(f"Unknown inference family: {inference}.")
    dag = DAG({"X1": [], "X2": [], "A": ["X1"], "B": ["X2"], "C": ["A", "B"]})

    a_sd = float(truth["A"].std())
    b_sd = float(truth["B"].std())
    z_a, z_b = torch.meshgrid(
        torch.linspace(
            float(truth["A"].min() - 0.4 * a_sd),
            float(truth["A"].max() + 0.4 * a_sd),
            cfg.child_inducing_per_axis,
        ),
        torch.linspace(
            float(truth["B"].min() - 0.4 * b_sd),
            float(truth["B"].max() + 0.4 * b_sd),
            cfg.child_inducing_per_axis,
        ),
        indexing="ij",
    )
    child_kernel = make_child_kernel(fusion, truth)
    module_class = MeanFieldSVGP if inference == "dag_vi" else StructuredNodeSVGP
    common = {
        "dag": dag,
        "root_dims": {"X1": 1, "X2": 1},
        "gp_modules": {
            "A": module_class(
                RBFKernel(1, ard=False, init_lengthscale=cfg.parent_lengthscale),
                truth["x_a"][:, None],
            ),
            "B": module_class(
                RBFKernel(1, ard=False, init_lengthscale=cfg.parent_lengthscale),
                truth["x_b"][:, None],
            ),
            "C": module_class(
                child_kernel,
                torch.stack([z_a.reshape(-1), z_b.reshape(-1)], dim=-1),
            ),
        },
        "likelihoods": {
            "A": GaussianLikelihood(cfg.parent_noise_std**2),
            "B": GaussianLikelihood(cfg.parent_noise_std**2),
            "C": GaussianLikelihood(cfg.child_noise_std**2),
        },
    }
    if inference == "dag_vi":
        model = DAGVIModel(**common)
    else:
        model = DAGSVIModel(
            **common,
            observed_nodes=("A", "B", "C"),
            marginal_backend="dense",
        )

    freeze(model.gp_modules["A"].kernel)
    freeze(model.gp_modules["B"].kernel)
    for node, likelihood in model.likelihoods.items():
        variance = cfg.child_noise_std**2 if node == "C" else cfg.parent_noise_std**2
        likelihood.set_variance(variance)
        freeze(likelihood)

    observed_c = dataset.y["C"][dataset.mask["C"]]
    child_mean = observed_c.mean()
    child_variance = (observed_c - child_mean).square().mean().clamp_min(1e-3)
    with torch.no_grad():
        model.gp_modules["C"].mean_offset.copy_(child_mean)
    if fusion == "additive":
        for component in child_kernel.kernels:
            set_positive_raw(component.raw_scale, float(child_variance / 2.0))
    else:
        set_positive_raw(child_kernel.raw_scale, float(child_variance))

    return model.to(device=device, dtype=torch.float64)


def train(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    cfg: Config,
    data_seed: int,
    fit_seed: int,
) -> None:
    torch.manual_seed(40_000 + 100 * data_seed + fit_seed)
    reference = next(model.parameters())
    data = dataset.to(device=reference.device, dtype=reference.dtype)

    kernel_parameters = [
        parameter
        for parameter in model.gp_modules["C"].kernel.parameters()
        if parameter.requires_grad
    ]
    kernel_ids = {id(parameter) for parameter in kernel_parameters}
    remaining_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in kernel_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": remaining_parameters, "lr": cfg.learning_rate},
            {"params": kernel_parameters, "lr": cfg.kernel_learning_rate},
        ]
    )

    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(data, num_mc_samples=cfg.num_mc_samples)
        (-elbo).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
    model.eval()


@torch.no_grad()
def evaluate(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    truth: dict[str, torch.Tensor],
    cfg: Config,
    data_seed: int,
    fit_seed: int,
) -> dict[str, float]:
    reference = next(model.parameters())
    evaluation_data = DAGDataset(
        roots=dataset.roots,
        y={},
    ).to(reference.device, reference.dtype)
    torch.manual_seed(50_000 + 100 * data_seed + fit_seed)
    samples = model.sample_latents(
        evaluation_data,
        num_samples=cfg.eval_samples,
    )["C"].squeeze(-1)
    samples = samples.reshape(cfg.eval_samples, cfg.n_a, cfg.n_b)

    test_mask = ~truth["mask_C"].to(samples.device)
    test_samples = samples[:, test_mask]
    latent_targets = truth["C"].to(samples.device)[test_mask]
    observed_targets = truth["y_C"].to(samples.device)[test_mask]

    torch.manual_seed(60_000 + 100 * data_seed + fit_seed)
    predictive_samples = (
        test_samples + cfg.child_noise_std * torch.randn_like(test_samples)
    )
    posterior_mean = test_samples.mean(0)
    residual_sum = (latent_targets - posterior_mean).square().sum()
    total_sum = (latent_targets - latent_targets.mean()).square().sum()
    return {
        "test_crps": empirical_crps(predictive_samples, observed_targets),
        "test_r2": float((1.0 - residual_sum / total_sum).cpu()),
    }


def run_fit(
    cfg: Config,
    inference: str,
    scenario: str,
    fusion: str,
    data_seed: int,
    fit_seed: int,
    device: str,
) -> dict[str, int | float | str]:
    dataset, truth = generate_data(cfg, scenario, data_seed)
    model = build_model(cfg, inference, fusion, truth, dataset, device)
    train(model, dataset, cfg, data_seed, fit_seed)
    metrics = evaluate(model, dataset, truth, cfg, data_seed, fit_seed)
    return {
        "inference": inference,
        "scenario": scenario,
        "fusion": fusion,
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        **metrics,
    }


def run_spec(
    spec: tuple[Config, str, str, str, int, int, str],
) -> dict[str, int | float | str]:
    return run_fit(*spec)


def sample_sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate(rows: list[dict[str, int | float | str]]) -> list[dict[str, float | str]]:
    table = []
    for inference in INFERENCES:
        for scenario in SCENARIOS:
            for fusion in FUSIONS:
                per_dataset: dict[int, dict[str, float]] = {}
                for data_seed in sorted({int(row["data_seed"]) for row in rows}):
                    subset = [
                        row
                        for row in rows
                        if row.get("inference", "dag_svi") == inference
                        and row["scenario"] == scenario
                        and row["fusion"] == fusion
                        and row["data_seed"] == data_seed
                    ]
                    if not subset:
                        continue
                    per_dataset[data_seed] = {
                        metric: sum(float(row[metric]) for row in subset) / len(subset)
                        for metric in METRICS
                    }
                if not per_dataset:
                    continue
                if len(per_dataset) < 2:
                    raise ValueError("At least two dataset seeds are required for a sample SD.")
                summary: dict[str, float | str] = {
                    "inference": inference,
                    "scenario": scenario,
                    "fusion": fusion,
                }
                for metric in METRICS:
                    values = [per_dataset[seed][metric] for seed in sorted(per_dataset)]
                    summary[f"{metric}_mean"] = sum(values) / len(values)
                    summary[f"{metric}_sd"] = sample_sd(values)
                table.append(summary)
    return table


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, table: list[dict[str, float | str]]) -> None:
    labels = {
        "dag_vi": "DAG-VI",
        "dag_svi": "DAG-SVI",
        "context": "Context-specific interaction",
        "independent": "Independent-additive",
        "additive": "Additive",
        "product": "Product",
    }
    lines = [
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        r"Inference & Mechanism & Fusion kernel & CRPS $\downarrow$ & $R^2$ $\uparrow$ \\",
        r"\midrule",
    ]
    for row in table:
        lines.append(
            f"{labels[str(row['inference'])]} & "
            f"{labels[str(row['scenario'])]} & {labels[str(row['fusion'])]} & "
            f"{float(row['test_crps_mean']):.3f} $\\pm$ "
            f"{float(row['test_crps_sd']):.3f} & "
            f"{float(row['test_r2_mean']):.4f} $\\pm$ "
            f"{float(row['test_r2_sd']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reproduction",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-samples", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--inferences",
        nargs="+",
        choices=INFERENCES,
        default=list(INFERENCES),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent fits to run concurrently.",
    )
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    cfg = Config(steps=args.steps, eval_samples=args.eval_samples)
    specs = []
    for inference in args.inferences:
        for scenario in SCENARIOS:
            for fusion in FUSIONS:
                for data_seed in args.data_seeds:
                    for fit_seed in args.fit_seeds:
                        specs.append(
                            (
                                cfg,
                                inference,
                                scenario,
                                fusion,
                                data_seed,
                                fit_seed,
                                args.device,
                            )
                        )

    if args.jobs == 1:
        row_iterator = map(run_spec, specs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=multiprocessing.get_context("spawn"),
        )
        row_iterator = executor.map(run_spec, specs)

    rows = []
    try:
        for row in row_iterator:
            rows.append(row)
            print(
                row["inference"],
                row["scenario"],
                row["fusion"],
                row["data_seed"],
                row["fit_seed"],
                row["test_crps"],
                row["test_r2"],
                flush=True,
            )
    finally:
        if args.jobs != 1:
            executor.shutdown()

    table = aggregate(rows)
    write_csv(args.output_dir / "per_fit.csv", rows)
    write_csv(args.output_dir / "table.csv", table)
    write_latex(args.output_dir / "table.tex", table)


if __name__ == "__main__":
    main()
