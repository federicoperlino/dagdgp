#!/usr/bin/env python3
"""Reproduce the DAG-DGP misspecification RMSE and CRPS figures."""

from __future__ import annotations

import argparse
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagdgp import (  
    DAG,
    DAGDataset,
    GaussianLikelihood,
    MeanFieldSVGP,
    DAGVIModel,
    DAGSVIModel,
    StructuredNodeSVGP,
)
from dagdgp.kernels import AdditiveFusionKernel, RBFKernel  
from dagdgp.utils import inv_softplus  


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dag_misspecification_RMSE.pdf"


@dataclass(frozen=True)
class ExperimentConfig:
    num_data: int = 120
    function_seed: int = 0
    num_replicates: int = 5
    parent_observation_fraction: float = 0.55
    child_train_fraction: float = 0.75
    parent_noise_std: float = 0.05
    child_noise_std: float = 0.08
    contribution_A: float = 0.75
    contribution_B: float = 0.35
    contribution_D: float = 0.12
    parent_dgp_lengthscale: float = 0.70
    parent_dgp_variance: float = 1.0
    child_dgp_lengthscale: float = 0.80
    child_dgp_component_variance: float = 1.0
    num_inducing_parent: int = 10
    num_inducing_child: int = 24
    steps: int = 5000
    learning_rate: float = 0.01
    likelihood_lr_scale: float = 0.5
    posterior_lr_scale: float = 1.0
    num_mc_samples: int = 4
    evaluation_samples: int = 600
    branch_lengthscale_init: float = 0.70
    child_lengthscale_init: float = 0.80
    child_component_scale_init: float = 0.50
    gradient_clip_norm: float = 100.0
    device: str = "cpu"


@dataclass(frozen=True)
class GraphVariant:
    key: str
    child_parents: tuple[str, ...]


VARIANTS = (
    GraphVariant("correct", ("A", "B", "D")),
    GraphVariant("spurious", ("A", "B", "D", "E")),
    GraphVariant("minus_D", ("A", "B")),
    GraphVariant("minus_BD", ("A",)),
    GraphVariant("minus_A", ("B", "D")),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}
PARENT_COLUMN = {"A": 0, "B": 1, "D": 2, "E": 3}


@dataclass(frozen=True)
class SyntheticTruth:
    observations: dict[str, torch.Tensor]
    child_test_mask: torch.Tensor


def _normalise_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _sample_gp(
    kernel: RBFKernel,
    inputs: torch.Tensor,
    *,
    generator: torch.Generator,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Draw an exact zero-mean GP sample at ``inputs``."""
    with torch.no_grad():
        covariance = kernel.K(inputs, inputs)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        covariance = covariance + jitter * torch.eye(inputs.shape[0], dtype=inputs.dtype)
        chol = torch.linalg.cholesky(covariance)
        noise = torch.randn(inputs.shape[0], generator=generator, dtype=inputs.dtype)
        return chol @ noise


def generate_dataset(
    config: ExperimentConfig,
    *,
    function_seed: int,
    observation_seed: int,
) -> tuple[DAGDataset, SyntheticTruth]:
    """Generate one observational replicate of a fixed ancestral DAG-DGP draw."""
    function_generator = torch.Generator(device="cpu").manual_seed(
        20261026 + int(function_seed)
    )
    sobol = torch.quasirandom.SobolEngine(
        dimension=4, scramble=True, seed=987 + int(function_seed)
    )
    root_matrix = 4.0 * sobol.draw(config.num_data).to(torch.float64) - 2.0

    parent_kernel = RBFKernel(
        1,
        ard=False,
        init_lengthscale=config.parent_dgp_lengthscale,
        init_variance=config.parent_dgp_variance,
    )
    parent_latents = {
        name: _sample_gp(
            parent_kernel,
            root_matrix[:, idx : idx + 1],
            generator=function_generator,
        )
        for idx, name in enumerate(("A", "B", "D", "E"))
    }

    child_kernel = RBFKernel(
        1,
        ard=False,
        init_lengthscale=config.child_dgp_lengthscale,
        init_variance=config.child_dgp_component_variance,
    )
    strengths = {
        "A": config.contribution_A,
        "B": config.contribution_B,
        "D": config.contribution_D,
    }
    with torch.no_grad():
        child_covariance = torch.zeros(
            (config.num_data, config.num_data), dtype=torch.float64
        )
        for parent, strength in strengths.items():
            values = parent_latents[parent].unsqueeze(-1)
            child_covariance.add_(strength**2 * child_kernel.K(values, values))
        child_covariance = 0.5 * (child_covariance + child_covariance.transpose(0, 1))
        child_covariance.add_(1e-6 * torch.eye(config.num_data, dtype=torch.float64))
        C = torch.linalg.cholesky(child_covariance) @ torch.randn(
            config.num_data, generator=function_generator, dtype=torch.float64
        )

    observation_generator = torch.Generator(device="cpu").manual_seed(
        303000 + int(observation_seed)
    )
    observations = {
        name: values
        + config.parent_noise_std
        * torch.randn(
            config.num_data, generator=observation_generator, dtype=torch.float64
        )
        for name, values in parent_latents.items()
    }
    observations["C"] = C + config.child_noise_std * torch.randn(
        config.num_data, generator=observation_generator, dtype=torch.float64
    )

    test_count = config.num_data - int(round(config.child_train_fraction * config.num_data))
    permutation = torch.randperm(config.num_data, generator=observation_generator)
    child_test_mask = torch.zeros(config.num_data, dtype=torch.bool)
    child_test_mask[permutation[:test_count]] = True

    masks: dict[str, torch.Tensor] = {}
    parent_count = int(round(config.parent_observation_fraction * config.num_data))
    for node in ("A", "B", "D", "E"):
        node_perm = torch.randperm(config.num_data, generator=observation_generator)
        node_mask = torch.zeros(config.num_data, dtype=torch.bool)
        node_mask[node_perm[:parent_count]] = True
        masks[node] = node_mask
    masks["C"] = ~child_test_mask

    roots = {f"X{idx + 1}": root_matrix[:, idx : idx + 1] for idx in range(4)}
    dataset = DAGDataset(roots=roots, y=observations, mask=masks)
    truth = SyntheticTruth(
        observations=observations,
        child_test_mask=child_test_mask,
    )
    return dataset, truth


def _initialise_child_kernel(
    parent_names: tuple[str, ...],
    config: ExperimentConfig,
) -> AdditiveFusionKernel:
    kernel = AdditiveFusionKernel(
        parent_names,
        [1] * len(parent_names),
        base_kernel="rbf",
        ard=False,
        trainable_scales=True,
    )
    for scaled_kernel in kernel.kernels:
        base = scaled_kernel.base_kernel
        if not isinstance(base, RBFKernel):
            raise TypeError("The controlled experiment expects RBF child components.")
        base.set_lengthscale(config.child_lengthscale_init)
        base.set_variance(1.0)
        # Fix component base variances to identify the learned additive scale.
        base.raw_variance.requires_grad_(False)
        target = torch.tensor(
            [config.child_component_scale_init],
            dtype=scaled_kernel.raw_scale.dtype,
            device=scaled_kernel.raw_scale.device,
        )
        with torch.no_grad():
            scaled_kernel.raw_scale.copy_(inv_softplus(target))
    return kernel


def build_model(
    config: ExperimentConfig,
    variant: GraphVariant,
    *,
    design_seed: int,
    family: str = "dag_svi",
) -> DAGVIModel | DAGSVIModel:
    if family not in {"dag_vi", "dag_svi"}:
        raise ValueError(f"Unknown variational family: {family}.")
    parents = {
        "X1": [],
        "X2": [],
        "X3": [],
        "X4": [],
        "A": ["X1"],
        "B": ["X2"],
        "D": ["X3"],
        "E": ["X4"],
        "C": list(variant.child_parents),
    }
    dag = DAG(parents)

    module_class = MeanFieldSVGP if family == "dag_vi" else StructuredNodeSVGP
    gp_modules: dict[str, MeanFieldSVGP | StructuredNodeSVGP] = {}
    parent_inducing = torch.linspace(-2.0, 2.0, config.num_inducing_parent).unsqueeze(-1)
    for node in ("A", "B", "D", "E"):
        branch_kernel = RBFKernel(
            1,
            ard=False,
            init_lengthscale=config.branch_lengthscale_init,
            init_variance=1.0,
        )
        module = module_class(branch_kernel, parent_inducing, jitter=1e-5)
        module.Z.requires_grad_(False)
        gp_modules[node] = module

    # Use one common four-dimensional Sobol design and project it onto the
    # parent coordinates retained by each assumed graph.  This makes the child
    # inducing design paired across graph perturbations while keeping the same M.
    sobol = torch.quasirandom.SobolEngine(
        dimension=4, scramble=True, seed=1777 + int(design_seed)
    )
    child_pool = 4.0 * sobol.draw(config.num_inducing_child).to(torch.float64) - 2.0
    selected_columns = [PARENT_COLUMN[parent] for parent in variant.child_parents]
    child_inducing = child_pool[:, selected_columns]
    child_kernel = _initialise_child_kernel(variant.child_parents, config)
    child_module = module_class(child_kernel, child_inducing, jitter=1e-5)
    child_module.Z.requires_grad_(False)
    gp_modules["C"] = child_module

    likelihoods = {
        node: GaussianLikelihood(
            init_variance=(config.child_noise_std if node == "C" else config.parent_noise_std) ** 2
        )
        for node in ("A", "B", "D", "E", "C")
    }

    common = {
        "dag": dag,
        "root_dims": {f"X{idx}": 1 for idx in range(1, 5)},
        "gp_modules": gp_modules,
        "likelihoods": likelihoods,
    }
    if family == "dag_vi":
        model = DAGVIModel(**common)
    else:
        model = DAGSVIModel(
            **common,
            observed_nodes=("A", "B", "D", "E", "C"),
            marginal_backend="dense",
            init_q_diag_scale=1.0,
            init_q_offdiag_scale=0.0,
            local_conditioning_jitter=1e-6,
        )
    device = _normalise_device(config.device)
    return model.to(device=device, dtype=torch.float64)


def train_model(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    config: ExperimentConfig,
    *,
    seed: int,
) -> None:
    """Full-batch optimization with the same budget for both variational families."""
    torch.manual_seed(int(seed))
    reference = next(model.parameters())
    data = dataset.to(device=reference.device, dtype=reference.dtype)

    posterior = getattr(model, "posterior", None)
    posterior_params = (
        [parameter for parameter in posterior.parameters() if parameter.requires_grad]
        if posterior is not None
        else []
    )
    posterior_ids = {id(parameter) for parameter in posterior_params}
    likelihood_params = [
        parameter
        for likelihood in model.likelihoods.values()
        for parameter in likelihood.parameters()
        if parameter.requires_grad
    ]
    likelihood_ids = {id(parameter) for parameter in likelihood_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in posterior_ids
        and id(parameter) not in likelihood_ids
    ]

    parameter_groups = []
    if other_params:
        parameter_groups.append({"params": other_params, "lr": config.learning_rate})
    if posterior_params:
        parameter_groups.append(
            {
                "params": posterior_params,
                "lr": config.learning_rate * config.posterior_lr_scale,
            }
        )
    if likelihood_params:
        parameter_groups.append(
            {
                "params": likelihood_params,
                "lr": config.learning_rate * config.likelihood_lr_scale,
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)

    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        elbo = model.elbo(data, indices=None, num_mc_samples=config.num_mc_samples)
        if not bool(torch.isfinite(elbo)):
            raise FloatingPointError(f"Non-finite ELBO at step {step + 1}: {elbo}")
        (-elbo).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}: {grad_norm}")
        optimizer.step()

    model.eval()


def empirical_crps(samples: torch.Tensor, targets: torch.Tensor) -> float:
    """Empirical CRPS, computed in O(S log S) for S predictive samples."""
    first_term = (samples - targets.unsqueeze(0)).abs().mean(dim=0)
    ordered = samples.sort(dim=0).values
    ranks = torch.arange(
        1,
        samples.shape[0] + 1,
        dtype=samples.dtype,
        device=samples.device,
    ).unsqueeze(1)
    second_term = (
        (2.0 * ranks - samples.shape[0] - 1.0) * ordered
    ).sum(dim=0) / float(samples.shape[0] ** 2)
    return float((first_term - second_term).mean().cpu())


@torch.no_grad()
def evaluate_metrics(
    model: DAGVIModel | DAGSVIModel,
    dataset: DAGDataset,
    truth: SyntheticTruth,
    config: ExperimentConfig,
    *,
    seed: int,
) -> dict[str, float]:
    test_indices = torch.nonzero(truth.child_test_mask, as_tuple=False).squeeze(-1)
    reference = next(model.parameters())
    evaluation_data = DAGDataset(
        roots={name: values.index_select(0, test_indices) for name, values in dataset.roots.items()},
        y={},
    ).to(device=reference.device, dtype=reference.dtype)
    torch.manual_seed(10000 + int(seed))
    latent_samples = model.sample_latents(
        evaluation_data, num_samples=config.evaluation_samples,
    )["C"].squeeze(-1)
    latent_mean = latent_samples.mean(dim=0)
    y_test = truth.observations["C"].index_select(0, test_indices).to(reference)
    rmse = float(torch.mean((latent_mean - y_test).square()).sqrt().cpu())

    likelihood_std = model.likelihood("C").variance.sqrt()
    torch.manual_seed(20000 + int(seed))
    predictive_samples = latent_samples + likelihood_std * torch.randn_like(latent_samples)
    return {
        "rmse": rmse,
        "crps": empirical_crps(predictive_samples, y_test),
    }


@dataclass(frozen=True)
class FitResult:
    family: str
    function_seed: int
    observation_seed: int
    fit_seed: int
    variant: str
    rmse: float
    crps: float


def run_fit(task: tuple[ExperimentConfig, str, int, int, int, str]) -> FitResult:
    config, family, function_seed, observation_seed, fit_seed, variant_key = task
    dataset, truth = generate_dataset(
        config,
        function_seed=function_seed,
        observation_seed=observation_seed,
    )
    torch.manual_seed(30000 + int(fit_seed))
    model = build_model(
        config,
        VARIANT_BY_KEY[variant_key],
        design_seed=function_seed,
        family=family,
    )
    train_model(model, dataset, config, seed=fit_seed)
    # Common random numbers keep Monte Carlo scoring noise out of the variation
    # across observational replicates.
    metrics = evaluate_metrics(model, dataset, truth, config, seed=function_seed)
    if any(not np.isfinite(value) or value <= 0 for value in metrics.values()):
        raise FloatingPointError(
            f"Invalid metrics for {family}, function_seed={function_seed}, "
            f"observation_seed={observation_seed}, "
            f"fit_seed={fit_seed}, {variant_key}: {metrics}"
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return FitResult(
        family,
        function_seed,
        observation_seed,
        fit_seed,
        variant_key,
        **metrics,
    )


def render_figure(
    results: list[FitResult],
    output: Path,
    *,
    metric: str = "rmse",
) -> None:
    if metric not in {"rmse", "crps"}:
        raise ValueError(f"Unsupported figure metric: {metric}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (
            r.family,
            r.function_seed,
            r.observation_seed,
            r.fit_seed,
            r.variant,
        ): getattr(r, metric)
        for r in results
    }
    repetitions = sorted({
        (r.function_seed, r.observation_seed, r.fit_seed) for r in results
    })
    expected = {
        (family, function_seed, observation_seed, fit_seed, variant.key)
        for family in ("dag_vi", "dag_svi")
        for function_seed, observation_seed, fit_seed in repetitions
        for variant in VARIANTS
    }
    if len(repetitions) < 2 or len(lookup) != len(results) or set(lookup) != expected:
        raise ValueError("The figure requires complete, paired fits for both families and all graphs.")
    labels = [
        "True\n" + r"$\{w_1,w_2,w_3\}$",
        r"$+w_4$",
        r"$-w_3$",
        r"$-\{w_2,w_3\}$",
        r"$-w_1$",
    ]
    style = {
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 10.5,
        "axes.labelsize": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.linewidth": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(4.15, 2.85))
        fig.subplots_adjust(left=.15, right=.98, top=.96, bottom=.25)
        x = np.arange(len(VARIANTS))
        upper_bounds = []
        for family, color, marker, offset, label in [
            ("dag_vi", "#376584", "s", -.11, "DAG-VI"),
            ("dag_svi", "#C16A45", "o", .11, "DAG-SVI"),
        ]:
            ratios = np.asarray([
                [
                    lookup[
                        family,
                        function_seed,
                        observation_seed,
                        fit_seed,
                        variant.key,
                    ]
                    / lookup[
                        family,
                        function_seed,
                        observation_seed,
                        fit_seed,
                        "correct",
                    ]
                    for function_seed, observation_seed, fit_seed in repetitions
                ]
                for variant in VARIANTS
            ])
            means = ratios.mean(axis=1)
            deviations = ratios.std(axis=1, ddof=1)
            upper_bounds.extend((means + deviations).tolist())
            ax.errorbar(
                x + offset, means, yerr=deviations,
                fmt=marker, linestyle="none", color=color, ecolor=color,
                markersize=4.8, markerfacecolor="white" if family == "dag_vi" else color,
                markeredgewidth=1.0, elinewidth=1.1, capsize=2.7, capthick=.9,
                barsabove=True, label=label, zorder=3,
            )
        ax.axhline(1, color="#727B82", linewidth=.75, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(-.5, 4.5)
        upper_limit = max(1.2, 1.04 * max(upper_bounds))
        tick_step = .5 if upper_limit <= 5 else 2.0
        ticks = np.arange(1.0, upper_limit + 1e-9, tick_step)
        ax.set_ylim(.8, upper_limit)
        ax.set_xticks(x, labels)
        ax.set_yticks(ticks, [f"{tick:g}×" for tick in ticks])
        ax.set_ylabel(f"Relative {metric.upper()}", labelpad=5)
        ax.set_xlabel(r"$\mathrm{Pa}(w_5)$", labelpad=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#6F777C")
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=3, width=.6, color="#6F777C")
        ax.grid(axis="y", color="#E6E8EA", linewidth=.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, handletextpad=.55,
                  labelspacing=.55, borderaxespad=.65)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=.045)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel fits (default: up to 8 CPU processes).")
    parser.add_argument("--device", default=os.environ.get("DAGDGP_DEVICE", "cpu"),
                        help="PyTorch device (default: cpu). Use --jobs 1 for CUDA.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination RMSE PDF; the CRPS PDF is saved alongside it.")
    parser.add_argument("--function-seeds", nargs="+", type=int, default=None,
                        help="DAG-DGP function-draw seeds (default: 0).")
    parser.add_argument("--replicate-seeds", nargs="+", type=int, default=None,
                        help="Paired observation/training seeds (default: 0,...,4).")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf suffix")
    device = _normalise_device(args.device)
    if device.type != "cpu" and args.jobs != 1:
        parser.error("Use --jobs 1 for a non-CPU device")
    config = ExperimentConfig(device=str(device))
    function_seeds = (
        args.function_seeds
        if args.function_seeds is not None
        else [config.function_seed]
    )
    replicate_seeds = (
        args.replicate_seeds
        if args.replicate_seeds is not None
        else list(range(config.num_replicates))
    )
    tasks = [
        (config, family, function_seed, replicate_seed, replicate_seed, variant.key)
        for family in ("dag_vi", "dag_svi")
        for function_seed in function_seeds
        for replicate_seed in replicate_seeds
        for variant in VARIANTS
    ]
    results = []

    def collect(iterator):
        for result in iterator:
            results.append(result)
            print(f"[{len(results)}/{len(tasks)}] {result.family} "
                  f"function_seed={result.function_seed} "
                  f"observation_seed={result.observation_seed} "
                  f"fit_seed={result.fit_seed} "
                  f"graph={result.variant} "
                  f"RMSE={result.rmse:.8f} CRPS={result.crps:.8f}", flush=True)

    if args.jobs == 1:
        collect(map(run_fit, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            collect(executor.map(run_fit, tasks))
    render_figure(results, args.output, metric="rmse")
    crps_stem = (
        f"{args.output.stem[:-5]}_CRPS"
        if args.output.stem.endswith("_RMSE")
        else f"{args.output.stem}_CRPS"
    )
    crps_output = args.output.with_name(f"{crps_stem}.pdf")
    render_figure(results, crps_output, metric="crps")
    print(f"Saved outputs to {args.output.parent}", flush=True)


if __name__ == "__main__":
    main()
