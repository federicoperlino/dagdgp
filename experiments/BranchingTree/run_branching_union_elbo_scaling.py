#!/usr/bin/env python3
"""Reproduce the final BranchingTree scaling figure.

This script benchmarks ELBO evaluation on branching-tree DAGs with:
- depths 2..9
- 20 training cases
- 20 inducing points per latent node
- 1 Monte Carlo sample
- GPU by default when available

It writes:
- benchmark.json
- elbo_eval_scaling.pdf
into this directory.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


DEPTHS = [2, 3, 4, 5, 6, 7, 8, 9]
NUM_TRAIN = 20
NUM_INDUCING = 20
NUM_MC_SAMPLES = 1
REPEATS = 3
SEED = 20260505
ROOT_DIM = 2
DTYPE = torch.float64
OUTPUT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




REPO_ROOT = Path(__file__).resolve().parents[2]
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


@dataclass(frozen=True)
class BranchingData:
    dag: DAG
    root_dims: dict[str, int]
    observed_nodes: tuple[str, ...]
    node_layers: dict[str, int]
    train: DAGDataset
    train_latents: dict[str, torch.Tensor]


def build_branching_dag(depth: int) -> tuple[DAG, dict[str, int]]:
    parents: dict[str, list[str]] = {"X": []}
    node_layers: dict[str, int] = {}
    frontier = ["X"]
    for layer in range(depth):
        next_frontier: list[str] = []
        for parent_index, parent in enumerate(frontier):
            for branch in range(2):
                node = f"L{layer}_{parent_index:02d}_{branch}"
                parents[node] = [parent]
                node_layers[node] = layer
                next_frontier.append(node)
        frontier = next_frontier
    return DAG(parents), node_layers


def draw_rff_gp(
    X: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    num_features: int,
    generator: torch.Generator,
) -> torch.Tensor:
    omega = torch.randn(
        (int(X.shape[1]), int(num_features)),
        generator=generator,
        device=X.device,
        dtype=X.dtype,
    ) / float(lengthscale)
    phase = 2.0 * math.pi * torch.rand(
        (int(num_features),),
        generator=generator,
        device=X.device,
        dtype=X.dtype,
    )
    coeff = torch.randn(
        (int(num_features),),
        generator=generator,
        device=X.device,
        dtype=X.dtype,
    )
    features = torch.cos(X @ omega + phase)
    scale = math.sqrt(2.0 * float(variance) / float(num_features))
    return scale * (features @ coeff)


def parent_matrix(
    dag: DAG,
    *,
    node: str,
    roots: dict[str, torch.Tensor],
    latents: dict[str, torch.Tensor],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for parent in dag.parents_of(node):
        parts.append(roots[parent] if parent in roots else latents[parent].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


def generate_data(depth: int) -> BranchingData:
    dag, node_layers = build_branching_dag(depth)
    observed_nodes = tuple(node for node, layer in node_layers.items() if layer == depth - 1)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    roots = {
        "X": -2.0 + 4.0 * torch.rand((NUM_TRAIN, ROOT_DIM), generator=generator, device=DEVICE, dtype=DTYPE)
    }

    latents: dict[str, torch.Tensor] = {}
    for node in dag.topo:
        if node in roots:
            continue
        parents = parent_matrix(dag, node=node, roots=roots, latents=latents)
        layer = node_layers[node]
        draw = draw_rff_gp(
            parents,
            lengthscale=1.2 if layer == 0 else 0.9,
            variance=1.0,
            num_features=96,
            generator=generator,
        )
        draw = draw + 0.02 * torch.randn(draw.shape, generator=generator, device=DEVICE, dtype=DTYPE)
        latents[node] = 1.25 * torch.tanh(draw / 1.25)

    observations = {
        node: latents[node] + 0.05 * torch.randn(latents[node].shape, generator=generator, device=DEVICE, dtype=DTYPE)
        for node in observed_nodes
    }

    return BranchingData(
        dag=dag,
        root_dims={"X": ROOT_DIM},
        observed_nodes=observed_nodes,
        node_layers=node_layers,
        train=DAGDataset(roots=roots, y=observations),
        train_latents=latents,
    )


def select_inducing_rows(X: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    if int(X.shape[0]) >= NUM_INDUCING:
        rows = torch.randperm(int(X.shape[0]), generator=generator, device=X.device)[:NUM_INDUCING]
        return X[rows].clone()
    rows = torch.randint(int(X.shape[0]), (NUM_INDUCING,), generator=generator, device=X.device)
    return X[rows].clone()


def initial_inducing_inputs(data: BranchingData, *, node: str, generator: torch.Generator) -> torch.Tensor:
    parents = parent_matrix(data.dag, node=node, roots=data.train.roots, latents=data.train_latents)
    return select_inducing_rows(parents, generator=generator)


def make_kernel(input_dim: int, *, layer: int) -> RBFKernel:
    return RBFKernel(
        input_dim,
        ard=True,
        init_lengthscale=1.2 if layer == 0 else 0.9,
        init_variance=1.0,
    )


def build_dag_svi_model(data: BranchingData, *, backend: str) -> DAGSVIModel:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)
    gp_modules = {}
    for node in data.dag.non_roots:
        input_dim = sum(data.root_dims[parent] if parent in data.root_dims else 1 for parent in data.dag.parents_of(node))
        gp_modules[node] = StructuredNodeSVGP(
            make_kernel(input_dim, layer=data.node_layers[node]),
            initial_inducing_inputs(data, node=node, generator=generator),
        )
    likelihoods = {node: GaussianLikelihood(init_variance=0.02) for node in data.observed_nodes}
    model = DAGSVIModel(
        dag=data.dag,
        root_dims=data.root_dims,
        gp_modules=gp_modules,
        likelihoods=likelihoods,
        observed_nodes=data.observed_nodes,
        analytic_gaussian_leaves=True,
        init_q_mean=0.0,
        init_q_diag_scale=1.0,
        init_q_offdiag_scale=0.02,
        q_diag_parametrization="softplus-diag",
        marginal_backend=backend,
    )
    return model.to(device=DEVICE, dtype=DTYPE)


def build_dag_vi_model(data: BranchingData) -> DAGVIModel:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)
    gp_modules = {}
    for node in data.dag.non_roots:
        input_dim = sum(data.root_dims[parent] if parent in data.root_dims else 1 for parent in data.dag.parents_of(node))
        gp_modules[node] = MeanFieldSVGP(
            make_kernel(input_dim, layer=data.node_layers[node]),
            initial_inducing_inputs(data, node=node, generator=generator),
            q_root_parametrization="softplus-diag",
        )
    likelihoods = {node: GaussianLikelihood(init_variance=0.02) for node in data.observed_nodes}
    model = DAGVIModel(
        dag=data.dag,
        root_dims=data.root_dims,
        gp_modules=gp_modules,
        likelihoods=likelihoods,
    )
    return model.to(device=DEVICE, dtype=DTYPE)


def synchronize() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def benchmark_model(model, dataset: DAGDataset, indices: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        torch.manual_seed(SEED)
        synchronize()
        elbo = float(model.elbo(dataset, indices=indices, num_mc_samples=NUM_MC_SAMPLES))
        synchronize()
        times: list[float] = []
        for _ in range(REPEATS):
            torch.manual_seed(SEED)
            synchronize()
            start = time.perf_counter()
            float(model.elbo(dataset, indices=indices, num_mc_samples=NUM_MC_SAMPLES))
            synchronize()
            times.append(time.perf_counter() - start)
    return {"elbo": elbo, "mean_s": float(sum(times) / len(times)), "times_s": times}


def render_plot(records: list[dict[str, object]]) -> None:
    x = [int(record["total_inducing_dim"]) for record in records]
    depths = [int(record["depth"]) for record in records]
    vi = [float(record["mf_union"]["mean_s"]) for record in records]
    dense = [float(record["dense_union"]["mean_s"]) for record in records]
    sparse = [float(record["sparse_union"]["mean_s"]) for record in records]

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "grid.color": "#CFCBC3",
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        x,
        vi,
        marker="o",
        linewidth=1.9,
        markersize=5.8,
        color="#6F7D8C",
        linestyle="-",
        label="DAG-VI",
    )
    ax.plot(
        x,
        dense,
        marker="s",
        linewidth=1.9,
        markersize=6.0,
        color="#E69F00",
        linestyle=(0, (4.4, 2.4)),
        markerfacecolor="white",
        markeredgecolor="#E69F00",
        markeredgewidth=1.25,
        label=r"DAG-SVI (dense $\Lambda^{-1}$)",
    )
    ax.plot(
        x,
        sparse,
        marker="^",
        linewidth=1.9,
        markersize=6.6,
        color="#0072B2",
        linestyle="-",
        markerfacecolor="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=r"DAG-SVI (sparse $\Lambda^{-1}$)",
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"Total inducing dimension ({NUM_INDUCING} per latent node)")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.grid(True, axis="y", which="both")
    ax.legend(frameon=False, loc="upper left", handlelength=2.8)
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels([str(depth) for depth in depths])
    top.set_xlabel("Branching-tree depth")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "elbo_eval_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Benchmarking on {DEVICE} with depths={DEPTHS}, num_train={NUM_TRAIN}, num_inducing={NUM_INDUCING}")

    records: list[dict[str, object]] = []
    for depth in DEPTHS:
        data = generate_data(depth)
        base_structured = build_dag_svi_model(data, backend="dense")
        dense_model = copy.deepcopy(base_structured)
        dense_model.marginal_backend = "dense"
        dense_model.eval()
        sparse_model = copy.deepcopy(base_structured)
        sparse_model.marginal_backend = "sparse"
        sparse_model.eval()
        mf_model = build_dag_vi_model(data)
        mf_model.eval()
        full_indices = torch.arange(data.train.num_data, device=DEVICE)

        record = {
            "depth": depth,
            "num_latent_nodes": len(base_structured.structure.order),
            "total_inducing_dim": int(base_structured.posterior.total_dim),
            "mf_union": benchmark_model(mf_model, data.train, full_indices),
            "dense_union": benchmark_model(dense_model, data.train, full_indices),
            "sparse_union": benchmark_model(sparse_model, data.train, full_indices),
        }
        records.append(record)
        print(json.dumps({
            "depth": depth,
            "total_inducing_dim": record["total_inducing_dim"],
            "mf_s": record["mf_union"]["mean_s"],
            "dense_s": record["dense_union"]["mean_s"],
            "sparse_s": record["sparse_union"]["mean_s"],
        }), flush=True)

    (OUTPUT_DIR / "benchmark.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    render_plot(records)


if __name__ == "__main__":
    main()
