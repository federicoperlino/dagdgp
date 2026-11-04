#!/usr/bin/env python3
"""Benchmark ELBO evaluation on K-wide, 3-deep layered DAGs.

For every width K, the DAG contains K parallel three-node branches:

    X -> C[k, 0] -> C[k, 1] -> C[k, 2],  k = 1, ..., K,

where X is the shared observed root and the final layer is observed. The
benchmark uses the same data generation, models, timing protocol, and plotting
style as the BranchingTree experiment.

It writes benchmark.json and elbo_eval_scaling.pdf into this directory.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagdgp import DAG, DAGDataset  
from experiments.BranchingTree import run_branching_union_elbo_scaling as bt  


WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
DEPTH = 3
OUTPUT_DIR = Path(__file__).resolve().parent


def build_k_wide_dag(width: int) -> tuple[DAG, dict[str, int], tuple[str, ...]]:
    parents: dict[str, list[str]] = {"X": []}
    node_layers: dict[str, int] = {}
    observed_nodes: list[str] = []

    for branch in range(width):
        parent = "X"
        for layer in range(DEPTH):
            node = f"C{branch:03d}_L{layer}"
            parents[node] = [parent]
            node_layers[node] = layer
            parent = node
        observed_nodes.append(parent)

    return DAG(parents), node_layers, tuple(observed_nodes)


def generate_data(width: int) -> bt.BranchingData:
    dag, node_layers, observed_nodes = build_k_wide_dag(width)
    generator = torch.Generator(device=bt.DEVICE)
    generator.manual_seed(bt.SEED)

    roots = {
        "X": -2.0
        + 4.0
        * torch.rand(
            (bt.NUM_TRAIN, bt.ROOT_DIM),
            generator=generator,
            device=bt.DEVICE,
            dtype=bt.DTYPE,
        )
    }

    latents: dict[str, torch.Tensor] = {}
    for node in dag.topo:
        if node in roots:
            continue
        parents = bt.parent_matrix(dag, node=node, roots=roots, latents=latents)
        layer = node_layers[node]
        draw = bt.draw_rff_gp(
            parents,
            lengthscale=1.2 if layer == 0 else 0.9,
            variance=1.0,
            num_features=96,
            generator=generator,
        )
        draw = draw + 0.02 * torch.randn(
            draw.shape,
            generator=generator,
            device=bt.DEVICE,
            dtype=bt.DTYPE,
        )
        latents[node] = 1.25 * torch.tanh(draw / 1.25)

    observations = {
        node: latents[node]
        + 0.05
        * torch.randn(
            latents[node].shape,
            generator=generator,
            device=bt.DEVICE,
            dtype=bt.DTYPE,
        )
        for node in observed_nodes
    }

    return bt.BranchingData(
        dag=dag,
        root_dims={"X": bt.ROOT_DIM},
        observed_nodes=observed_nodes,
        node_layers=node_layers,
        train=DAGDataset(roots=roots, y=observations),
        train_latents=latents,
    )


def render_plot(records: list[dict[str, object]]) -> None:
    x = [int(record["total_inducing_dim"]) for record in records]
    widths = [int(record["width"]) for record in records]
    vi = [float(record["mf_union"]["mean_s"]) for record in records]
    dense = [float(record["dense_union"]["mean_s"]) for record in records]
    sparse = [float(record["sparse_union"]["mean_s"]) for record in records]

    bt.plt.rcParams.update({
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
    fig, ax = bt.plt.subplots(figsize=(7.2, 4.6))
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
    ax.set_xlabel(f"Total inducing dimension ({bt.NUM_INDUCING} per latent node)")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.grid(True, axis="y", which="both")
    ax.legend(frameon=False, loc="upper left", handlelength=2.8)
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels([str(width) for width in widths])
    top.set_xlabel(r"Width $W$ (three-node branches)")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "elbo_eval_scaling.pdf", bbox_inches="tight")
    bt.plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Benchmarking on {bt.DEVICE} with widths={WIDTHS}, depth={DEPTH}, "
        f"num_train={bt.NUM_TRAIN}, num_inducing={bt.NUM_INDUCING}"
    )

    records: list[dict[str, object]] = []
    for width in WIDTHS:
        data = generate_data(width)
        base_structured = bt.build_dag_svi_model(data, backend="dense")
        dense_model = copy.deepcopy(base_structured)
        dense_model.marginal_backend = "dense"
        dense_model.eval()
        sparse_model = copy.deepcopy(base_structured)
        sparse_model.marginal_backend = "sparse"
        sparse_model.eval()
        mf_model = bt.build_dag_vi_model(data)
        mf_model.eval()
        full_indices = torch.arange(data.train.num_data, device=bt.DEVICE)

        record = {
            "width": width,
            "depth": DEPTH,
            "num_components": width,
            "num_latent_nodes": len(base_structured.structure.order),
            "total_inducing_dim": int(base_structured.posterior.total_dim),
            "mf_union": bt.benchmark_model(mf_model, data.train, full_indices),
            "dense_union": bt.benchmark_model(dense_model, data.train, full_indices),
            "sparse_union": bt.benchmark_model(sparse_model, data.train, full_indices),
        }
        records.append(record)
        print(
            json.dumps({
                "width": width,
                "total_inducing_dim": record["total_inducing_dim"],
                "mf_s": record["mf_union"]["mean_s"],
                "dense_s": record["dense_union"]["mean_s"],
                "sparse_s": record["sparse_union"]["mean_s"],
            }),
            flush=True,
        )

    (OUTPUT_DIR / "benchmark.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    render_plot(records)


if __name__ == "__main__":
    main()
