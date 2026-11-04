"""Posterior predictions and output utilities for HeavyIon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dagdgp import DAGDataset, DAGVIModel, DAGSVIModel
from dagdgp.dag_helpers import sampled_node_inputs
from dagdgp.gaussian import (
    sample_f_conditional_joint_from_whitened_xi,
    solve_lower_triangular,
    stable_cholesky,
)
from experiments.HeavyIon.common.metrics import summarise_scalar_samples


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="ascii")


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


@torch.no_grad()
def _joint_dag_svi_samples(model: DAGSVIModel, dataset, num_samples: int):
    """Draw DAG-SVI latent samples jointly across inputs."""
    xi = model.posterior.sample_whitened(num_samples)
    samples = {}
    for node in model.dag.topo:
        if node in model.root_dims:
            continue
        gp = model.gp_modules[node]
        inducing = gp.Z.detach()
        luu = stable_cholesky(gp.kernel.K(inducing, inducing), jitter=gp.jitter)
        inputs = sampled_node_inputs(
            model.dag,
            node=node,
            root_dims=model.root_dims,
            root_values=dataset.roots,
            latent_samples=samples,
            num_samples=num_samples,
        )
        node_xi = xi[:, model.posterior.block_slice(node)]
        if all(parent in model.root_dims for parent in model.dag.parents[node]):
            x = inputs[0]
            interpolation = solve_lower_triangular(luu, gp.kernel.K(inducing, x))
            residual = gp.kernel.K(x, x) - interpolation.T @ interpolation
            chol = stable_cholesky(0.5 * (residual + residual.T), jitter=gp.jitter)
            noise = torch.randn(
                num_samples, len(x), device=x.device, dtype=x.dtype
            )
            draws = node_xi @ interpolation + noise @ chol.T + gp.mean_offset
        else:
            draws = torch.stack([
                gp.mean_offset + sample_f_conditional_joint_from_whitened_xi(
                    gp.kernel,
                    inducing,
                    inputs[s],
                    node_xi[s],
                    jitter=gp.jitter,
                    Luu=luu,
                )
                for s in range(num_samples)
            ])
        samples[node] = draws.unsqueeze(-1)
    return samples


def posterior_payload(
    model: DAGVIModel | DAGSVIModel, dataset, *, num_samples: int, joint: bool = False
) -> dict[str, Any]:
    if joint:
        keep = torch.stack(list(dataset.mask.values())).any(dim=0)
        dataset = DAGDataset(
            roots={k: v[keep] for k, v in dataset.roots.items()},
            y={k: v[keep] for k, v in dataset.y.items()},
            mask={k: v[keep] for k, v in dataset.mask.items()},
        )
    with torch.no_grad():
        if joint and isinstance(model, DAGSVIModel):
            samples = _joint_dag_svi_samples(model, dataset, int(num_samples))
        else:
            samples = model.sample_latents(dataset, num_samples=int(num_samples))

    node_samples = {
        node: value.squeeze(-1).detach().cpu() for node, value in samples.items()
    }
    summary = {
        node: {
            name: tensor.detach().cpu()
            for name, tensor in summarise_scalar_samples(value).items()
        }
        for node, value in node_samples.items()
    }
    return {
        "num_samples": int(num_samples),
        "roots": {
            name: tensor.detach().cpu() for name, tensor in dataset.roots.items()
        },
        "y": {name: tensor.detach().cpu() for name, tensor in dataset.y.items()},
        "mask": {name: tensor.detach().cpu() for name, tensor in dataset.mask.items()},
        "node_samples": node_samples,
        "summary": summary,
    }
