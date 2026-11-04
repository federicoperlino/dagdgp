"""DAG helper functions shared across inference families."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .dag import DAG
from .data import DAGDataset


def validate_dataset_roots(root_dims: Mapping[str, int], dataset: DAGDataset) -> None:
    """Validate that a dataset provides exactly the required root inputs."""
    for root, dim in root_dims.items():
        values = dataset.roots.get(root)
        if values is None:
            raise ValueError(f"Dataset missing root inputs for required root '{root}'.")
        if values.dim() != 2:
            raise ValueError(f"Root input '{root}' must be rank-2, got shape {tuple(values.shape)}.")
        if int(values.shape[1]) != int(dim):
            raise ValueError(f"Root '{root}' has dimension {values.shape[1]}, expected {int(dim)}.")

    extra = sorted(set(dataset.roots) - set(root_dims))
    if extra:
        raise ValueError(f"Dataset contains extra roots not present in root_dims: {extra}.")


def prepare_dataset(dataset: DAGDataset, *, root_dims: Mapping[str, int], reference: Tensor) -> DAGDataset:
    """Move a dataset to the reference tensor device/dtype and validate roots."""
    prepared = dataset.to(device=reference.device, dtype=reference.dtype)
    validate_dataset_roots(root_dims, prepared)
    return prepared


def assemble_node_inputs(
    dag: DAG,
    *,
    node: str,
    root_dims: Mapping[str, int],
    root_values: Mapping[str, Tensor],
    latent_values: Mapping[str, Tensor],
) -> Tensor:
    """Assemble the parent inputs for one node."""
    parts: list[Tensor] = []
    for parent in dag.parents_of(node):
        if parent in root_dims:
            parts.append(root_values[parent])
        else:
            values = latent_values[parent]
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            parts.append(values)

    if len(parts) == 0:
        raise ValueError("Non-root nodes must have at least one parent.")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def sampled_node_inputs(
    dag: DAG,
    *,
    node: str,
    root_dims: Mapping[str, int],
    root_values: Mapping[str, Tensor],
    latent_samples: Mapping[str, Tensor],
    num_samples: int,
) -> Tensor:
    """Assemble parent inputs when latent parents are represented by samples."""
    batch_size = next(iter(root_values.values())).shape[0]
    parts: list[Tensor] = []

    for parent in dag.parents_of(node):
        if parent in root_dims:
            values = root_values[parent]
            parts.append(values.unsqueeze(0).expand(num_samples, batch_size, values.shape[1]))
        else:
            values = latent_samples[parent]
            if values.dim() == 2:
                values = values.unsqueeze(-1)
            if values.dim() != 3:
                raise ValueError(f"Sampled latent '{parent}' must be rank-3, got {tuple(values.shape)}.")
            if int(values.shape[0]) != int(num_samples):
                raise ValueError(
                    f"Sampled latent '{parent}' has sample dimension {values.shape[0]}, expected {num_samples}."
                )
            parts.append(values)

    if len(parts) == 0:
        raise ValueError("Non-root nodes must have at least one parent.")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
