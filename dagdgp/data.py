"""Dataset containers for DAG-structured Gaussian process models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class DAGDataset:
    """Observed data for a DAG model with scalar latent nodes.

    Shapes for a dataset with ``N`` cases:
    - ``roots[r]``: ``(N, d_r)``
    - ``y[w]``: ``(N,)`` or ``(N, 1)``
    - ``mask[w]``: ``(N,)`` or ``(N, 1)``
    """

    roots: Mapping[str, Tensor]
    y: Mapping[str, Tensor]
    mask: Mapping[str, Tensor] | None = None

    def __post_init__(self) -> None:
        n: int | None = None

        for root, x in self.roots.items():
            if x.dim() != 2:
                raise ValueError(f"Root input '{root}' must have shape (N, d).")
            if n is None:
                n = int(x.shape[0])
            elif int(x.shape[0]) != n:
                raise ValueError("All root inputs must share the same number of rows.")

        if n is None:
            raise ValueError("At least one root input is required.")

        for node, values in self.y.items():
            if values.dim() not in (1, 2) or int(values.shape[0]) != n:
                raise ValueError(f"Observation y['{node}'] must have shape (N,) or (N, 1).")

        if self.mask is None:
            object.__setattr__(self, "mask", {})

        for node, values in self.mask.items():
            if values.dim() not in (1, 2) or int(values.shape[0]) != n:
                raise ValueError(f"Mask mask['{node}'] must have shape (N,) or (N, 1).")

    @property
    def num_data(self) -> int:
        """Number of cases in the dataset."""
        return next(iter(self.roots.values())).shape[0]

    def get_observations(self, node: str) -> tuple[Tensor, Tensor]:
        """Return ``(y, mask)`` for a node.

        Nodes without an explicit mask are treated as fully observed.
        """
        if node not in self.y:
            raise KeyError(f"No observations provided for node '{node}'.")

        y = self.y[node]
        if y.dim() == 2:
            y = y.squeeze(-1)

        mask = self.mask.get(node)
        if mask is None:
            mask = torch.ones_like(y, dtype=torch.bool)
        else:
            if mask.dim() == 2:
                mask = mask.squeeze(-1)
            mask = mask.to(dtype=torch.bool)

        return y, mask

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "DAGDataset":
        """Move all tensors to ``device`` and optionally cast floating tensors."""

        def _cast(tensor: Tensor) -> Tensor:
            out = tensor.to(device)
            if dtype is not None and out.is_floating_point():
                out = out.to(dtype=dtype)
            return out

        return DAGDataset(
            roots={name: _cast(x) for name, x in self.roots.items()},
            y={name: _cast(values) for name, values in self.y.items()},
            mask={name: _cast(values) for name, values in self.mask.items()},
        )
