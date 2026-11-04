"""Block-sparse whitened Gaussian posterior for DAG-SVI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..gaussian import solve_right_lower_triangular
from ..utils import inv_softplus, softplus_positive, validate_positive_finite
from .graph import ChordalStructure
from .precision import precision_blocks_from_root, precision_vector_product


def _canonical_diag_parametrization(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "direct": "direct",
        "standard": "direct",
        "softplus": "softplus-diag",
        "softplus-diag": "softplus-diag",
    }
    if key not in aliases:
        raise ValueError("diag_parametrization must be one of {'direct', 'softplus-diag'}.")
    return aliases[key]


def _lower_triangular_block(raw_block: Tensor, *, parametrization: str, eps: float = 1e-6) -> Tensor:
    if raw_block.dim() != 2 or raw_block.shape[0] != raw_block.shape[1]:
        raise ValueError("raw_block must be square.")
    block = torch.tril(raw_block)
    if parametrization == "softplus-diag":
        diag = torch.diagonal(block)
        positive_diag = softplus_positive(diag, minimum=eps)
        block = block - torch.diag(diag) + torch.diag(positive_diag)
    return block


@dataclass(frozen=True)
class BlockMarginals:
    """Posterior means and marginal covariance blocks for each node."""

    mean_blocks: Mapping[str, Tensor]
    cov_blocks: Mapping[str, Tensor]


@dataclass(frozen=True)
class BlockSparseStatistics:
    """Cached posterior statistics built from one selected-inverse pass."""

    marginals: BlockMarginals
    trace_covariance: Tensor
    logdet_precision: Tensor


class BlockSparsePrecisionGMRF(nn.Module):
    """Whitened Gaussian posterior with a block-sparse precision Cholesky root."""

    def __init__(
        self,
        *,
        structure: ChordalStructure,
        block_sizes: Mapping[str, int],
        init_mean: float = 0.0,
        init_diag_scale: float = 1.0,
        init_offdiag_scale: float = 0.0,
        diag_parametrization: str = "softplus-diag",
    ) -> None:
        super().__init__()

        validate_positive_finite(init_diag_scale, "init_diag_scale")
        self.structure = structure
        self.diag_parametrization = _canonical_diag_parametrization(diag_parametrization)

        self.block_sizes = {node: int(block_sizes[node]) for node in self.structure.order}
        missing = sorted(set(self.structure.order) - set(self.block_sizes))
        if missing:
            raise ValueError(f"Missing block sizes for nodes: {missing}.")

        self._offsets = [0]
        for node in self.structure.order:
            size = int(self.block_sizes[node])
            if size <= 0:
                raise ValueError(f"Block size for node '{node}' must be positive.")
            self._offsets.append(self._offsets[-1] + size)
        total_dim = int(self._offsets[-1])
        self.q_mean = nn.Parameter(torch.full((total_dim,), float(init_mean), dtype=torch.float64))

        diag_params = nn.ParameterList()
        for node in self.structure.order:
            size = int(self.block_sizes[node])
            raw = torch.zeros((size, size), dtype=torch.float64)
            init_diag = torch.full((size,), float(init_diag_scale), dtype=torch.float64)
            if self.diag_parametrization == "softplus-diag":
                init_diag = inv_softplus(init_diag)
            raw = raw + torch.diag(init_diag)
            diag_params.append(nn.Parameter(raw))
        self.raw_diag_blocks = diag_params

        offdiag_params = nn.ParameterDict()
        for row, col in self.structure.lower_block_pairs:
            shape = (self.block_size_by_index(row), self.block_size_by_index(col))
            raw = torch.full(shape, float(init_offdiag_scale), dtype=torch.float64)
            offdiag_params[self._pair_key(row, col)] = nn.Parameter(raw)
        self.raw_offdiag_blocks = offdiag_params

    @property
    def total_dim(self) -> int:
        return int(self._offsets[-1])

    def block_size_by_index(self, index: int) -> int:
        return int(self._offsets[index + 1] - self._offsets[index])

    def block_slice_by_index(self, index: int) -> slice:
        return slice(int(self._offsets[index]), int(self._offsets[index + 1]))

    def block_slice(self, node: str) -> slice:
        return self.block_slice_by_index(self.structure.node_to_index[node])

    def _pair_key(self, row: int, col: int) -> str:
        return f"{int(row)}|{int(col)}"

    def diag_blocks(self) -> list[Tensor]:
        return [
            _lower_triangular_block(raw_block, parametrization=self.diag_parametrization)
            for raw_block in self.raw_diag_blocks
        ]

    def offdiag_blocks(self) -> dict[tuple[int, int], Tensor]:
        return {
            (row, col): self.raw_offdiag_blocks[self._pair_key(row, col)]
            for row, col in self.structure.lower_block_pairs
        }

    def split_vector(self, vector: Tensor) -> dict[str, Tensor]:
        if vector.dim() != 1 or vector.shape[0] != self.total_dim:
            raise ValueError("vector must have shape (total_dim,).")
        return {
            node: vector[self.block_slice(node)]
            for node in self.structure.order
        }

    def split_samples(self, samples: Tensor) -> dict[str, Tensor]:
        if samples.dim() != 2 or samples.shape[1] != self.total_dim:
            raise ValueError("samples must have shape (num_samples, total_dim).")
        return {
            node: samples[:, self.block_slice(node)]
            for node in self.structure.order
        }

    def mean_blocks(self) -> dict[str, Tensor]:
        return self.split_vector(self.q_mean)

    def root_matrix(self) -> Tensor:
        root = self.q_mean.new_zeros((self.total_dim, self.total_dim))
        diag_blocks = self.diag_blocks()
        offdiag_blocks = self.offdiag_blocks()
        for idx, block in enumerate(diag_blocks):
            sl = self.block_slice_by_index(idx)
            root[sl, sl] = block
        for (row, col), block in offdiag_blocks.items():
            root[self.block_slice_by_index(row), self.block_slice_by_index(col)] = block
        return root

    def precision_blocks(self) -> dict[tuple[int, int], Tensor]:
        """Return precision blocks using ``precision_blocks_from_root``."""
        return precision_blocks_from_root(
            structure=self.structure,
            diag_blocks=self.diag_blocks(),
            offdiag_blocks=self.offdiag_blocks(),
        )

    def information_blocks(self) -> tuple[Tensor, ...]:
        """Return sparse information-vector blocks ``Q m``."""
        precision = self.precision_blocks()
        mean = tuple(self.mean_blocks()[node] for node in self.structure.order)
        return precision_vector_product(
            structure=self.structure,
            precision_blocks=precision,
            vector_blocks=mean,
        )

    def logdet_precision(self) -> Tensor:
        total = self.q_mean.new_tensor(0.0)
        for block in self.diag_blocks():
            total = total + 2.0 * torch.log(torch.diagonal(block).abs().clamp_min(1e-12)).sum()
        return total

    def sample_whitened(self, num_samples: int, *, eps: Tensor | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        diag_blocks = self.diag_blocks()
        offdiag_blocks = self.offdiag_blocks()
        if eps is None:
            eps = torch.randn((num_samples, self.total_dim), device=self.q_mean.device, dtype=self.q_mean.dtype)
        else:
            if eps.shape != (num_samples, self.total_dim):
                raise ValueError("eps must have shape (num_samples, total_dim).")
            eps = eps.to(device=self.q_mean.device, dtype=self.q_mean.dtype)

        eps_blocks = [eps[:, self.block_slice_by_index(idx)].transpose(0, 1) for idx in range(len(self.structure.order))]
        solved_blocks: list[Tensor | None] = [None] * len(self.structure.order)
        for idx in range(len(self.structure.order) - 1, -1, -1):
            rhs = eps_blocks[idx]
            for row in self.structure.lower_children[idx]:
                rhs = rhs - offdiag_blocks[(row, idx)].transpose(0, 1) @ solved_blocks[row]
            solved_blocks[idx] = torch.linalg.solve_triangular(diag_blocks[idx].transpose(0, 1), rhs, upper=True)

        samples = torch.cat([block.transpose(0, 1) for block in solved_blocks], dim=1)
        return samples + self.q_mean.unsqueeze(0)

    def covariance_matrix(self) -> Tensor:
        """Return the covariance matrix ``precision^{-1}`` of the global posterior."""
        root = self.root_matrix()
        cov = torch.cholesky_inverse(root, upper=False)
        return 0.5 * (cov + cov.transpose(0, 1))

    def selected_inverse(self) -> dict[tuple[int, int], Tensor]:
        diag_blocks = self.diag_blocks()
        offdiag_blocks = self.offdiag_blocks()
        unit_lower_blocks = {
            (row, col): solve_right_lower_triangular(block, diag_blocks[col])
            for (row, col), block in offdiag_blocks.items()
        }
        e_inv = [
            torch.cholesky_solve(
                torch.eye(block.shape[0], device=block.device, dtype=block.dtype),
                block,
            )
            for block in diag_blocks
        ]

        selected: dict[tuple[int, int], Tensor] = {}

        def get_block(row: int, col: int) -> Tensor:
            if row == col:
                return selected[(row, col)]
            if row > col:
                return selected[(row, col)]
            return selected[(col, row)].transpose(0, 1)

        for idx in range(len(self.structure.order) - 1, -1, -1):
            for row in self.structure.lower_children[idx]:
                acc = None
                for neighbor in self.structure.lower_children[idx]:
                    term = get_block(row, neighbor) @ unit_lower_blocks[(neighbor, idx)]
                    acc = term if acc is None else acc + term
                if acc is None:
                    acc = e_inv[idx].new_zeros((self.block_size_by_index(row), self.block_size_by_index(idx)))
                selected[(row, idx)] = -acc

            diag = e_inv[idx]
            for row in self.structure.lower_children[idx]:
                diag = diag - unit_lower_blocks[(row, idx)].transpose(0, 1) @ selected[(row, idx)]
            selected[(idx, idx)] = 0.5 * (diag + diag.transpose(0, 1))

        return selected

    def statistics(self) -> BlockSparseStatistics:
        selected = self.selected_inverse()
        mean = self.mean_blocks()
        cov = {
            node: selected[(idx, idx)]
            for idx, node in enumerate(self.structure.order)
        }
        total = self.q_mean.new_tensor(0.0)
        for idx in range(len(self.structure.order)):
            total = total + torch.trace(selected[(idx, idx)])
        return BlockSparseStatistics(
            marginals=BlockMarginals(mean_blocks=mean, cov_blocks=cov),
            trace_covariance=total,
            logdet_precision=self.logdet_precision(),
        )

    def marginals(self) -> BlockMarginals:
        return self.statistics().marginals

    def trace_covariance(self) -> Tensor:
        return self.statistics().trace_covariance

    def kl(self) -> Tensor:
        stats = self.statistics()
        return 0.5 * (
            stats.trace_covariance
            + self.q_mean.square().sum()
            - float(self.total_dim)
            + stats.logdet_precision
        )


__all__ = ["BlockMarginals", "BlockSparsePrecisionGMRF", "BlockSparseStatistics"]
