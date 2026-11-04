"""Fusion kernels for combining parent contributions."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import inv_softplus, softplus_positive
from .base import Block, Kernel, LinearKernel, PeriodicKernel, RBFKernel


def _as_blocks(names: Sequence[str], dims: Sequence[int]) -> list[Block]:
    if len(names) != len(dims):
        raise ValueError("names and dims must have the same length.")

    blocks: list[Block] = []
    cursor = 0
    for name, dim in zip(names, dims):
        if dim <= 0:
            raise ValueError("All block dims must be positive.")
        blocks.append(Block(name=name, start=cursor, end=cursor + int(dim)))
        cursor += int(dim)
    return blocks


def _make_base_kernel(name: str, input_dim: int, *, ard: bool) -> Kernel:
    name = name.lower()
    if name in {"linear", "lin"}:
        return LinearKernel(input_dim=input_dim)
    if name == "rbf":
        return RBFKernel(input_dim=input_dim, ard=ard)
    if name in {"periodic", "periodic-rbf", "per"}:
        return PeriodicKernel(input_dim=input_dim, ard=ard)
    raise ValueError(f"Unknown base kernel '{name}'.")


class _ScaledKernel(nn.Module):
    """Apply a positive scalar scale to a base kernel."""

    def __init__(self, base_kernel: Kernel, *, trainable: bool, init_scale: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.base_kernel = base_kernel
        self.input_dim = getattr(base_kernel, "input_dim", None)
        if self.input_dim is None:
            raise ValueError("base_kernel must expose input_dim.")

        self.eps = float(eps)
        init = torch.tensor([float(init_scale)])
        raw = inv_softplus(init)
        if trainable:
            self.raw_scale = nn.Parameter(raw)
        else:
            self.register_buffer("raw_scale", raw)

    @property
    def scale(self) -> Tensor:
        return softplus_positive(self.raw_scale, minimum=self.eps)

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        return self.scale * self.base_kernel.K(X1, X2)

    def Kdiag(self, X: Tensor) -> Tensor:  
        return self.scale * self.base_kernel.Kdiag(X)

class AdditiveFusionKernel(nn.Module):
    """Additive fusion: ``k(x, x') = sum_i w_i k_i(x_i, x'_i)``."""

    def __init__(
        self,
        block_names: Sequence[str],
        block_dims: Sequence[int],
        *,
        base_kernel: str = "rbf",
        ard: bool = True,
        trainable_scales: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = _as_blocks(block_names, block_dims)
        self.input_dim = sum(int(dim) for dim in block_dims)
        self.kernels = nn.ModuleList(
            [
                _ScaledKernel(_make_base_kernel(base_kernel, block.dim, ard=ard), trainable=trainable_scales)
                for block in self.blocks
            ]
        )

    def _slice(self, X: Tensor, block: Block) -> Tensor:
        return X[:, block.start : block.end]

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")

        result: Tensor | None = None
        for kernel, block in zip(self.kernels, self.blocks):
            term = kernel.K(self._slice(X1, block), self._slice(X2, block))
            result = term if result is None else result + term
        assert result is not None
        return result

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")

        result: Tensor | None = None
        for kernel, block in zip(self.kernels, self.blocks):
            term = kernel.Kdiag(self._slice(X, block))
            result = term if result is None else result + term
        assert result is not None
        return result


class ProductFusionKernel(nn.Module):
    """Product fusion: ``k(x, x') = w * prod_i k_i(x_i, x'_i)``."""

    def __init__(
        self,
        block_names: Sequence[str],
        block_dims: Sequence[int],
        *,
        base_kernel: str = "rbf",
        ard: bool = True,
        trainable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = _as_blocks(block_names, block_dims)
        self.input_dim = sum(int(dim) for dim in block_dims)
        self.kernels = nn.ModuleList([_make_base_kernel(base_kernel, block.dim, ard=ard) for block in self.blocks])
        self._scale_eps = 1e-6

        init = torch.tensor([1.0])
        raw = inv_softplus(init)
        if trainable_scale:
            self.raw_scale = nn.Parameter(raw)
        else:
            self.register_buffer("raw_scale", raw)

    @property
    def scale(self) -> Tensor:
        return softplus_positive(self.raw_scale, minimum=self._scale_eps)

    def _slice(self, X: Tensor, block: Block) -> Tensor:
        return X[:, block.start : block.end]

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")

        result: Tensor | None = None
        for kernel, block in zip(self.kernels, self.blocks):
            term = kernel.K(self._slice(X1, block), self._slice(X2, block))
            result = term if result is None else result * term
        assert result is not None
        return self.scale * result

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")

        result: Tensor | None = None
        for kernel, block in zip(self.kernels, self.blocks):
            term = kernel.Kdiag(self._slice(X, block))
            result = term if result is None else result * term
        assert result is not None
        return self.scale * result


class ConcatFusionKernel(nn.Module):
    """Concatenation fusion: apply one base kernel to the full input."""

    def __init__(
        self,
        input_dim: int,
        *,
        base_kernel: str = "rbf",
        ard: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.kernel = _make_base_kernel(base_kernel, self.input_dim, ard=ard)

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        return self.kernel.K(X1, X2)

    def Kdiag(self, X: Tensor) -> Tensor:  
        return self.kernel.Kdiag(X)


class ANOVA2FusionKernel(nn.Module):
    """Main effects plus selected two-way interactions."""

    def __init__(
        self,
        block_names: Sequence[str],
        block_dims: Sequence[int],
        interactions: Iterable[tuple[str, str]],
        *,
        base_kernel: str = "rbf",
        ard: bool = True,
        trainable_scales: bool = True,
    ) -> None:
        super().__init__()

        self.blocks = _as_blocks(block_names, block_dims)
        self.input_dim = sum(int(dim) for dim in block_dims)

        self.main_kernels = nn.ModuleList(
            [
                _ScaledKernel(_make_base_kernel(base_kernel, block.dim, ard=ard), trainable=trainable_scales)
                for block in self.blocks
            ]
        )

        name_to_idx = {block.name: idx for idx, block in enumerate(self.blocks)}
        interaction_indices: list[tuple[int, int]] = []
        for left_name, right_name in interactions:
            if left_name not in name_to_idx or right_name not in name_to_idx:
                raise ValueError(f"Unknown interaction ({left_name}, {right_name}).")
            left_idx, right_idx = name_to_idx[left_name], name_to_idx[right_name]
            if left_idx == right_idx:
                raise ValueError("Self-interactions are not supported.")
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx
            interaction_indices.append((left_idx, right_idx))
        self.interactions = tuple(sorted(set(interaction_indices)))

        self._scale_eps = 1e-6
        if len(self.interactions) == 0:
            self.register_buffer("raw_interaction_scales", torch.empty(0))
        else:
            init = torch.full((len(self.interactions),), 1.0)
            raw = inv_softplus(init)
            if trainable_scales:
                self.raw_interaction_scales = nn.Parameter(raw)
            else:
                self.register_buffer("raw_interaction_scales", raw)

    def _slice(self, X: Tensor, block: Block) -> Tensor:
        return X[:, block.start : block.end]

    def _interaction_scale(self, idx: int) -> Tensor:
        return softplus_positive(self.raw_interaction_scales[idx], minimum=self._scale_eps)

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        if X1.shape[1] != self.input_dim or X2.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")

        block_X1 = [self._slice(X1, block) for block in self.blocks]
        block_X2 = [self._slice(X2, block) for block in self.blocks]

        result: Tensor | None = None
        for kernel, x1, x2 in zip(self.main_kernels, block_X1, block_X2):
            term = kernel.K(x1, x2)
            result = term if result is None else result + term
        assert result is not None

        for idx, (left_idx, right_idx) in enumerate(self.interactions):
            scale = self._interaction_scale(idx)
            left_kernel = self.main_kernels[left_idx].base_kernel
            right_kernel = self.main_kernels[right_idx].base_kernel
            term = (
                scale
                * left_kernel.K(block_X1[left_idx], block_X2[left_idx])
                * right_kernel.K(block_X1[right_idx], block_X2[right_idx])
            )
            result = result + term
        return result

    def Kdiag(self, X: Tensor) -> Tensor:  
        if X.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")

        blocks = [self._slice(X, block) for block in self.blocks]

        result: Tensor | None = None
        for kernel, block_values in zip(self.main_kernels, blocks):
            term = kernel.Kdiag(block_values)
            result = term if result is None else result + term
        assert result is not None

        for idx, (left_idx, right_idx) in enumerate(self.interactions):
            scale = self._interaction_scale(idx)
            left_kernel = self.main_kernels[left_idx].base_kernel
            right_kernel = self.main_kernels[right_idx].base_kernel
            term = scale * left_kernel.Kdiag(blocks[left_idx]) * right_kernel.Kdiag(blocks[right_idx])
            result = result + term
        return result


class MultifidelityFusionKernel(nn.Module):
    """Nonlinear multi-fidelity fusion kernel."""

    def __init__(
        self,
        block_names: Sequence[str],
        block_dims: Sequence[int],
        *,
        input_blocks: Sequence[str],
        ard: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = _as_blocks(block_names, block_dims)
        self.input_dim = sum(int(dim) for dim in block_dims)

        if len(input_blocks) == 0:
            raise ValueError("MultifidelityFusionKernel requires at least one input block.")

        name_to_block = {block.name: block for block in self.blocks}
        unknown = sorted(set(input_blocks) - set(name_to_block))
        if unknown:
            raise ValueError(f"Unknown multifidelity input block(s): {unknown}.")

        self.input_block_names = tuple(input_blocks)
        self.input_blocks = tuple(name_to_block[name] for name in self.input_block_names)
        self.latent_blocks = tuple(block for block in self.blocks if block.name not in set(self.input_block_names))
        if len(self.latent_blocks) == 0:
            raise ValueError("MultifidelityFusionKernel requires at least one latent parent block.")

        x_dim = sum(block.dim for block in self.input_blocks)
        z_dim = sum(block.dim for block in self.latent_blocks)
        self.k_rho_x = RBFKernel(input_dim=x_dim, ard=ard)
        self.k_delta_x = RBFKernel(input_dim=x_dim, ard=ard)
        self.k_lin_z = LinearKernel(input_dim=z_dim)
        self.k_se_z = RBFKernel(input_dim=z_dim, ard=ard)

    def _select_blocks(self, X: Tensor, blocks: Sequence[Block]) -> Tensor:
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch.")
        if len(blocks) == 1:
            block = blocks[0]
            return X[:, block.start : block.end]
        return torch.cat([X[:, block.start : block.end] for block in blocks], dim=-1)

    def K(self, X1: Tensor, X2: Tensor) -> Tensor:  
        x1 = self._select_blocks(X1, self.input_blocks)
        x2 = self._select_blocks(X2, self.input_blocks)
        z1 = self._select_blocks(X1, self.latent_blocks)
        z2 = self._select_blocks(X2, self.latent_blocks)
        return self.k_rho_x.K(x1, x2) * (self.k_lin_z.K(z1, z2) + self.k_se_z.K(z1, z2)) + self.k_delta_x.K(x1, x2)

    def Kdiag(self, X: Tensor) -> Tensor:  
        x = self._select_blocks(X, self.input_blocks)
        z = self._select_blocks(X, self.latent_blocks)
        return self.k_rho_x.Kdiag(x) * (self.k_lin_z.Kdiag(z) + self.k_se_z.Kdiag(z)) + self.k_delta_x.Kdiag(x)
