"""Clique-aware sparse Gaussian backend for DAG-SVI marginalisation.

These routines operate directly on the fixed chordal sparsity pattern of the
structured posterior. They avoid materialising the dense global covariance by:

1. assembling sparse precision blocks from the posterior Cholesky root,
2. applying diagonal site updates induced by conditioned latent draws,
3. refactorising the updated precision with a block sparse Cholesky pass, and
4. recovering the selected inverse blocks needed for predictive conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from ..gaussian import solve_lower_triangular, solve_right_lower_triangular, solve_upper_triangular, stable_cholesky
from .graph import ChordalStructure


def _symmetrise(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _infer_batch_shape_from_blocks(blocks: Iterable[Tensor]) -> tuple[int, ...]:
    batch_shape: tuple[int, ...] = ()
    for block in blocks:
        if block.dim() < 2:
            raise ValueError("Block tensors must have rank >= 2.")
        current = tuple(block.shape[:-2])
        if len(current) > len(batch_shape):
            batch_shape = current
    for block in blocks:
        current = tuple(block.shape[:-2])
        if current not in {(), batch_shape}:
            raise ValueError("All block tensors must share a common batch shape.")
    return batch_shape


def _expand_block(block: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    if tuple(block.shape[:-2]) == batch_shape:
        return block
    if block.dim() != 2:
        raise ValueError("Unbatched blocks must have rank 2.")
    if not batch_shape:
        return block
    return block.reshape((1,) * len(batch_shape) + tuple(block.shape)).expand(batch_shape + tuple(block.shape))


def _expand_vector(vector: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    if tuple(vector.shape[:-1]) == batch_shape:
        return vector
    if vector.dim() != 1:
        raise ValueError("Unbatched vectors must have rank 1.")
    if not batch_shape:
        return vector
    return vector.reshape((1,) * len(batch_shape) + tuple(vector.shape)).expand(batch_shape + tuple(vector.shape))


def _expand_rhs(rhs: Tensor, batch_shape: tuple[int, ...], block_size: int) -> Tensor:
    current = tuple(rhs.shape[:-1]) if rhs.dim() >= 1 else ()
    if rhs.dim() >= 2 and int(rhs.shape[-2]) == block_size:
        current = tuple(rhs.shape[:-2])
        if current == batch_shape:
            return rhs
        if current == ():
            if not batch_shape:
                return rhs
            return rhs.reshape((1,) * len(batch_shape) + tuple(rhs.shape)).expand(batch_shape + tuple(rhs.shape))
    if rhs.dim() == 1 and int(rhs.shape[0]) == block_size:
        if not batch_shape:
            return rhs
        return rhs.reshape((1,) * len(batch_shape) + tuple(rhs.shape)).expand(batch_shape + tuple(rhs.shape))
    if rhs.dim() >= 2 and int(rhs.shape[-2]) == block_size:
        raise ValueError("All RHS tensors must share a common batch shape.")
    raise ValueError("RHS tensor has incompatible shape for the given block size.")


def _block_matmul(block: Tensor, rhs: Tensor) -> Tensor:
    if rhs.dim() == block.dim() - 1:
        return (block @ rhs.unsqueeze(-1)).squeeze(-1)
    return block @ rhs


def _batched_stable_cholesky(matrix: Tensor, *, jitter: float) -> Tensor:
    """Robust Cholesky for tensors with shape ``(..., n, n)``."""
    matrix = _symmetrise(matrix)
    if matrix.dim() == 2:
        return stable_cholesky(matrix, jitter=jitter)

    n = int(matrix.shape[-1])
    flat = matrix.reshape(-1, n, n)
    eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    base_jitter = max(float(jitter), 0.0)
    if base_jitter == 0.0:
        base_jitter = 1e-9
    try:
        chol = torch.linalg.cholesky(flat + base_jitter * eye)
    except RuntimeError:
        chols = [stable_cholesky(candidate, jitter=base_jitter) for candidate in flat]
        chol = torch.stack(chols, dim=0)
    return chol.reshape(matrix.shape)


def _row_to_previous_neighbors(structure: ChordalStructure) -> tuple[tuple[int, ...], ...]:
    incoming: list[list[int]] = [[] for _ in structure.order]
    for row, col in structure.lower_block_pairs:
        incoming[row].append(col)
    return tuple(tuple(sorted(cols)) for cols in incoming)


def _uniform_block_size(block_sizes: Sequence[int], *, start_index: int = 0) -> int | None:
    if start_index >= len(block_sizes):
        return None
    sizes = tuple(int(size) for size in block_sizes[start_index:])
    if not sizes:
        return None
    first = sizes[0]
    if any(size != first for size in sizes[1:]):
        return None
    return first


def _solve_suffix_grouped_uniform(
    factor: "BlockSparseCholeskyFactor",
    rhs_blocks: Sequence[Tensor],
    *,
    start_index: int,
) -> tuple[Tensor, ...]:
    """Grouped suffix solve for uniform block sizes.

    This path batches rows that become available at the same elimination level
    and have the same predecessor/child arity. Grouping compatible blocks
    reduces the number of separate CUDA operations.
    """
    start = int(start_index)
    num_blocks = len(factor.structure.order)
    if start == num_blocks:
        return tuple()

    batch_shape = factor.batch_shape
    if len(batch_shape) != 1:
        raise ValueError("Grouped suffix solve expects a single batch dimension.")

    expanded_rhs = [
        _expand_rhs(rhs_blocks[idx - start], batch_shape, factor.block_sizes[idx])
        for idx in range(start, num_blocks)
    ]
    squeeze_last = expanded_rhs[0].dim() == len(batch_shape) + 1
    normalised_rhs = [rhs.unsqueeze(-1) if squeeze_last else rhs for rhs in expanded_rhs]

    levels: list[int | None] = [None] * num_blocks
    for idx in range(start, num_blocks):
        level = 0
        for prev in factor.incoming_neighbors[idx]:
            if prev < start:
                continue
            prev_level = levels[prev]
            if prev_level is not None and prev_level + 1 > level:
                level = prev_level + 1
        levels[idx] = level
    max_level = max(int(levels[idx]) for idx in range(start, num_blocks))

    forward: list[Tensor | None] = [None] * num_blocks
    for level in range(max_level + 1):
        rows = [idx for idx in range(start, num_blocks) if levels[idx] == level]
        grouped: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for row in rows:
            prevs = tuple(prev for prev in factor.incoming_neighbors[row] if prev >= start)
            grouped.setdefault(len(prevs), []).append((row, prevs))

        for arity, entries in grouped.items():
            row_group = [row for row, _ in entries]
            rhs = torch.stack([normalised_rhs[row - start] for row in row_group], dim=1)
            if arity:
                edge_blocks = torch.stack(
                    [
                        torch.stack([factor.offdiag_blocks[(row, prev)] for prev in prevs], dim=1)
                        for row, prevs in entries
                    ],
                    dim=1,
                )
                prev_forward = torch.stack(
                    [
                        torch.stack([forward[prev] for prev in prevs], dim=1)
                        for _, prevs in entries
                    ],
                    dim=1,
                )
                rhs = rhs - torch.einsum("bgaij,bgajr->bgir", edge_blocks, prev_forward)

            diag = torch.stack([factor.diag_blocks[row] for row in row_group], dim=1)
            solved = solve_lower_triangular(diag, rhs)
            for pos, row in enumerate(row_group):
                forward[row] = solved[:, pos]

    reverse_levels: list[int | None] = [None] * num_blocks
    for idx in range(num_blocks - 1, start - 1, -1):
        level = 0
        for child in factor.structure.lower_children[idx]:
            if child < start:
                continue
            child_level = reverse_levels[child]
            if child_level is not None and child_level + 1 > level:
                level = child_level + 1
        reverse_levels[idx] = level
    max_reverse_level = max(int(reverse_levels[idx]) for idx in range(start, num_blocks))

    solution: list[Tensor | None] = [None] * num_blocks
    for level in range(max_reverse_level + 1):
        rows = [idx for idx in range(start, num_blocks) if reverse_levels[idx] == level]
        grouped: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for row in rows:
            children = tuple(child for child in factor.structure.lower_children[row] if child >= start)
            grouped.setdefault(len(children), []).append((row, children))

        for arity, entries in grouped.items():
            row_group = [row for row, _ in entries]
            rhs = torch.stack([forward[row] for row in row_group], dim=1)
            if arity:
                edge_blocks = torch.stack(
                    [
                        torch.stack([factor.offdiag_blocks[(child, row)] for child in children], dim=1)
                        for row, children in entries
                    ],
                    dim=1,
                )
                child_solution = torch.stack(
                    [
                        torch.stack([solution[child] for child in children], dim=1)
                        for _, children in entries
                    ],
                    dim=1,
                )
                rhs = rhs - torch.einsum(
                    "bgaij,bgajr->bgir",
                    edge_blocks.transpose(-1, -2),
                    child_solution,
                )

            diag = torch.stack([factor.diag_blocks[row].transpose(-1, -2) for row in row_group], dim=1)
            solved = solve_upper_triangular(diag, rhs)
            for pos, row in enumerate(row_group):
                solution[row] = solved[:, pos]

    suffix_solution = tuple(solution[idx] for idx in range(start, num_blocks))
    if squeeze_last:
        return tuple(block.squeeze(-1) for block in suffix_solution)
    return suffix_solution


@dataclass(frozen=True)
class BlockSparseCholeskyFactor:
    """Block sparse lower Cholesky factor on a fixed chordal graph."""

    structure: ChordalStructure
    block_sizes: tuple[int, ...]
    diag_blocks: tuple[Tensor, ...]
    offdiag_blocks: dict[tuple[int, int], Tensor]
    batch_shape: tuple[int, ...]
    incoming_neighbors: tuple[tuple[int, ...], ...]

    def solve(self, rhs_blocks: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Solve ``(L L^T) x = rhs`` blockwise for batched right-hand sides."""
        if len(rhs_blocks) != len(self.structure.order):
            raise ValueError("rhs_blocks must match the number of latent blocks.")
        batch_shape = self.batch_shape

        forward: list[Tensor] = [None] * len(self.structure.order)  
        for idx in range(len(self.structure.order)):
            rhs = _expand_rhs(rhs_blocks[idx], batch_shape, self.block_sizes[idx])
            for prev in self.incoming_neighbors[idx]:
                term = _block_matmul(self.offdiag_blocks[(idx, prev)], forward[prev])
                rhs = rhs - term
            solved = solve_lower_triangular(self.diag_blocks[idx], rhs)
            forward[idx] = solved

        solution: list[Tensor] = [None] * len(self.structure.order)  
        for idx in range(len(self.structure.order) - 1, -1, -1):
            rhs = forward[idx]
            for row in self.structure.lower_children[idx]:
                term = _block_matmul(self.offdiag_blocks[(row, idx)].transpose(-1, -2), solution[row])
                rhs = rhs - term
            solved = solve_upper_triangular(self.diag_blocks[idx].transpose(-1, -2), rhs)
            solution[idx] = solved
        return tuple(solution)

    def solve_suffix(self, rhs_blocks: Sequence[Tensor], *, start_index: int) -> tuple[Tensor, ...]:
        """Solve exactly on a suffix when the RHS is zero on the prefix."""
        start = int(start_index)
        if start < 0 or start > len(self.structure.order):
            raise ValueError("start_index must be between 0 and the number of latent blocks.")
        if len(rhs_blocks) != len(self.structure.order) - start:
            raise ValueError("rhs_blocks must match the suffix size starting at start_index.")
        if start == len(self.structure.order):
            return tuple()

        block_size = _uniform_block_size(self.block_sizes, start_index=start)
        if (
            block_size is not None
            and len(self.batch_shape) == 1
            and self.diag_blocks[start].device.type == "cuda"
        ):
            return _solve_suffix_grouped_uniform(self, rhs_blocks, start_index=start)

        batch_shape = self.batch_shape
        forward: list[Tensor | None] = [None] * len(self.structure.order)
        for idx in range(start, len(self.structure.order)):
            rhs = _expand_rhs(rhs_blocks[idx - start], batch_shape, self.block_sizes[idx])
            for prev in self.incoming_neighbors[idx]:
                if prev < start:
                    continue
                prev_forward = forward[prev]
                if prev_forward is None:
                    raise RuntimeError("Missing forward state for suffix sparse solve.")
                rhs = rhs - _block_matmul(self.offdiag_blocks[(idx, prev)], prev_forward)
            forward[idx] = solve_lower_triangular(self.diag_blocks[idx], rhs)

        solution: list[Tensor | None] = [None] * len(self.structure.order)
        for idx in range(len(self.structure.order) - 1, start - 1, -1):
            rhs = forward[idx]
            if rhs is None:
                raise RuntimeError("Missing forward state for suffix sparse solve.")
            for row in self.structure.lower_children[idx]:
                if row < start:
                    continue
                row_solution = solution[row]
                if row_solution is None:
                    raise RuntimeError("Missing backward state for suffix sparse solve.")
                rhs = rhs - _block_matmul(self.offdiag_blocks[(row, idx)].transpose(-1, -2), row_solution)
            solution[idx] = solve_upper_triangular(self.diag_blocks[idx].transpose(-1, -2), rhs)

        return tuple(solution[idx] for idx in range(start, len(self.structure.order)))

    def selected_inverse(self) -> dict[tuple[int, int], Tensor]:
        """Return selected inverse blocks for diagonals and graph edges."""
        unit_lower_blocks = {
            (row, col): solve_right_lower_triangular(self.offdiag_blocks[(row, col)], self.diag_blocks[col])
            for (row, col) in self.structure.lower_block_pairs
        }
        eye_cache = [
            torch.eye(size, device=self.diag_blocks[idx].device, dtype=self.diag_blocks[idx].dtype)
            for idx, size in enumerate(self.block_sizes)
        ]
        e_inv = [
            torch.cholesky_solve(_expand_block(eye_cache[idx], self.batch_shape), self.diag_blocks[idx])
            for idx in range(len(self.structure.order))
        ]

        selected: dict[tuple[int, int], Tensor] = {}

        def get_block(row: int, col: int) -> Tensor:
            if row == col:
                return selected[(row, col)]
            if row > col:
                return selected[(row, col)]
            return selected[(col, row)].transpose(-1, -2)

        for idx in range(len(self.structure.order) - 1, -1, -1):
            for row in self.structure.lower_children[idx]:
                acc = None
                for neighbor in self.structure.lower_children[idx]:
                    term = get_block(row, neighbor) @ unit_lower_blocks[(neighbor, idx)]
                    acc = term if acc is None else acc + term
                if acc is None:
                    shape = self.batch_shape + (self.block_sizes[row], self.block_sizes[idx])
                    acc = self.diag_blocks[idx].new_zeros(shape)
                selected[(row, idx)] = -acc

            diag = e_inv[idx]
            for row in self.structure.lower_children[idx]:
                diag = diag - unit_lower_blocks[(row, idx)].transpose(-1, -2) @ selected[(row, idx)]
            selected[(idx, idx)] = _symmetrise(diag)

        return selected


def diagonal_rank1_update_(
    factor: BlockSparseCholeskyFactor,
    *,
    block_index: int,
    update_vector: Tensor,
) -> BlockSparseCholeskyFactor:
    """Apply an exact rank-1 update ``A <- A + u u^T``.

    ``update_vector`` is the local block of ``u`` at ``block_index``; all other
    blocks of ``u`` are initially zero. When gradients are disabled we update
    the factor in place via a QR-based sweep for speed; otherwise we fall back
    to a differentiable functional cholupdate.
    """
    start = int(block_index)
    if start < 0 or start >= len(factor.structure.order):
        raise ValueError("block_index must be between 0 and the number of latent blocks.")

    batch_shape = factor.batch_shape

    if not torch.is_grad_enabled():
        if factor.diag_blocks[start].device.type == "cuda":
            update_blocks: dict[int, Tensor] = {
                start: _expand_vector(update_vector, batch_shape).clone(),
            }

            for idx in range(start, len(factor.structure.order)):
                current_update = update_blocks.pop(idx, None)
                if current_update is None:
                    continue

                diag_block = factor.diag_blocks[idx]
                block_size = int(factor.block_sizes[idx])
                stacked = torch.cat([diag_block.transpose(-1, -2), current_update.unsqueeze(-2)], dim=-2)
                qr_compact, tau = torch.geqrf(stacked)

                r_top = qr_compact[..., :block_size, :]
                diag_entries = torch.diagonal(r_top, dim1=-2, dim2=-1)
                signs = torch.where(diag_entries >= 0, torch.ones_like(diag_entries), -torch.ones_like(diag_entries))
                full_signs = torch.cat(
                    [
                        signs,
                        torch.ones(signs.shape[:-1] + (1,), device=signs.device, dtype=signs.dtype),
                    ],
                    dim=-1,
                )
                diag_block.copy_(torch.tril((signs.unsqueeze(-1) * r_top).transpose(-1, -2)))

                for row in factor.structure.lower_children[idx]:
                    row_block = factor.offdiag_blocks[(row, idx)]
                    row_update = update_blocks.get(row)
                    if row_update is None:
                        row_update = row_block.new_zeros(batch_shape + (row_block.shape[-2],))
                    augmented = torch.cat([row_block, row_update.unsqueeze(-1)], dim=-1)
                    transformed = torch.ormqr(qr_compact, tau, augmented, left=False, transpose=False)
                    transformed = transformed * full_signs.unsqueeze(-2)
                    row_block.copy_(transformed[..., :block_size])
                    update_blocks[row] = transformed[..., block_size]

            return factor

        update_blocks: dict[int, Tensor] = {
            start: _expand_vector(update_vector, batch_shape).clone(),
        }

        for idx in range(start, len(factor.structure.order)):
            current_update = update_blocks.pop(idx, None)
            if current_update is None:
                continue

            diag_block = factor.diag_blocks[idx]
            block_size = int(factor.block_sizes[idx])
            stacked = torch.cat([diag_block.transpose(-1, -2), current_update.unsqueeze(-2)], dim=-2)
            q_factor, r_factor = torch.linalg.qr(stacked, mode="complete")

            diag_entries = torch.diagonal(r_factor[..., :block_size, :], dim1=-2, dim2=-1)
            signs = torch.where(diag_entries >= 0, torch.ones_like(diag_entries), -torch.ones_like(diag_entries))
            sign_column = torch.ones(
                signs.shape[:-1] + (1,),
                device=signs.device,
                dtype=signs.dtype,
            )
            full_signs = torch.cat([signs, sign_column], dim=-1)
            q_factor = q_factor * full_signs.unsqueeze(-2)
            r_factor = full_signs.unsqueeze(-1) * r_factor

            new_diag = torch.tril(r_factor[..., :block_size, :].transpose(-1, -2))
            diag_block.copy_(new_diag)

            for row in factor.structure.lower_children[idx]:
                row_block = factor.offdiag_blocks[(row, idx)]
                row_update = update_blocks.get(row)
                if row_update is None:
                    row_update = row_block.new_zeros(batch_shape + (row_block.shape[-2],))
                augmented = torch.cat([row_block, row_update.unsqueeze(-1)], dim=-1)
                transformed = augmented @ q_factor
                row_block.copy_(transformed[..., :block_size])
                update_blocks[row] = transformed[..., block_size]

        return factor

    diag_blocks = list(factor.diag_blocks)
    offdiag_blocks = dict(factor.offdiag_blocks)
    update_blocks: dict[int, Tensor] = {
        start: _expand_vector(update_vector, batch_shape).clone(),
    }

    for idx in range(start, len(factor.structure.order)):
        current_update = update_blocks.pop(idx, None)
        if current_update is None:
            continue

        diag_work = diag_blocks[idx].clone()
        block_size = int(factor.block_sizes[idx])
        update_tail = current_update.clone()
        child_updates: dict[int, Tensor] = {}
        child_blocks = {
            row: offdiag_blocks[(row, idx)].clone()
            for row in factor.structure.lower_children[idx]
        }

        for col in range(block_size):
            diag_entry = diag_work[..., col, col].clone()
            update_entry = update_tail[..., col]
            radius = torch.sqrt(diag_entry.square() + update_entry.square())
            c = radius / diag_entry
            s = update_entry / diag_entry
            diag_work[..., col, col] = radius

            if col + 1 < block_size:
                block_tail = diag_work[..., col + 1 :, col].clone()
                update_vector_tail = update_tail[..., col + 1 :]
                new_block_tail = (block_tail + s.unsqueeze(-1) * update_vector_tail) / c.unsqueeze(-1)
                diag_work[..., col + 1 :, col] = new_block_tail
                new_update_tail = update_tail.clone()
                new_update_tail[..., col + 1 :] = (
                    c.unsqueeze(-1) * update_vector_tail - s.unsqueeze(-1) * new_block_tail
                )
                update_tail = new_update_tail

            for row in factor.structure.lower_children[idx]:
                row_block = child_blocks[row]
                row_update = child_updates.get(row, update_blocks.get(row))
                if row_update is None:
                    row_update = row_block.new_zeros(batch_shape + (row_block.shape[-2],))
                row_col = row_block[..., :, col].clone()
                new_row_col = (row_col + s.unsqueeze(-1) * row_update) / c.unsqueeze(-1)
                row_block[..., :, col] = new_row_col
                child_updates[row] = c.unsqueeze(-1) * row_update - s.unsqueeze(-1) * new_row_col

        diag_blocks[idx] = diag_work
        for row, row_block in child_blocks.items():
            offdiag_blocks[(row, idx)] = row_block
            update_blocks[row] = child_updates[row]

    return BlockSparseCholeskyFactor(
        structure=factor.structure,
        block_sizes=factor.block_sizes,
        diag_blocks=tuple(diag_blocks),
        offdiag_blocks=offdiag_blocks,
        batch_shape=factor.batch_shape,
        incoming_neighbors=factor.incoming_neighbors,
    )

def factorize_precision_blocks(
    *,
    structure: ChordalStructure,
    block_sizes: Sequence[int],
    diag_precision_blocks: Sequence[Tensor] | None = None,
    offdiag_precision_blocks: dict[tuple[int, int], Tensor] | None = None,
    precision_blocks: dict[tuple[int, int], Tensor] | None = None,
    jitter: float = 1e-9,
) -> BlockSparseCholeskyFactor:
    """Block sparse Cholesky factorisation following the perfect elimination order."""
    if precision_blocks is not None:
        if diag_precision_blocks is not None or offdiag_precision_blocks is not None:
            raise ValueError("Pass either precision_blocks or split diag/offdiag blocks, not both.")
        diag_precision_blocks = tuple(precision_blocks[(idx, idx)] for idx in range(len(structure.order)))
        offdiag_precision_blocks = {
            (row, col): precision_blocks[(row, col)]
            for (row, col) in structure.lower_block_pairs
        }
    if diag_precision_blocks is None or offdiag_precision_blocks is None:
        raise ValueError("Missing precision blocks for sparse factorization.")
    if len(diag_precision_blocks) != len(structure.order):
        raise ValueError("diag_precision_blocks must match the number of latent blocks.")

    block_sizes_tuple = tuple(int(size) for size in block_sizes)
    batch_shape = _infer_batch_shape_from_blocks(tuple(diag_precision_blocks) + tuple(offdiag_precision_blocks.values()))
    work = {
        (idx, idx): _expand_block(block, batch_shape).clone()
        for idx, block in enumerate(diag_precision_blocks)
    }
    for key, block in offdiag_precision_blocks.items():
        work[key] = _expand_block(block, batch_shape).clone()

    diag_factors: list[Tensor | None] = [None] * len(structure.order)
    offdiag_factors: dict[tuple[int, int], Tensor] = {}

    for idx in range(len(structure.order)):
        current_diag = _symmetrise(work[(idx, idx)])
        diag_factor = _batched_stable_cholesky(current_diag, jitter=jitter)
        diag_factors[idx] = diag_factor

        children = structure.lower_children[idx]
        for row in children:
            factor = solve_lower_triangular(diag_factor, work[(row, idx)].transpose(-1, -2)).transpose(-1, -2)
            offdiag_factors[(row, idx)] = factor

        for left_pos, row in enumerate(children):
            row_factor = offdiag_factors[(row, idx)]
            work[(row, row)] = _symmetrise(work[(row, row)] - row_factor @ row_factor.transpose(-1, -2))
            for col in children[:left_pos]:
                col_factor = offdiag_factors[(col, idx)]
                work[(row, col)] = work[(row, col)] - row_factor @ col_factor.transpose(-1, -2)

    if any(block is None for block in diag_factors):
        raise RuntimeError("Sparse Cholesky factorization did not fill all diagonal blocks.")

    return BlockSparseCholeskyFactor(
        structure=structure,
        block_sizes=block_sizes_tuple,
        diag_blocks=tuple(block for block in diag_factors if block is not None),
        offdiag_blocks=offdiag_factors,
        batch_shape=batch_shape,
        incoming_neighbors=_row_to_previous_neighbors(structure),
    )


__all__ = [
    "BlockSparseCholeskyFactor",
    "diagonal_rank1_update_",
    "factorize_precision_blocks",
]
