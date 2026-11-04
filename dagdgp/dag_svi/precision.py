"""Block precision utilities for DAG-SVI."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from torch import Tensor

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


def _infer_batch_shape_from_vectors(vectors: Iterable[Tensor]) -> tuple[int, ...]:
    batch_shape: tuple[int, ...] = ()
    for vector in vectors:
        if vector.dim() < 1:
            raise ValueError("Vector tensors must have rank >= 1.")
        current = tuple(vector.shape[:-1])
        if len(current) > len(batch_shape):
            batch_shape = current
    for vector in vectors:
        current = tuple(vector.shape[:-1])
        if current not in {(), batch_shape}:
            raise ValueError("All vector tensors must share a common batch shape.")
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


def precision_blocks_from_root(
    *,
    structure: ChordalStructure,
    diag_blocks: Sequence[Tensor],
    offdiag_blocks: dict[tuple[int, int], Tensor],
) -> dict[tuple[int, int], Tensor]:
    """Materialise precision blocks ``Q = L L^T`` on the chordal graph."""
    if len(diag_blocks) != len(structure.order):
        raise ValueError("diag_blocks must match the number of latent blocks.")
    batch_shape = _infer_batch_shape_from_blocks(list(diag_blocks) + list(offdiag_blocks.values()))
    block_sizes = tuple(int(block.shape[-1]) for block in diag_blocks)

    blocks: dict[tuple[int, int], Tensor] = {}
    for idx, size in enumerate(block_sizes):
        shape = batch_shape + (size, size)
        blocks[(idx, idx)] = diag_blocks[idx].new_zeros(shape)
    for row, col in structure.lower_block_pairs:
        shape = batch_shape + (block_sizes[row], block_sizes[col])
        blocks[(row, col)] = diag_blocks[0].new_zeros(shape)

    for col in range(len(structure.order)):
        column_entries: list[tuple[int, Tensor]] = [(col, _expand_block(diag_blocks[col], batch_shape))]
        column_entries.extend((row, _expand_block(offdiag_blocks[(row, col)], batch_shape)) for row in structure.lower_children[col])
        for left_idx, (left, left_block) in enumerate(column_entries):
            for right, right_block in column_entries[: left_idx + 1]:
                if left >= right:
                    key = (left, right)
                    term = left_block @ right_block.transpose(-1, -2)
                else:
                    key = (right, left)
                    term = right_block @ left_block.transpose(-1, -2)
                blocks[key] = blocks[key] + term

    for idx in range(len(structure.order)):
        blocks[(idx, idx)] = _symmetrise(blocks[(idx, idx)])
    return blocks


def precision_vector_product(
    *,
    structure: ChordalStructure,
    precision_blocks: dict[tuple[int, int], Tensor],
    vector_blocks: Sequence[Tensor],
) -> tuple[Tensor, ...]:
    """Multiply a block precision matrix by block vectors."""
    if len(vector_blocks) != len(structure.order):
        raise ValueError("vector_blocks must match the number of latent blocks.")
    batch_shape = _infer_batch_shape_from_vectors(vector_blocks)
    outputs = []
    for idx in range(len(structure.order)):
        diag = _expand_block(precision_blocks[(idx, idx)], batch_shape)
        vec = _expand_vector(vector_blocks[idx], batch_shape)
        outputs.append((diag @ vec.unsqueeze(-1)).squeeze(-1))

    for row, col in structure.lower_block_pairs:
        block = _expand_block(precision_blocks[(row, col)], batch_shape)
        col_vec = _expand_vector(vector_blocks[col], batch_shape)
        row_vec = _expand_vector(vector_blocks[row], batch_shape)
        outputs[row] = outputs[row] + (block @ col_vec.unsqueeze(-1)).squeeze(-1)
        outputs[col] = outputs[col] + (block.transpose(-1, -2) @ row_vec.unsqueeze(-1)).squeeze(-1)
    return tuple(outputs)


__all__ = ["precision_blocks_from_root", "precision_vector_product"]
