"""DAG-SVI ELBO and latent-sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..dag_helpers import prepare_dataset, sampled_node_inputs
from ..data import DAGDataset
from ..gaussian import solve_lower_triangular, solve_upper_triangular
from ..likelihoods import GaussianLikelihood
from .graph import ChordalStructure
from .sparse_backend import BlockSparseCholeskyFactor, diagonal_rank1_update_, factorize_precision_blocks


@dataclass(frozen=True)
class _DensePosteriorMoments:
    """Dense moments of the global whitened inducing posterior."""

    mean_blocks: dict[str, Tensor]
    covariance: Tensor
    slices: dict[str, slice]
    trace_covariance: Tensor
    logdet_precision: Tensor

    def cov_block(self, left: str, right: str) -> Tensor:
        return self.covariance[self.slices[left], self.slices[right]]


@dataclass(frozen=True)
class _SparsePosteriorBase:
    """Sparse canonical form of the global whitened inducing posterior."""

    components: tuple["_SparseComponentBase", ...]
    node_to_component: tuple[int, ...]
    trace_covariance: Tensor
    logdet_precision: Tensor


@dataclass
class _SparseCaseCache:
    """Cached conditioning state for each connected component and data point."""

    num_cases: int
    components: list["_SparseComponentCaseCache | None"]


@dataclass(frozen=True)
class _SparseComponentBase:
    """Sparse posterior data restricted to one connected chordal component."""

    global_indices: tuple[int, ...]
    global_to_local: dict[int, int]
    structure: ChordalStructure
    block_sizes: tuple[int, ...]
    precision_blocks: dict[tuple[int, int], Tensor]
    mean_blocks: tuple[Tensor, ...]
    marginal_covariance_blocks: tuple[Tensor, ...]
    base_factor: BlockSparseCholeskyFactor


@dataclass
class _SparseComponentCaseCache:
    """Per-case sparse conditioning state for one connected component."""

    num_cases: int
    factor: BlockSparseCholeskyFactor
    site_precision: list[Tensor]
    site_info: list[Tensor]
    first_site_index: int | None


def _connected_index_components(structure: ChordalStructure) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(structure.order)))
    components: list[tuple[int, ...]] = []
    adjacency_by_index = tuple(
        tuple(sorted(structure.node_to_index[neighbor] for neighbor in structure.adjacency[node]))
        for node in structure.order
    )
    while remaining:
        root = min(remaining)
        stack = [root]
        visited: list[int] = []
        remaining.remove(root)
        while stack:
            idx = stack.pop()
            visited.append(idx)
            for neighbor in adjacency_by_index[idx]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(visited)))
    return tuple(components)


def _component_structure(structure: ChordalStructure, global_indices: tuple[int, ...]) -> ChordalStructure:
    local_nodes = tuple(structure.order[idx] for idx in global_indices)
    local_node_set = set(local_nodes)
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_indices)}
    adjacency = {
        node: frozenset(
            neighbor
            for neighbor in structure.adjacency[node]
            if neighbor in local_node_set
        )
        for node in local_nodes
    }
    lower_pairs = tuple(
        (global_to_local[row], global_to_local[col])
        for row, col in structure.lower_block_pairs
        if row in global_to_local and col in global_to_local
    )
    lower_children_lists: list[list[int]] = [[] for _ in local_nodes]
    for row, col in lower_pairs:
        lower_children_lists[col].append(row)
    lower_children = tuple(tuple(sorted(children)) for children in lower_children_lists)
    maximal_cliques = tuple(
        clique
        for clique in structure.maximal_cliques
        if set(clique).issubset(local_node_set)
    )
    observed_nodes = tuple(node for node in structure.observed_nodes if node in local_node_set)
    ancestor_nodes = tuple(node for node in structure.ancestor_nodes if node in local_node_set)
    return ChordalStructure(
        latent_nodes=local_nodes,
        observed_nodes=observed_nodes,
        ancestor_nodes=ancestor_nodes,
        order=local_nodes,
        adjacency=adjacency,
        lower_block_pairs=lower_pairs,
        lower_children=lower_children,
        maximal_cliques=maximal_cliques,
    )


def _build_sparse_components(
    posterior,
    structure: ChordalStructure,
    *,
    precision_blocks: dict[tuple[int, int], Tensor],
    mean_blocks: tuple[Tensor, ...],
    marginal_covariance_blocks: tuple[Tensor, ...],
) -> tuple[tuple[_SparseComponentBase, ...], tuple[int, ...]]:
    components: list[_SparseComponentBase] = []
    node_to_component = [-1] * len(structure.order)
    for component_id, global_indices in enumerate(_connected_index_components(structure)):
        local_structure = _component_structure(structure, global_indices)
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_indices)}
        local_block_sizes = tuple(posterior.block_size_by_index(global_idx) for global_idx in global_indices)
        local_precision_blocks: dict[tuple[int, int], Tensor] = {}
        for row, col in local_structure.lower_block_pairs:
            global_row = global_indices[row]
            global_col = global_indices[col]
            local_precision_blocks[(row, col)] = precision_blocks[(global_row, global_col)]
        for local_idx, global_idx in enumerate(global_indices):
            local_precision_blocks[(local_idx, local_idx)] = precision_blocks[(global_idx, global_idx)]
            node_to_component[global_idx] = component_id
        component = _SparseComponentBase(
            global_indices=global_indices,
            global_to_local=global_to_local,
            structure=local_structure,
            block_sizes=local_block_sizes,
            precision_blocks=local_precision_blocks,
            mean_blocks=tuple(mean_blocks[global_idx] for global_idx in global_indices),
            marginal_covariance_blocks=tuple(marginal_covariance_blocks[global_idx] for global_idx in global_indices),
            base_factor=factorize_precision_blocks(
                structure=local_structure,
                block_sizes=local_block_sizes,
                diag_precision_blocks=tuple(
                    local_precision_blocks[(idx, idx)]
                    for idx in range(len(local_block_sizes))
                ),
                offdiag_precision_blocks={
                    key: value
                    for key, value in local_precision_blocks.items()
                    if key[0] != key[1]
                },
                jitter=1e-9,
            ),
        )
        components.append(component)
    return tuple(components), tuple(int(value) for value in node_to_component)


def _build_dense_posterior_moments(model) -> _DensePosteriorMoments:
    posterior = model.posterior
    covariance = posterior.covariance_matrix()
    slices = {node: posterior.block_slice(node) for node in posterior.structure.order}
    return _DensePosteriorMoments(
        mean_blocks=posterior.mean_blocks(),
        covariance=covariance,
        slices=slices,
        trace_covariance=torch.trace(covariance),
        logdet_precision=posterior.logdet_precision(),
    )


def _build_sparse_posterior_base(model) -> _SparsePosteriorBase:
    posterior = model.posterior
    structure = posterior.structure
    precision_blocks = posterior.precision_blocks()
    mean_by_node = posterior.mean_blocks()
    mean_blocks = tuple(mean_by_node[node] for node in structure.order)
    selected = posterior.selected_inverse()
    trace_covariance = posterior.q_mean.new_tensor(0.0)
    for idx in range(len(structure.order)):
        trace_covariance = trace_covariance + torch.trace(selected[(idx, idx)])

    marginal_covariance_blocks = tuple(selected[(idx, idx)] for idx in range(len(structure.order)))
    components, node_to_component = _build_sparse_components(
        posterior,
        structure,
        precision_blocks=precision_blocks,
        mean_blocks=mean_blocks,
        marginal_covariance_blocks=marginal_covariance_blocks,
    )

    return _SparsePosteriorBase(
        components=components,
        node_to_component=node_to_component,
        trace_covariance=trace_covariance,
        logdet_precision=posterior.logdet_precision(),
    )


def _prepare_sparse_case_cache(base: _SparsePosteriorBase, *, num_cases: int) -> _SparseCaseCache:
    return _SparseCaseCache(
        num_cases=int(num_cases),
        components=[None for _ in base.components],
    )


def _get_sparse_component_case_cache(
    base: _SparsePosteriorBase,
    cache: _SparseCaseCache,
    *,
    component_id: int,
) -> _SparseComponentCaseCache:
    component_cache = cache.components[component_id]
    if component_cache is not None:
        return component_cache

    component = base.components[component_id]
    num_cases = int(cache.num_cases)
    base_factor = component.base_factor
    component_cache = _SparseComponentCaseCache(
        num_cases=num_cases,
        factor=BlockSparseCholeskyFactor(
            structure=base_factor.structure,
            block_sizes=base_factor.block_sizes,
            diag_blocks=tuple(
                block.unsqueeze(0).expand(num_cases, *block.shape).clone()
                for block in base_factor.diag_blocks
            ),
            offdiag_blocks={
                key: block.unsqueeze(0).expand(num_cases, *block.shape).clone()
                for key, block in base_factor.offdiag_blocks.items()
            },
            batch_shape=(num_cases,),
            incoming_neighbors=base_factor.incoming_neighbors,
        ),
        site_precision=[
            component.mean_blocks[0].new_zeros((num_cases, size, size))
            for size in component.block_sizes
        ],
        site_info=[
            component.mean_blocks[0].new_zeros((num_cases, size))
            for size in component.block_sizes
        ],
        first_site_index=None,
    )
    cache.components[component_id] = component_cache
    return component_cache


def _append_node_latent_variables(
    model,
    *,
    node: str,
    node_inputs: Tensor,
    indices: Optional[Tensor],
    sample_from_prior: bool = False,
) -> Tensor:
    """Append per-data-point variational latent variables to GP inputs."""
    z = model.sample_latent_layer(
        node,
        num_samples=int(node_inputs.shape[0]),
        indices=indices,
        batch_size=int(node_inputs.shape[1]),
        sample_from_prior=sample_from_prior,
    )

    if z is None:
        return node_inputs

    if z.shape[:2] != node_inputs.shape[:2]:
        raise RuntimeError(
            f"Latent variable samples for node '{node}' have shape {tuple(z.shape)}, "
            f"expected leading shape {tuple(node_inputs.shape[:2])}."
        )

    return torch.cat([node_inputs, z], dim=-1)


def _bilinear(left: Tensor, matrix: Tensor, right: Tensor) -> Tensor:
    """Return ``left^T matrix right`` for batches of row vectors."""
    return torch.einsum("...i,...ij,...j->...", left, matrix, right)


def _solve_local_cholesky(cholesky: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor]:
    """Solve a batched local system from its Cholesky factor.

    The second result is the intermediate lower-triangular solve. The caller
    reuses its cross-covariance column to append the next row and column to the
    factor without refactorising the growing local covariance matrix.
    """
    num_dim = int(cholesky.shape[-1])
    if num_dim == 0:
        return rhs, rhs

    flat_cholesky = cholesky.reshape(-1, num_dim, num_dim)
    if rhs.dim() == cholesky.dim():
        flat_rhs = rhs.reshape(-1, *rhs.shape[-2:])
        squeeze_last = False
    else:
        flat_rhs = rhs.reshape(-1, num_dim, 1)
        squeeze_last = True

    lower_solved = solve_lower_triangular(flat_cholesky, flat_rhs)
    solved = solve_upper_triangular(flat_cholesky.transpose(-1, -2), lower_solved)

    if squeeze_last:
        lower_solved = lower_solved.squeeze(-1)
        solved = solved.squeeze(-1)
    return solved.reshape(rhs.shape), lower_solved.reshape(rhs.shape)


def _augment_local_cholesky(
    local_cholesky: Tensor | None,
    cross_projection: Tensor | None,
    diagonal: Tensor,
    *,
    jitter: float,
) -> Tensor:
    """Append one variable to the Cholesky factor of ``C + jitter I``.

    If ``L L^T = C + jitter I`` and ``v = L^{-1} c``, bordering gives

    ``[[L, 0], [v^T, sqrt(d + jitter - v^T v)]]``.
    """
    regularised_diagonal = diagonal + float(jitter)
    if local_cholesky is None:
        return torch.sqrt(regularised_diagonal).unsqueeze(-1).unsqueeze(-1)
    if cross_projection is None:
        raise ValueError(
            "cross_projection must be provided when augmenting a non-empty Cholesky factor."
        )

    num_previous = int(local_cholesky.shape[-1])
    if int(cross_projection.shape[-1]) != num_previous:
        raise ValueError(
            "cross_projection must match the size of the existing Cholesky factor."
        )

    schur_complement = regularised_diagonal - cross_projection.square().sum(dim=-1)
    new_diagonal = torch.sqrt(schur_complement)
    zero_column = local_cholesky.new_zeros(local_cholesky.shape[:-1] + (1,))
    top = torch.cat([local_cholesky, zero_column], dim=-1)
    bottom = torch.cat(
        [cross_projection, new_diagonal.unsqueeze(-1)],
        dim=-1,
    ).unsqueeze(-2)
    return torch.cat([top, bottom], dim=-2)


def _sample_from_gaussian(mean: Tensor, variance: Tensor) -> Tensor:
    eps = torch.randn_like(mean)
    return mean + torch.sqrt(variance.clamp_min(1e-9)) * eps


def _conditional_from_previous(
    *,
    interp: Tensor,
    base_mean: Tensor,
    base_variance: Tensor,
    moments: _DensePosteriorMoments,
    node: str,
    previous_nodes: list[str],
    interpolation_history: dict[str, Tensor],
    local_base_mean: Tensor | None,
    local_draws: Tensor | None,
    local_cholesky: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if not previous_nodes:
        return base_mean, base_variance, None
    if local_base_mean is None or local_draws is None or local_cholesky is None:
        raise RuntimeError("Previous-node conditioning state is incomplete.")

    cross_terms = [
        _bilinear(
            interp,
            moments.cov_block(node, previous_node),
            interpolation_history[previous_node],
        )
        for previous_node in previous_nodes
    ]
    cross_to_previous = torch.stack(cross_terms, dim=-1)
    delta = local_draws - local_base_mean
    solve_rhs = torch.stack([delta, cross_to_previous], dim=-1)
    solve_result, lower_solve_result = _solve_local_cholesky(
        local_cholesky,
        solve_rhs,
    )
    alpha = solve_result[..., 0]
    beta = solve_result[..., 1]
    conditional_mean = base_mean + torch.sum(cross_to_previous * alpha, dim=-1)
    conditional_variance = base_variance - torch.sum(cross_to_previous * beta, dim=-1)
    cross_projection = lower_solve_result[..., 1]
    return conditional_mean, conditional_variance, cross_projection


def _sample_latents_dag_svi_dense(
    model,
    dataset: DAGDataset,
    *,
    num_samples: int,
    indices: Optional[Tensor] = None,
    sample_latent_prior: bool = False,
    local_conditioning_jitter: float = 1e-6,
) -> dict[str, Tensor]:
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    roots = data.roots if indices is None else {name: values[indices] for name, values in data.roots.items()}
    moments = _build_dense_posterior_moments(model)

    latent_samples: dict[str, Tensor] = {}
    interpolation_history: dict[str, Tensor] = {}
    local_base_mean: Tensor | None = None
    local_draws: Tensor | None = None
    local_cholesky: Tensor | None = None
    previous_nodes: list[str] = []

    for node in model.dag.topo:
        if node in model.root_dims:
            continue

        node_inputs = sampled_node_inputs(
            model.dag,
            node=node,
            root_dims=model.root_dims,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_samples,
        )
        node_inputs = _append_node_latent_variables(
            model,
            node=node,
            node_inputs=node_inputs,
            indices=indices,
            sample_from_prior=sample_latent_prior,
        )
        num_mc, batch_size, input_dim = node_inputs.shape
        flat_inputs = node_inputs.reshape(num_mc * batch_size, input_dim)

        interp_flat, residual_flat = model.gp_modules[node].conditional_statistics(flat_inputs)
        interp = interp_flat.reshape(num_mc, batch_size, -1)
        residual = residual_flat.reshape(num_mc, batch_size)

        base_mean = (
            torch.einsum("sbm,m->sb", interp, moments.mean_blocks[node])
            + model.gp_modules[node].mean_offset
        )
        base_variance = residual + _bilinear(interp, moments.cov_block(node, node), interp)
        conditional_mean, conditional_variance, cross_projection = _conditional_from_previous(
            interp=interp,
            base_mean=base_mean,
            base_variance=base_variance,
            moments=moments,
            node=node,
            previous_nodes=previous_nodes,
            interpolation_history=interpolation_history,
            local_base_mean=local_base_mean,
            local_draws=local_draws,
            local_cholesky=local_cholesky,
        )

        samples = _sample_from_gaussian(conditional_mean, conditional_variance)
        latent_samples[node] = samples.unsqueeze(-1)
        interpolation_history[node] = interp

        local_base_mean = (
            base_mean.unsqueeze(-1)
            if local_base_mean is None
            else torch.cat([local_base_mean, base_mean.unsqueeze(-1)], dim=-1)
        )
        local_draws = (
            samples.unsqueeze(-1)
            if local_draws is None
            else torch.cat([local_draws, samples.unsqueeze(-1)], dim=-1)
        )
        local_cholesky = _augment_local_cholesky(
            local_cholesky,
            cross_projection,
            base_variance,
            jitter=local_conditioning_jitter,
        )
        previous_nodes.append(node)

    return latent_samples


def _elbo_dag_svi_union_dense(
    model,
    dataset: DAGDataset,
    *,
    indices: Optional[Tensor],
    num_mc_samples: int,
    local_conditioning_jitter: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    roots = data.roots if indices is None else {name: values[indices] for name, values in data.roots.items()}
    moments = _build_dense_posterior_moments(model)

    analytic_gaussian_nodes: set[str] = set()
    if getattr(model, "analytic_gaussian_leaves", True):
        analytic_gaussian_nodes = {
            node
            for node in data.y
            if node in model.likelihoods and isinstance(model.likelihood(node), GaussianLikelihood)
        }

    latent_samples: dict[str, Tensor] = {}
    interpolation_history: dict[str, Tensor] = {}
    local_base_mean: Tensor | None = None
    local_draws: Tensor | None = None
    local_cholesky: Tensor | None = None
    previous_nodes: list[str] = []
    total_ll = reference.new_tensor(0.0)

    for node in model.dag.topo:
        if node in model.root_dims:
            continue

        node_inputs = sampled_node_inputs(
            model.dag,
            node=node,
            root_dims=model.root_dims,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_mc_samples,
        )
        node_inputs = _append_node_latent_variables(
            model,
            node=node,
            node_inputs=node_inputs,
            indices=indices,
        )
        num_samples, batch_size, input_dim = node_inputs.shape
        flat_inputs = node_inputs.reshape(num_samples * batch_size, input_dim)

        interp_flat, residual_flat = model.gp_modules[node].conditional_statistics(flat_inputs)
        interp = interp_flat.reshape(num_samples, batch_size, -1)
        residual = residual_flat.reshape(num_samples, batch_size)

        base_mean = (
            torch.einsum("sbm,m->sb", interp, moments.mean_blocks[node])
            + model.gp_modules[node].mean_offset
        )
        base_variance = residual + _bilinear(interp, moments.cov_block(node, node), interp)
        conditional_mean, conditional_variance, cross_projection = _conditional_from_previous(
            interp=interp,
            base_mean=base_mean,
            base_variance=base_variance,
            moments=moments,
            node=node,
            previous_nodes=previous_nodes,
            interpolation_history=interpolation_history,
            local_base_mean=local_base_mean,
            local_draws=local_draws,
            local_cholesky=local_cholesky,
        )
        conditional_variance = conditional_variance.clamp_min(1e-9)
        sampled_values = _sample_from_gaussian(conditional_mean, conditional_variance)
        latent_samples[node] = sampled_values.unsqueeze(-1)
        interpolation_history[node] = interp

        if node in data.y:
            y, mask = data.get_observations(node)
            if indices is not None:
                y = y[indices]
                mask = mask[indices]
            y = torch.where(mask, y, torch.zeros_like(y))
            if node in analytic_gaussian_nodes:
                y_mc = y.unsqueeze(0).expand_as(conditional_mean)
                ll = model.likelihood(node).expected_log_prob(y_mc, conditional_mean, conditional_variance)
            else:
                ll = model.likelihood(node).log_prob(y.unsqueeze(0), sampled_values)
            total_ll = total_ll + (ll * mask.unsqueeze(0)).mean(dim=0).sum()

        local_base_mean = (
            base_mean.unsqueeze(-1)
            if local_base_mean is None
            else torch.cat([local_base_mean, base_mean.unsqueeze(-1)], dim=-1)
        )
        local_draws = (
            sampled_values.unsqueeze(-1)
            if local_draws is None
            else torch.cat([local_draws, sampled_values.unsqueeze(-1)], dim=-1)
        )
        local_cholesky = _augment_local_cholesky(
            local_cholesky,
            cross_projection,
            base_variance,
            jitter=local_conditioning_jitter,
        )
        previous_nodes.append(node)

    return total_ll, moments.trace_covariance, moments.logdet_precision


def _node_likelihood_term_dag_svi_dense(
    model,
    data: DAGDataset,
    *,
    node: str,
    num_mc_samples: int,
    moments: _DensePosteriorMoments,
    local_conditioning_jitter: float = 1e-6,
) -> Tensor:
    """Compute a node's expected log likelihood at its observed data points."""
    reference = next(model.parameters())
    y_node, mask_node = data.get_observations(node)
    obs_idx = torch.nonzero(mask_node, as_tuple=False).squeeze(-1)
    if int(obs_idx.numel()) == 0:
        return reference.new_tensor(0.0)

    roots = {name: values[obs_idx] for name, values in data.roots.items()}
    active_nodes = set(model.dag.ancestors_of((node,), include_self=True))

    latent_samples: dict[str, Tensor] = {}
    interpolation_history: dict[str, Tensor] = {}
    local_base_mean: Tensor | None = None
    local_draws: Tensor | None = None
    local_cholesky: Tensor | None = None
    previous_nodes: list[str] = []

    for latent in model.dag.topo:
        if latent in model.root_dims or latent not in active_nodes:
            continue

        node_inputs = sampled_node_inputs(
            model.dag,
            node=latent,
            root_dims=model.root_dims,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_mc_samples,
        )
        node_inputs = _append_node_latent_variables(
            model,
            node=latent,
            node_inputs=node_inputs,
            indices=obs_idx,
        )
        num_samples, batch_size, input_dim = node_inputs.shape
        flat_inputs = node_inputs.reshape(num_samples * batch_size, input_dim)

        interp_flat, residual_flat = model.gp_modules[latent].conditional_statistics(flat_inputs)
        interp = interp_flat.reshape(num_samples, batch_size, -1)
        residual = residual_flat.reshape(num_samples, batch_size)

        base_mean = (
            torch.einsum("sbm,m->sb", interp, moments.mean_blocks[latent])
            + model.gp_modules[latent].mean_offset
        )
        base_variance = residual + _bilinear(interp, moments.cov_block(latent, latent), interp)
        conditional_mean, conditional_variance, cross_projection = _conditional_from_previous(
            interp=interp,
            base_mean=base_mean,
            base_variance=base_variance,
            moments=moments,
            node=latent,
            previous_nodes=previous_nodes,
            interpolation_history=interpolation_history,
            local_base_mean=local_base_mean,
            local_draws=local_draws,
            local_cholesky=local_cholesky,
        )
        conditional_variance = conditional_variance.clamp_min(1e-9)

        if latent == node:
            y_obs = y_node[obs_idx]
            if latent in data.y and isinstance(model.likelihood(latent), GaussianLikelihood):
                y_mc = y_obs.unsqueeze(0).expand_as(conditional_mean)
                ll = model.likelihood(latent).expected_log_prob(y_mc, conditional_mean, conditional_variance)
            else:
                sampled_values = _sample_from_gaussian(conditional_mean, conditional_variance)
                ll = model.likelihood(latent).log_prob(y_obs.unsqueeze(0), sampled_values)
            return ll.mean(dim=0).sum()

        sampled_values = _sample_from_gaussian(conditional_mean, conditional_variance)
        latent_samples[latent] = sampled_values.unsqueeze(-1)
        interpolation_history[latent] = interp
        local_base_mean = (
            base_mean.unsqueeze(-1)
            if local_base_mean is None
            else torch.cat([local_base_mean, base_mean.unsqueeze(-1)], dim=-1)
        )
        local_draws = (
            sampled_values.unsqueeze(-1)
            if local_draws is None
            else torch.cat([local_draws, sampled_values.unsqueeze(-1)], dim=-1)
        )
        local_cholesky = _augment_local_cholesky(
            local_cholesky,
            cross_projection,
            base_variance,
            jitter=local_conditioning_jitter,
        )
        previous_nodes.append(latent)

    raise RuntimeError(f"Failed to evaluate likelihood term for node '{node}'.")


def _elbo_dag_svi_nodewise_dense(
    model,
    dataset: DAGDataset,
    *,
    num_mc_samples: int,
    local_conditioning_jitter: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute nodewise expected log likelihoods and KL statistics using dense covariance."""
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    moments = _build_dense_posterior_moments(model)

    total_ll = reference.new_tensor(0.0)
    for node in model.dag.topo:
        if node in model.root_dims or node not in data.y:
            continue
        total_ll = total_ll + _node_likelihood_term_dag_svi_dense(
            model,
            data,
            node=node,
            num_mc_samples=num_mc_samples,
            moments=moments,
            local_conditioning_jitter=local_conditioning_jitter,
        )
    return total_ll, moments.trace_covariance, moments.logdet_precision


def _conditioned_sparse_moments(
    base: _SparseComponentBase,
    cache: _SparseComponentCaseCache,
    *,
    node_index: int,
    interp_flat: Tensor,
) -> tuple[Tensor, Tensor]:
    num_cases = cache.num_cases
    if int(interp_flat.shape[0]) != num_cases:
        raise ValueError("interp_flat batch dimension must match the sparse case cache.")

    if cache.first_site_index is None:
        mean_block = base.mean_blocks[node_index].unsqueeze(0).expand(num_cases, -1)
        covariance_quadratic = _bilinear(
            interp_flat,
            base.marginal_covariance_blocks[node_index],
            interp_flat,
        )
        return mean_block, covariance_quadratic

    factor = cache.factor
    start = min(int(cache.first_site_index), int(node_index))
    rhs_blocks: list[Tensor] = []
    zero_block = None
    for idx in range(start, len(base.block_sizes)):
        block_size = int(base.block_sizes[idx])
        if zero_block is None or int(zero_block.shape[1]) != block_size:
            zero_block = interp_flat.new_zeros((num_cases, block_size, 2))
        rhs_block = zero_block.clone()
        if idx >= int(cache.first_site_index):
            rhs_block[:, :, 0] = cache.site_info[idx] - torch.matmul(cache.site_precision[idx], base.mean_blocks[idx])
        if idx == node_index:
            rhs_block[:, :, 1] = interp_flat
        rhs_blocks.append(rhs_block)

    solved_blocks = factor.solve_suffix(rhs_blocks, start_index=start)
    target_solution = solved_blocks[node_index - start]
    mean_correction = target_solution[:, :, 0]
    mean_block = base.mean_blocks[node_index].unsqueeze(0).expand(num_cases, -1) + mean_correction
    query_solution = target_solution[:, :, 1]
    covariance_quadratic = torch.sum(interp_flat * query_solution, dim=-1)
    return mean_block, covariance_quadratic


def _site_update_from_local_draw(interp: Tensor, residual: Tensor, draw: Tensor) -> tuple[Tensor, Tensor]:
    inv_residual = residual.clamp_min(1e-9).reciprocal()
    precision_update = inv_residual.unsqueeze(-1).unsqueeze(-1) * (
        interp.unsqueeze(-1) @ interp.unsqueeze(-2)
    )
    info_update = interp * (draw * inv_residual).unsqueeze(-1)
    return precision_update, info_update


def _sample_latents_dag_svi_sparse(
    model,
    dataset: DAGDataset,
    *,
    num_samples: int,
    indices: Optional[Tensor] = None,
    sample_latent_prior: bool = False,
) -> dict[str, Tensor]:
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    roots = data.roots if indices is None else {name: values[indices] for name, values in data.roots.items()}
    base = _build_sparse_posterior_base(model)

    latent_samples: dict[str, Tensor] = {}
    cache: _SparseCaseCache | None = None
    node_to_index = model.posterior.structure.node_to_index

    for node in model.dag.topo:
        if node in model.root_dims:
            continue

        node_inputs = sampled_node_inputs(
            model.dag,
            node=node,
            root_dims=model.root_dims,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_samples,
        )
        node_inputs = _append_node_latent_variables(
            model,
            node=node,
            node_inputs=node_inputs,
            indices=indices,
            sample_from_prior=sample_latent_prior,
        )
        num_mc, batch_size, input_dim = node_inputs.shape
        num_cases = int(num_mc * batch_size)
        if cache is None:
            cache = _prepare_sparse_case_cache(base, num_cases=num_cases)

        flat_inputs = node_inputs.reshape(num_cases, input_dim)
        interp_flat, residual_flat = model.gp_modules[node].conditional_statistics(flat_inputs)

        global_idx = node_to_index[node]
        component_id = base.node_to_component[global_idx]
        component = base.components[component_id]
        component_cache = _get_sparse_component_case_cache(base, cache, component_id=component_id)
        local_idx = component.global_to_local[global_idx]
        mean_block, covariance_quadratic = _conditioned_sparse_moments(
            component,
            component_cache,
            node_index=local_idx,
            interp_flat=interp_flat,
        )
        mean_offset = model.gp_modules[node].mean_offset
        conditional_mean = torch.sum(interp_flat * mean_block, dim=-1) + mean_offset
        conditional_variance = residual_flat + covariance_quadratic
        conditional_variance = conditional_variance.clamp_min(1e-9)
        sampled_flat = _sample_from_gaussian(conditional_mean, conditional_variance)

        latent_samples[node] = sampled_flat.reshape(num_mc, batch_size).unsqueeze(-1)
        precision_update, info_update = _site_update_from_local_draw(
            interp_flat,
            residual_flat,
            sampled_flat - mean_offset,
        )
        component_cache.site_precision[local_idx] = component_cache.site_precision[local_idx] + precision_update
        component_cache.site_info[local_idx] = component_cache.site_info[local_idx] + info_update
        update_vector = interp_flat * residual_flat.clamp_min(1e-9).rsqrt().unsqueeze(-1)
        component_cache.factor = diagonal_rank1_update_(
            component_cache.factor,
            block_index=local_idx,
            update_vector=update_vector,
        )
        component_first = component_cache.first_site_index
        component_cache.first_site_index = local_idx if component_first is None else min(component_first, local_idx)

    return latent_samples


def _elbo_dag_svi_union_sparse(
    model,
    dataset: DAGDataset,
    *,
    indices: Optional[Tensor],
    num_mc_samples: int,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    roots = data.roots if indices is None else {name: values[indices] for name, values in data.roots.items()}
    base = _build_sparse_posterior_base(model)

    analytic_gaussian_nodes: set[str] = set()
    if getattr(model, "analytic_gaussian_leaves", True):
        analytic_gaussian_nodes = {
            node
            for node in data.y
            if node in model.likelihoods and isinstance(model.likelihood(node), GaussianLikelihood)
        }

    latent_samples: dict[str, Tensor] = {}
    total_ll = reference.new_tensor(0.0)
    cache: _SparseCaseCache | None = None
    node_to_index = model.posterior.structure.node_to_index

    for node in model.dag.topo:
        if node in model.root_dims:
            continue

        node_inputs = sampled_node_inputs(
            model.dag,
            node=node,
            root_dims=model.root_dims,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_mc_samples,
        )
        node_inputs = _append_node_latent_variables(
            model,
            node=node,
            node_inputs=node_inputs,
            indices=indices,
        )
        num_samples, batch_size, input_dim = node_inputs.shape
        num_cases = int(num_samples * batch_size)
        if cache is None:
            cache = _prepare_sparse_case_cache(base, num_cases=num_cases)

        flat_inputs = node_inputs.reshape(num_cases, input_dim)
        interp_flat, residual_flat = model.gp_modules[node].conditional_statistics(flat_inputs)
        global_idx = node_to_index[node]
        component_id = base.node_to_component[global_idx]
        component = base.components[component_id]
        component_cache = _get_sparse_component_case_cache(base, cache, component_id=component_id)
        local_idx = component.global_to_local[global_idx]
        mean_block, covariance_quadratic = _conditioned_sparse_moments(
            component,
            component_cache,
            node_index=local_idx,
            interp_flat=interp_flat,
        )
        mean_offset = model.gp_modules[node].mean_offset
        conditional_mean_flat = torch.sum(interp_flat * mean_block, dim=-1) + mean_offset
        conditional_variance_flat = residual_flat + covariance_quadratic
        conditional_variance_flat = conditional_variance_flat.clamp_min(1e-9)

        sampled_flat = _sample_from_gaussian(conditional_mean_flat, conditional_variance_flat)
        sampled_values = sampled_flat.reshape(num_samples, batch_size)
        conditional_mean = conditional_mean_flat.reshape(num_samples, batch_size)
        conditional_variance = conditional_variance_flat.reshape(num_samples, batch_size)
        latent_samples[node] = sampled_values.unsqueeze(-1)

        if node in data.y:
            y, mask = data.get_observations(node)
            if indices is not None:
                y = y[indices]
                mask = mask[indices]
            y = torch.where(mask, y, torch.zeros_like(y))
            if node in analytic_gaussian_nodes:
                y_mc = y.unsqueeze(0).expand_as(conditional_mean)
                ll = model.likelihood(node).expected_log_prob(y_mc, conditional_mean, conditional_variance)
            else:
                ll = model.likelihood(node).log_prob(y.unsqueeze(0), sampled_values)
            total_ll = total_ll + (ll * mask.unsqueeze(0)).mean(dim=0).sum()

        precision_update, info_update = _site_update_from_local_draw(
            interp_flat,
            residual_flat,
            sampled_flat - mean_offset,
        )
        component_cache.site_precision[local_idx] = component_cache.site_precision[local_idx] + precision_update
        component_cache.site_info[local_idx] = component_cache.site_info[local_idx] + info_update
        update_vector = interp_flat * residual_flat.clamp_min(1e-9).rsqrt().unsqueeze(-1)
        component_cache.factor = diagonal_rank1_update_(
            component_cache.factor,
            block_index=local_idx,
            update_vector=update_vector,
        )
        component_first = component_cache.first_site_index
        component_cache.first_site_index = local_idx if component_first is None else min(component_first, local_idx)

    return total_ll, base.trace_covariance, base.logdet_precision


def _node_likelihood_term_dag_svi_sparse(
    model,
    data: DAGDataset,
    *,
    node: str,
    num_mc_samples: int,
    base: _SparsePosteriorBase,
) -> Tensor:
    """Compute one likelihood term on its observed support only using sparse updates."""
    reference = next(model.parameters())
    y_node, mask_node = data.get_observations(node)
    obs_idx = torch.nonzero(mask_node, as_tuple=False).squeeze(-1)
    if int(obs_idx.numel()) == 0:
        return reference.new_tensor(0.0)

    roots = {name: values[obs_idx] for name, values in data.roots.items()}
    active_nodes = set(model.dag.ancestors_of((node,), include_self=True))
    node_to_index = model.posterior.structure.node_to_index

    latent_samples: dict[str, Tensor] = {}
    cache: _SparseCaseCache | None = None

    for latent in model.dag.topo:
        if latent in model.root_dims or latent not in active_nodes:
            continue

        node_inputs = sampled_node_inputs(
            model.dag,
            node=latent,
            root_dims=model.root_dims,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_mc_samples,
        )
        node_inputs = _append_node_latent_variables(
            model,
            node=latent,
            node_inputs=node_inputs,
            indices=obs_idx,
        )
        num_samples, batch_size, input_dim = node_inputs.shape
        num_cases = int(num_samples * batch_size)
        if cache is None:
            cache = _prepare_sparse_case_cache(base, num_cases=num_cases)

        flat_inputs = node_inputs.reshape(num_cases, input_dim)
        interp_flat, residual_flat = model.gp_modules[latent].conditional_statistics(flat_inputs)
        global_idx = node_to_index[latent]
        component_id = base.node_to_component[global_idx]
        component = base.components[component_id]
        component_cache = _get_sparse_component_case_cache(base, cache, component_id=component_id)
        local_idx = component.global_to_local[global_idx]
        mean_block, covariance_quadratic = _conditioned_sparse_moments(
            component,
            component_cache,
            node_index=local_idx,
            interp_flat=interp_flat,
        )
        mean_offset = model.gp_modules[latent].mean_offset
        conditional_mean_flat = torch.sum(interp_flat * mean_block, dim=-1) + mean_offset
        conditional_variance_flat = residual_flat + covariance_quadratic
        conditional_variance_flat = conditional_variance_flat.clamp_min(1e-9)
        conditional_mean = conditional_mean_flat.reshape(num_samples, batch_size)
        conditional_variance = conditional_variance_flat.reshape(num_samples, batch_size)

        if latent == node:
            y_obs = y_node[obs_idx]
            if latent in data.y and isinstance(model.likelihood(latent), GaussianLikelihood):
                y_mc = y_obs.unsqueeze(0).expand_as(conditional_mean)
                ll = model.likelihood(latent).expected_log_prob(y_mc, conditional_mean, conditional_variance)
            else:
                sampled_values = _sample_from_gaussian(conditional_mean, conditional_variance)
                ll = model.likelihood(latent).log_prob(y_obs.unsqueeze(0), sampled_values)
            return ll.mean(dim=0).sum()

        sampled_flat = _sample_from_gaussian(conditional_mean_flat, conditional_variance_flat)
        latent_samples[latent] = sampled_flat.reshape(num_samples, batch_size).unsqueeze(-1)
        precision_update, info_update = _site_update_from_local_draw(
            interp_flat,
            residual_flat,
            sampled_flat - mean_offset,
        )
        component_cache.site_precision[local_idx] = component_cache.site_precision[local_idx] + precision_update
        component_cache.site_info[local_idx] = component_cache.site_info[local_idx] + info_update
        update_vector = interp_flat * residual_flat.clamp_min(1e-9).rsqrt().unsqueeze(-1)
        component_cache.factor = diagonal_rank1_update_(
            component_cache.factor,
            block_index=local_idx,
            update_vector=update_vector,
        )
        component_first = component_cache.first_site_index
        component_cache.first_site_index = local_idx if component_first is None else min(component_first, local_idx)

    raise RuntimeError(f"Failed to evaluate likelihood term for node '{node}'.")


def _elbo_dag_svi_nodewise_sparse(
    model,
    dataset: DAGDataset,
    *,
    num_mc_samples: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute nodewise expected log likelihoods and KL statistics using sparse updates."""
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    base = _build_sparse_posterior_base(model)

    total_ll = reference.new_tensor(0.0)
    for node in model.dag.topo:
        if node in model.root_dims or node not in data.y:
            continue
        total_ll = total_ll + _node_likelihood_term_dag_svi_sparse(
            model,
            data,
            node=node,
            num_mc_samples=num_mc_samples,
            base=base,
        )
    return total_ll, base.trace_covariance, base.logdet_precision


def sample_latents_dag_svi(
    model,
    dataset: DAGDataset,
    *,
    num_samples: int,
    indices: Optional[Tensor] = None,
    sample_latent_prior: bool = False,
    local_conditioning_jitter: float = 1e-6,
) -> dict[str, Tensor]:
    """Sample all latent nodes under the structured posterior."""
    backend = getattr(model, "marginal_backend", "dense")
    if backend == "sparse":
        return _sample_latents_dag_svi_sparse(
            model,
            dataset,
            num_samples=num_samples,
            indices=indices,
            sample_latent_prior=sample_latent_prior,
        )
    return _sample_latents_dag_svi_dense(
        model,
        dataset,
        num_samples=num_samples,
        indices=indices,
        sample_latent_prior=sample_latent_prior,
        local_conditioning_jitter=local_conditioning_jitter,
    )


def elbo_dag_svi(
    model,
    dataset: DAGDataset,
    *,
    indices: Optional[Tensor],
    num_mc_samples: int,
    local_conditioning_jitter: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the DAG-SVI ELBO data term and KL statistics."""
    backend = getattr(model, "marginal_backend", "dense")
    if backend == "sparse":
        if indices is None:
            return _elbo_dag_svi_nodewise_sparse(
                model,
                dataset,
                num_mc_samples=num_mc_samples,
            )
        return _elbo_dag_svi_union_sparse(
            model,
            dataset,
            indices=indices,
            num_mc_samples=num_mc_samples,
        )
    if indices is None:
        return _elbo_dag_svi_nodewise_dense(
            model,
            dataset,
            num_mc_samples=num_mc_samples,
            local_conditioning_jitter=local_conditioning_jitter,
        )
    return _elbo_dag_svi_union_dense(
        model,
        dataset,
        indices=indices,
        num_mc_samples=num_mc_samples,
        local_conditioning_jitter=local_conditioning_jitter,
    )
