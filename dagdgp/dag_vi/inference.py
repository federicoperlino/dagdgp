"""Mean-field DAG ELBO and latent-sampling helpers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..dag_helpers import prepare_dataset, sampled_node_inputs
from ..data import DAGDataset
from ..likelihoods import GaussianLikelihood


def build_node_input(
    model,
    *,
    node: str,
    root_values: dict[str, Tensor],
    latent_samples: dict[str, Tensor],
    num_samples: int,
    indices: Optional[Tensor] = None,
    sample_latent_prior: bool = False,
) -> Tensor:
    """Build GP inputs for one DAG node, optionally appending latent-variable-layer samples."""
    node_inputs = sampled_node_inputs(
        model.dag,
        node=node,
        root_dims=model.root_dims,
        root_values=root_values,
        latent_samples=latent_samples,
        num_samples=num_samples,
    )

    if not hasattr(model, "sample_latent_layer"):
        return node_inputs

    z = model.sample_latent_layer(
        node,
        num_samples=num_samples,
        indices=indices,
        batch_size=node_inputs.shape[1],
        sample_from_prior=sample_latent_prior,
    )

    if z is None:
        return node_inputs

    if z.ndim != 3:
        raise ValueError(
            f"Latent-variable layer for node '{node}' must return shape "
            f"[num_samples, batch_size, latent_dim], got {tuple(z.shape)}."
        )

    if z.shape[0] != node_inputs.shape[0] or z.shape[1] != node_inputs.shape[1]:
        raise ValueError(
            f"Latent-variable layer shape mismatch for node '{node}': "
            f"node_inputs={tuple(node_inputs.shape)}, z={tuple(z.shape)}."
        )

    return torch.cat([node_inputs, z], dim=-1)


def sample_latents_mean_field(
    model,
    dataset: DAGDataset,
    *,
    num_samples: int,
    indices: Optional[Tensor] = None,
    sample_latent_prior: bool = False,
) -> dict[str, Tensor]:
    """Sample all latent nodes under the mean-field posterior.

    Set sample_latent_prior=True for prediction on new/test data, where no
    learned per-row q(z_n) exists.
    """

    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    roots = data.roots if indices is None else {name: values[indices] for name, values in data.roots.items()}

    samples: dict[str, Tensor] = {}
    for node in model.dag.topo:
        if node in model.root_dims:
            continue

        node_inputs = build_node_input(
            model,
            node=node,
            root_values=roots,
            latent_samples=samples,
            num_samples=num_samples,
            indices=indices,
            sample_latent_prior=sample_latent_prior,
        )

        node_samples: list[Tensor] = []
        for sample_idx in range(num_samples):
            sample = model.gp_modules[node].sample_f_joint(node_inputs[sample_idx], num_samples=1)
            node_samples.append(sample.squeeze(0))

        samples[node] = torch.stack(node_samples, dim=0).unsqueeze(-1)

    return samples


def _elbo_mean_field_union(
    model,
    dataset: DAGDataset,
    *,
    indices: Optional[Tensor],
    num_mc_samples: int,
) -> Tensor:
    """Compute the expected log likelihood over shared input rows using observation masks."""
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)
    roots = data.roots if indices is None else {name: values[indices] for name, values in data.roots.items()}

    children = model.dag.children_map()
    analytic_gaussian_nodes = {
        node
        for node in model.latent_nodes
        if node in data.y and node in model.likelihoods and isinstance(model.likelihoods[node], GaussianLikelihood)
    }

    latent_samples: dict[str, Tensor] = {}
    total_ll = reference.new_tensor(0.0)

    for node in model.dag.topo:
        if node in model.root_dims:
            continue

        node_inputs = build_node_input(
            model,
            node=node,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_mc_samples,
            indices=indices,
            sample_latent_prior=False,
        )

        num_samples, batch_size, input_dim = node_inputs.shape
        flat_inputs = node_inputs.reshape(num_samples * batch_size, input_dim)

        analytic_mean = None
        analytic_var = None

        if node in analytic_gaussian_nodes:
            analytic_mean, analytic_var = model.gp_modules[node].predict_f(flat_inputs)
            analytic_mean = analytic_mean.reshape(num_samples, batch_size)
            analytic_var = analytic_var.reshape(num_samples, batch_size)

            y, mask = data.get_observations(node)
            if indices is not None:
                y = y[indices]
                mask = mask[indices]

            y = torch.where(mask, y, torch.zeros_like(y))
            y_mc = y.unsqueeze(0).expand_as(analytic_mean)
            ll = model.likelihood(node).expected_log_prob(y_mc, analytic_mean, analytic_var)
            total_ll = total_ll + (ll * mask.unsqueeze(0)).mean(dim=0).sum()

        needs_sample = len(children[node]) > 0 or node not in analytic_gaussian_nodes

        if needs_sample:
            if analytic_mean is not None and analytic_var is not None:
                sample = analytic_mean + torch.sqrt(analytic_var.clamp_min(1e-9)) * torch.randn_like(analytic_mean)
            else:
                sample = model.gp_modules[node].sample_f(flat_inputs, num_samples=1).reshape(num_samples, batch_size)

            latent_samples[node] = sample.unsqueeze(-1)

        if node not in data.y or node in analytic_gaussian_nodes:
            continue

        y, mask = data.get_observations(node)
        if indices is not None:
            y = y[indices]
            mask = mask[indices]

        y = torch.where(mask, y, torch.zeros_like(y))
        f = latent_samples[node].squeeze(-1)
        ll = model.likelihood(node).log_prob(y.unsqueeze(0), f)
        total_ll = total_ll + (ll * mask.unsqueeze(0)).mean(dim=0).sum()

    return total_ll


def _node_likelihood_term_mean_field(
    model,
    data: DAGDataset,
    *,
    node: str,
    num_mc_samples: int,
) -> Tensor:
    """Compute one node likelihood term on its observed support only."""
    reference = next(model.parameters())

    y_node, mask_node = data.get_observations(node)
    obs_idx = torch.nonzero(mask_node, as_tuple=False).squeeze(-1)

    if int(obs_idx.numel()) == 0:
        return reference.new_tensor(0.0)

    roots = {name: values[obs_idx] for name, values in data.roots.items()}
    active_nodes = set(model.dag.ancestors_of((node,), include_self=True))
    children = model.dag.children_map()

    analytic_gaussian_nodes = {
        latent
        for latent in active_nodes
        if latent in model.likelihoods and isinstance(model.likelihoods[latent], GaussianLikelihood)
    }

    latent_samples: dict[str, Tensor] = {}

    for latent in model.dag.topo:
        if latent in model.root_dims or latent not in active_nodes:
            continue

        node_inputs = build_node_input(
            model,
            node=latent,
            root_values=roots,
            latent_samples=latent_samples,
            num_samples=num_mc_samples,
            indices=obs_idx,
            sample_latent_prior=False,
        )

        num_samples, batch_size, input_dim = node_inputs.shape
        flat_inputs = node_inputs.reshape(num_samples * batch_size, input_dim)

        analytic_mean = None
        analytic_var = None

        if latent in analytic_gaussian_nodes:
            analytic_mean, analytic_var = model.gp_modules[latent].predict_f(flat_inputs)
            analytic_mean = analytic_mean.reshape(num_samples, batch_size)
            analytic_var = analytic_var.reshape(num_samples, batch_size)

            if latent == node:
                y_obs = y_node[obs_idx]
                y_mc = y_obs.unsqueeze(0).expand_as(analytic_mean)
                ll = model.likelihood(latent).expected_log_prob(y_mc, analytic_mean, analytic_var)

                return ll.mean(dim=0).sum()
            
        

        needs_sample = latent != node and any(child in active_nodes for child in children[latent])

        if needs_sample:
            if analytic_mean is not None and analytic_var is not None:
                sample = analytic_mean + torch.sqrt(analytic_var.clamp_min(1e-9)) * torch.randn_like(analytic_mean)
            else:
                sample = model.gp_modules[latent].sample_f(flat_inputs, num_samples=1).reshape(num_samples, batch_size)

            latent_samples[latent] = sample.unsqueeze(-1)

        if latent == node:
            sampled_values = model.gp_modules[latent].sample_f(flat_inputs, num_samples=1).reshape(
                num_samples,
                batch_size,
            )
            y_obs = y_node[obs_idx]
            ll = model.likelihood(latent).log_prob(y_obs.unsqueeze(0), sampled_values)
            return ll.mean(dim=0).sum()

    raise RuntimeError(f"Failed to evaluate likelihood term for node '{node}'.")


def _elbo_mean_field_nodewise(
    model,
    dataset: DAGDataset,
    *,
    num_mc_samples: int,
) -> Tensor:
    """Compute the ELBO data term node-by-node on observed supports."""
    reference = next(model.parameters())
    data = prepare_dataset(dataset, root_dims=model.root_dims, reference=reference)

    total_ll = reference.new_tensor(0.0)

    for node in model.dag.topo:
        if node in model.root_dims or node not in data.y:
            continue

        total_ll = total_ll + _node_likelihood_term_mean_field(
            model,
            data,
            node=node,
            num_mc_samples=num_mc_samples,
        )

    return total_ll


def elbo_mean_field(
    model,
    dataset: DAGDataset,
    *,
    indices: Optional[Tensor],
    num_mc_samples: int,
) -> Tensor:
    """Compute the stochastic ELBO data term for the mean-field family."""
    if indices is None:
        return _elbo_mean_field_nodewise(
            model,
            dataset,
            num_mc_samples=num_mc_samples,
        )

    return _elbo_mean_field_union(
        model,
        dataset,
        indices=indices,
        num_mc_samples=num_mc_samples,
    )
