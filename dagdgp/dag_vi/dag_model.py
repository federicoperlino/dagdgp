"""Mean-field DSVI DAG (DAG-VI) model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from ..dag import DAG
from ..data import DAGDataset
from ..kernels import RBFKernel
from ..likelihoods import GaussianLikelihood
from .inference import elbo_mean_field, sample_latents_mean_field
from .svgp import MeanFieldSVGP


@dataclass(frozen=True)
class DAGVIModelConfig:
    # inducing points
    num_inducing_points: int = 64

    # likelihood
    likelihood_variance: float = 0.01
    likelihood_eps: float = 1e-6

    # kernels
    kernel_class: type[torch.nn.Module] = RBFKernel
    kernel_ard: bool = True
    kernel_init_lengthscale: float = 0.7
    kernel_init_variance: float = 1.0
    kernel_eps: float = 1e-6
    use_fusion_kernel: bool = True

    # numerical stability
    svgp_jitter: float = 1e-6
    structured_local_jitter: float = 1e-6

    # latent variable layer
    latent_variable_dims: Mapping[str, int] = field(default_factory=dict)
    latent_prior_variance: float = 1.0
    latent_init_std: float = 0.01
    parent_mean_linked_nodes: tuple[str, ...] = ()


class VariationalLatentVariableLayer(nn.Module):
    """
    Per-data-point variational latent variables.

    q(z_n) = N(mu_n, diag(sigma_n^2))
    p(z_n) = N(0, prior_variance I)

    Shapes:
        q_mu:      [N, D]
        q_log_var: [N, D]
        sample:    [S, B, D]
    """

    def __init__(
        self,
        *,
        num_data: int,
        latent_dim: int,
        prior_variance: float = 1.0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_data <= 0:
            raise ValueError("num_data must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive.")

        self.num_data = int(num_data)
        self.latent_dim = int(latent_dim)
        self.prior_variance = float(prior_variance)

        self.q_mu = nn.Parameter(torch.randn(num_data, latent_dim) * init_std)
        self.q_log_var = nn.Parameter(torch.full((num_data, latent_dim), -5.0))

    def _select(self, indices: Tensor | None) -> tuple[Tensor, Tensor]:
        if indices is None:
            return self.q_mu, self.q_log_var
        indices = indices.to(device=self.q_mu.device, dtype=torch.long)
        return self.q_mu.index_select(0, indices), self.q_log_var.index_select(0, indices)

    def rsample(self, *, num_samples: int, indices: Tensor | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        mu, log_var = self._select(indices)
        std = torch.exp(0.5 * log_var)

        eps = torch.randn(
            num_samples,
            mu.shape[0],
            self.latent_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def kl(self, indices: Tensor | None = None) -> Tensor:
        mu, log_var = self._select(indices)
        var = torch.exp(log_var)

        prior_var = mu.new_tensor(self.prior_variance)
        kl = 0.5 * (
            (var + mu.square()) / prior_var
            - 1.0
            + torch.log(prior_var)
            - log_var
        )
        return kl.sum()

class DAGVIModel(nn.Module):
    """DAG model with whitened mean-field SVGPs and optional latent variable layers."""

    def __init__(
        self,
        *,
        dag: DAG,
        root_dims: Mapping[str, int],
        gp_modules: Mapping[str, MeanFieldSVGP],
        likelihoods: Mapping[str, GaussianLikelihood] | None = None,
        latent_variable_dims: Mapping[str, int] | None = None,
        num_data: int | None = None,
        latent_prior_variance: float = 1.0,
        latent_init_std: float = 0.01,
    ) -> None:
        super().__init__()

        self.dag = dag
        self.root_dims = {root: int(dim) for root, dim in root_dims.items()}
        self.latent_variable_dims = {
            node: int(dim)
            for node, dim in (latent_variable_dims or {}).items()
        }

        root_keys = set(self.root_dims)
        dag_root_keys = set(self.dag.roots)
        if root_keys != dag_root_keys:
            missing = sorted(dag_root_keys - root_keys)
            extra = sorted(root_keys - dag_root_keys)
            raise ValueError(
                f"root_dims must match the DAG roots exactly; missing={missing}, extra={extra}."
            )

        unknown_lv_nodes = sorted(set(self.latent_variable_dims) - set(self.dag.non_roots))
        if unknown_lv_nodes:
            raise ValueError(f"latent_variable_dims contains unknown non-root nodes: {unknown_lv_nodes}.")

        if self.latent_variable_dims and num_data is None:
            raise ValueError("num_data is required when latent_variable_dims is non-empty.")

        self.latent_layers = nn.ModuleDict()
        if num_data is not None:
            for node, dim in self.latent_variable_dims.items():
                self.latent_layers[node] = VariationalLatentVariableLayer(
                    num_data=num_data,
                    latent_dim=dim,
                    prior_variance=latent_prior_variance,
                    init_std=latent_init_std,
                )

        module_keys = set(gp_modules)
        latent_keys = set(self.dag.non_roots)
        if module_keys != latent_keys:
            missing = sorted(latent_keys - module_keys)
            extra = sorted(module_keys - latent_keys)
            raise ValueError(
                f"gp_modules must match the DAG non-roots exactly; missing={missing}, extra={extra}."
            )

        module_dict: dict[str, MeanFieldSVGP] = {}
        for node in self.dag.non_roots:
            module = gp_modules[node]
            if not isinstance(module, MeanFieldSVGP):
                raise TypeError(f"gp_modules['{node}'] must be a MeanFieldSVGP instance.")

            expected_input_dim = self._node_input_dim(node)
            actual_input_dim = int(getattr(module.kernel, "input_dim"))

            if actual_input_dim != expected_input_dim:
                raise ValueError(
                    f"gp_modules['{node}'] has kernel input_dim={actual_input_dim}, "
                    f"expected {expected_input_dim}."
                )

            module_dict[node] = module

        self.gp_modules = nn.ModuleDict(module_dict)

        likelihoods = {} if likelihoods is None else dict(likelihoods)
        unknown_likelihood_nodes = sorted(set(likelihoods) - latent_keys)
        if unknown_likelihood_nodes:
            raise ValueError(f"Unknown nodes in likelihoods: {unknown_likelihood_nodes}.")
        self.likelihoods = nn.ModuleDict(likelihoods)

    @property
    def latent_nodes(self) -> tuple[str, ...]:
        return tuple(self.dag.non_roots)

    def _parent_dim(self, parent: str) -> int:
        if parent in self.root_dims:
            return int(self.root_dims[parent])
        if parent in self.dag.non_roots:
            return 1
        raise KeyError(f"Unknown parent node '{parent}'.")

    def _node_input_dim(self, node: str) -> int:
        parent_dim = sum(self._parent_dim(parent) for parent in self.dag.parents_of(node))
        lv_dim = self.latent_variable_dims.get(node, 0)
        return parent_dim + lv_dim

    def _normalise_indices(self, indices: Tensor | None) -> Tensor | None:
        if indices is None:
            return None
        return indices.to(device=next(self.parameters()).device, dtype=torch.long)

    def likelihood(self, node: str) -> GaussianLikelihood:
        if node not in self.likelihoods:
            raise ValueError(f"Missing likelihood for observed node '{node}'.")
        return self.likelihoods[node]

    def gp_kl(self) -> Tensor:
        total = next(self.parameters()).new_tensor(0.0)
        for node in self.latent_nodes:
            total = total + self.gp_modules[node].kl()
        return total

    def latent_kl(self, *, indices: Tensor | None = None, num_data: int | None = None) -> Tensor:
        total = next(self.parameters()).new_tensor(0.0)
        for layer in self.latent_layers.values():
            total = total + layer.kl(indices)

        if indices is not None:
            if num_data is None:
                raise ValueError("num_data is required when scaling minibatch latent KL.")
            batch_size = int(indices.shape[0])
            total = total * (float(num_data) / float(batch_size))

        return total

    def kl(self, *, indices: Tensor | None = None, num_data: int | None = None) -> Tensor:
        return self.gp_kl() + self.latent_kl(indices=indices, num_data=num_data)

    def sample_latent_layer(
        self,
        node: str,
        *,
        num_samples: int,
        indices: Tensor | None,
        batch_size: int | None = None,
        sample_from_prior: bool = False,
    ) -> Tensor | None:
        if node not in self.latent_layers:
            return None

        layer = self.latent_layers[node]

        if sample_from_prior:
            if batch_size is None:
                raise ValueError("batch_size is required when sample_from_prior=True.")

            reference = next(self.parameters())
            prior_std = float(layer.prior_variance) ** 0.5

            return torch.randn(
                num_samples,
                batch_size,
                layer.latent_dim,
                device=reference.device,
                dtype=reference.dtype,
            ) * prior_std

        return layer.rsample(
            num_samples=num_samples,
            indices=indices,
        )

    def elbo(
        self,
        dataset: DAGDataset,
        *,
        indices: Tensor | None = None,
        num_mc_samples: int = 5,
    ) -> Tensor:
        if num_mc_samples <= 0:
            raise ValueError("num_mc_samples must be positive.")

        indices = self._normalise_indices(indices)
        batch_size = dataset.num_data if indices is None else int(indices.shape[0])

        total_ll = elbo_mean_field(
            self,
            dataset,
            indices=indices,
            num_mc_samples=num_mc_samples,
        )
        total_ll = total_ll * (float(dataset.num_data) / float(batch_size))

        return total_ll - self.kl(indices=indices, num_data=dataset.num_data)

    def sample_latents(
        self,
        dataset: DAGDataset,
        *,
        num_samples: int,
        indices: Tensor | None = None,
        sample_latent_prior: bool = False,
    ) -> dict[str, Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        indices = self._normalise_indices(indices)

        return sample_latents_mean_field(
            self,
            dataset,
            num_samples=num_samples,
            indices=indices,
            sample_latent_prior=sample_latent_prior,
        )

    def sample_predictive_likelihood(
        self,
        dataset: DAGDataset,
        *,
        num_samples: int,
        indices: Tensor | None = None,
        sample_latent_prior: bool = False,
    ) -> dict[str, Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        indices = self._normalise_indices(indices)

        latent_samples = sample_latents_mean_field(
            self,
            dataset,
            num_samples=num_samples,
            indices=indices,
            sample_latent_prior=sample_latent_prior,
        )

        predictive_samples: dict[str, Tensor] = {}
        for node, f_samples in latent_samples.items():
            if node not in self.likelihoods:
                continue
            noise = torch.randn_like(f_samples) * torch.sqrt(self.likelihood(node).variance)
            predictive_samples[node] = f_samples + noise

        return predictive_samples
