"""HeavyIon baselines with one low-fidelity parent or deep GPs with two latent layers."""

from __future__ import annotations

import torch

from dagdgp import DAG, GaussianLikelihood, MeanFieldSVGP, DAGVIModel
from dagdgp.kernels.fusion import ConcatFusionKernel, MultifidelityFusionKernel
from experiments.HeavyIon.common.real_config import DAGVIRunSpec
from experiments.HeavyIon.models.dag import (
    _fit_local_svgp_predictor,
    sample_inducing_inputs,
)


def _proxy_inducing(
    x_h: torch.Tensor,
    y_h: torch.Tensor,
    *,
    num_inducing: int,
    spec: DAGVIRunSpec,
) -> torch.Tensor:
    proxy = _fit_local_svgp_predictor(
        X=x_h,
        y=y_h,
        num_inducing=max(int(num_inducing), 8),
        pretrain_steps=spec.pretrain_steps_h,
        pretrain_lr=spec.pretrain_lr,
        batch_size=spec.batch_size,
    )
    z_x = sample_inducing_inputs(x_h, num_inducing)
    with torch.no_grad():
        proxy_mean, _ = proxy.predict_f(z_x)
    return proxy_mean.unsqueeze(-1)


def _multifidelity_dgp_inputs(
    data,
    spec: DAGVIRunSpec,
    *,
    num_inducing: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initialise (x, L1, L2) inputs using LF predictive means at H locations."""
    x_h = torch.from_numpy(data.node_inputs["h"]).to(device=device, dtype=dtype)
    z_x = sample_inducing_inputs(x_h, num_inducing)
    parent_means = []
    for parent, num_inducing, pretrain_steps in (
        ("l1", spec.num_inducing_l1, spec.pretrain_steps_l1),
        ("l2", spec.num_inducing_l2, spec.pretrain_steps_l2),
    ):
        x_parent = torch.from_numpy(data.node_inputs[parent]).to(
            device=device, dtype=dtype
        )
        y_parent = torch.from_numpy(data.node_targets[parent]).to(
            device=device, dtype=dtype
        )
        predictor = _fit_local_svgp_predictor(
            X=x_parent,
            y=y_parent,
            num_inducing=num_inducing,
            pretrain_steps=pretrain_steps,
            pretrain_lr=spec.pretrain_lr,
            batch_size=spec.batch_size,
        )
        with torch.no_grad():
            mean, _ = predictor.predict_f(z_x)
        parent_means.append(mean.unsqueeze(-1))
    return torch.cat([z_x, *parent_means], dim=-1)


def build_input_dgp2_mf_inputs_model(
    data,
    spec: DAGVIRunSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DAGVIModel:
    """Build a DGP baseline using x and predictions from L1 and L2 as inputs."""
    input_dim = int(data.x_union.shape[1])
    x_l1 = torch.from_numpy(data.node_inputs["l1"]).to(device=device, dtype=dtype)
    x_l2 = torch.from_numpy(data.node_inputs["l2"]).to(device=device, dtype=dtype)
    x_h = torch.from_numpy(data.node_inputs["h"]).to(device=device, dtype=dtype)
    y_h = torch.from_numpy(data.node_targets["h"]).to(device=device, dtype=dtype)
    num_inducing_deep = int(x_h.shape[0])
    dag = DAG(
        {
            "x": [],
            "l1": ["x"],
            "l2": ["x"],
            "z1": ["x", "l1", "l2"],
            "z2": ["z1"],
            "h": ["z2"],
        }
    )
    gp_modules = {
        "l1": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_l1, spec.num_inducing_l1),
        ),
        "l2": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_l2, spec.num_inducing_l2),
        ),
        "z1": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim + 2, base_kernel="rbf", ard=True),
            _multifidelity_dgp_inputs(
                data,
                spec,
                num_inducing=num_inducing_deep,
                device=device,
                dtype=dtype,
            ),
        ),
        "z2": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=1, base_kernel="rbf", ard=True),
            _proxy_inducing(x_h, y_h, num_inducing=num_inducing_deep, spec=spec),
        ),
        "h": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=1, base_kernel="rbf", ard=True),
            _proxy_inducing(x_h, y_h, num_inducing=num_inducing_deep, spec=spec),
        ),
    }
    likelihoods = {
        "l1": GaussianLikelihood(init_variance=1e-4, eps=1e-4),
        "l2": GaussianLikelihood(init_variance=1e-4, eps=1e-4),
        "h": GaussianLikelihood(init_variance=1e-4, eps=1e-4),
    }
    return DAGVIModel(
        dag=dag,
        root_dims={"x": input_dim},
        gp_modules=gp_modules,
        likelihoods=likelihoods,
    )


def _single_parent_h_inducing(
    data,
    spec: DAGVIRunSpec,
    *,
    parent: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_h = torch.from_numpy(data.node_inputs["h"]).to(device=device, dtype=dtype)
    z_x = sample_inducing_inputs(x_h, spec.num_inducing_h)
    x_parent = torch.from_numpy(data.node_inputs[parent]).to(device=device, dtype=dtype)
    y_parent = torch.from_numpy(data.node_targets[parent]).to(
        device=device, dtype=dtype
    )
    num_inducing = spec.num_inducing_l1 if parent == "l1" else spec.num_inducing_l2
    predictor = _fit_local_svgp_predictor(
        X=x_parent,
        y=y_parent,
        num_inducing=num_inducing,
        pretrain_steps=spec.pretrain_steps_l1
        if parent == "l1"
        else spec.pretrain_steps_l2,
        pretrain_lr=spec.pretrain_lr,
        batch_size=spec.batch_size,
    )
    with torch.no_grad():
        parent_mean, _ = predictor.predict_f(z_x)
    return torch.cat([z_x, parent_mean.unsqueeze(-1)], dim=-1)


def build_single_parent_model(
    data,
    spec: DAGVIRunSpec,
    *,
    parent: str,
    device: torch.device,
    dtype: torch.dtype,
) -> DAGVIModel:
    input_dim = int(data.x_union.shape[1])
    x_parent = torch.from_numpy(data.node_inputs[parent]).to(device=device, dtype=dtype)
    num_inducing = spec.num_inducing_l1 if parent == "l1" else spec.num_inducing_l2
    dag = DAG({"x": [], parent: ["x"], "h": ["x", parent]})
    gp_modules = {
        parent: MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_parent, num_inducing),
        ),
        "h": MeanFieldSVGP(
            MultifidelityFusionKernel(
                block_names=("x", parent),
                block_dims=(input_dim, 1),
                input_blocks=("x",),
                ard=True,
            ),
            _single_parent_h_inducing(
                data,
                spec,
                parent=parent,
                device=device,
                dtype=dtype,
            ),
        ),
    }
    likelihoods = {
        parent: GaussianLikelihood(init_variance=1e-4, eps=1e-4),
        "h": GaussianLikelihood(init_variance=1e-4, eps=1e-4),
    }
    return DAGVIModel(
        dag=dag,
        root_dims={"x": input_dim},
        gp_modules=gp_modules,
        likelihoods=likelihoods,
    )


def build_input_dgp2_model(
    data,
    spec: DAGVIRunSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DAGVIModel:
    """Build a DGP baseline with architecture x -> z1 -> z2 -> h, trained only on H."""
    input_dim = int(data.x_union.shape[1])
    x_h = torch.from_numpy(data.node_inputs["h"]).to(device=device, dtype=dtype)
    y_h = torch.from_numpy(data.node_targets["h"]).to(device=device, dtype=dtype)
    num_inducing = int(x_h.shape[0])
    dag = DAG({"x": [], "z1": ["x"], "z2": ["z1"], "h": ["z2"]})
    gp_modules = {
        "z1": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_h, num_inducing),
        ),
        "z2": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=1, base_kernel="rbf", ard=True),
            _proxy_inducing(x_h, y_h, num_inducing=num_inducing, spec=spec),
        ),
        "h": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=1, base_kernel="rbf", ard=True),
            _proxy_inducing(x_h, y_h, num_inducing=num_inducing, spec=spec),
        ),
    }
    return DAGVIModel(
        dag=dag,
        root_dims={"x": input_dim},
        gp_modules=gp_modules,
        likelihoods={"h": GaussianLikelihood(init_variance=1e-4, eps=1e-4)},
    )
