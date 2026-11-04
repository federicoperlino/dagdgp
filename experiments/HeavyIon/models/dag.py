"""Model construction and training routines for the real-data experiment."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from dagdgp import (
    DAG,
    GaussianLikelihood,
    MeanFieldSVGP,
    DAGVIModel,
    DAGSVIModel,
    StructuredNodeSVGP,
)
from dagdgp.dag_helpers import assemble_node_inputs, prepare_dataset
from dagdgp.kernels.fusion import ConcatFusionKernel, MultifidelityFusionKernel
from dagdgp.utils import inv_softplus
from experiments.HeavyIon.common.data import GMGPRealData
from experiments.HeavyIon.common.real_config import (
    DAGVIRunSpec,
    DAGSVIRunSpec,
    set_all_seeds,
)
from experiments.HeavyIon.common.training import (
    DAGVIFitConfig,
    DAGSVIFitConfig,
    fit_dag_vi,
    fit_dag_svi,
)

ProgressCallback = Callable[[dict[str, float | int | str]], None]
DAGVITrainingHook = Callable[[dict[str, float | int | str], DAGVIModel], None]
DAGSVITrainingHook = Callable[
    [dict[str, float | int | str], DAGSVIModel], None
]


def sample_inducing_inputs(X: Tensor, num_inducing: int) -> Tensor:
    if X.dim() != 2:
        raise ValueError("X must have shape (N, D).")
    if int(num_inducing) <= 0:
        raise ValueError("num_inducing must be positive.")
    if int(X.shape[0]) == 0:
        raise ValueError("Cannot sample inducing inputs from an empty tensor.")
    if int(X.shape[0]) >= int(num_inducing):
        perm = torch.randperm(int(X.shape[0]), device=X.device)[: int(num_inducing)]
        return X[perm].clone()
    idx = torch.randint(0, int(X.shape[0]), (int(num_inducing),), device=X.device)
    return X[idx].clone()


def _svgp_regression_elbo(
    module: MeanFieldSVGP,
    X: Tensor,
    y: Tensor,
    likelihood: GaussianLikelihood,
    *,
    num_data: int,
) -> Tensor:
    mean, var = module.predict_f(X)
    ell = likelihood.expected_log_prob(y, mean, var).sum()
    ell = ell * (float(num_data) / float(X.shape[0]))
    return ell - module.kl()


def _copy_module_inducing_inputs(
    module: MeanFieldSVGP | StructuredNodeSVGP, inducing_inputs: Tensor
) -> None:
    if hasattr(module, "copy_inducing_inputs_"):
        module.copy_inducing_inputs_(inducing_inputs)
        return
    with torch.no_grad():
        module.Z.copy_(inducing_inputs)


def _set_module_inducing_trainable(
    module: MeanFieldSVGP | StructuredNodeSVGP, trainable: bool
) -> None:
    if hasattr(module, "set_inducing_trainable"):
        module.set_inducing_trainable(trainable)
        return
    module.Z.requires_grad_(bool(trainable))


def _fit_local_svgp_predictor(
    *,
    X: Tensor,
    y: Tensor,
    num_inducing: int,
    pretrain_steps: int,
    pretrain_lr: float,
    batch_size: int,
) -> MeanFieldSVGP:
    module = MeanFieldSVGP(
        ConcatFusionKernel(input_dim=int(X.shape[1]), base_kernel="rbf", ard=True),
        sample_inducing_inputs(X, num_inducing),
    ).to(device=X.device, dtype=X.dtype)
    likelihood = GaussianLikelihood(init_variance=0.0001, eps=0.0001).to(
        device=X.device, dtype=X.dtype
    )
    params = [
        param
        for param in list(module.parameters()) + list(likelihood.parameters())
        if param.requires_grad
    ]
    optimizer = torch.optim.Adam(params, lr=float(pretrain_lr))
    num_data = int(X.shape[0])
    batch_size = min(int(batch_size), num_data)
    for _ in range(int(pretrain_steps)):
        if batch_size >= num_data:
            xb = X
            yb = y
        else:
            batch = torch.randperm(num_data, device=X.device)[:batch_size]
            xb = X[batch]
            yb = y[batch]
        optimizer.zero_grad(set_to_none=True)
        elbo = _svgp_regression_elbo(module, xb, yb, likelihood, num_data=num_data)
        (-elbo).backward()
        optimizer.step()
    module.eval()
    return module


def _random_h_inducing_inputs(
    data: GMGPRealData, *, num_inducing: int, seed: int
) -> Tensor:
    """Sample training locations and independent empirical parent coordinates."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x = data.train_data.roots["x"].detach().cpu()
    h_mask = data.train_data.mask["h"].detach().cpu()
    x_h = x[h_mask]
    if num_inducing <= 0 or x_h.shape[0] == 0:
        raise ValueError("Random initialization requires training H inputs and inducing points.")
    if num_inducing <= x_h.shape[0]:
        indices = torch.randperm(x_h.shape[0], generator=generator)[:num_inducing]
    else:
        indices = torch.randint(x_h.shape[0], (num_inducing,), generator=generator)
    columns = [x_h[indices]]
    for node in ("l1", "l2"):
        values = data.train_data.y[node].detach().cpu()
        mask = data.train_data.mask[node].detach().cpu()
        observed = values[mask].reshape(-1)
        if observed.numel() == 0:
            raise ValueError(f"Random initialization requires training observations for {node}.")
        indices = torch.randint(observed.numel(), (num_inducing,), generator=generator)
        columns.append(observed[indices, None])
    return torch.cat(columns, dim=-1)


def initialise_h_inducing_inputs(
    *,
    data: GMGPRealData,
    spec: DAGVIRunSpec | DAGSVIRunSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if spec.h_inducing_init == "random":
        if spec.learn_inducing:
            raise ValueError("Random empirical inducing inputs must remain fixed.")
        return _random_h_inducing_inputs(
            data, num_inducing=spec.num_inducing_h, seed=spec.seed
        ).to(device=device, dtype=dtype)
    x_h = torch.from_numpy(data.node_inputs["h"]).to(device=device, dtype=dtype)
    if spec.h_inducing_init != "predmean":
        raise ValueError(f"Unsupported h_inducing_init '{spec.h_inducing_init}'.")
    x_l1 = torch.from_numpy(data.node_inputs["l1"]).to(device=device, dtype=dtype)
    x_l2 = torch.from_numpy(data.node_inputs["l2"]).to(device=device, dtype=dtype)
    y_l1 = torch.from_numpy(data.node_targets["l1"]).to(device=device, dtype=dtype)
    y_l2 = torch.from_numpy(data.node_targets["l2"]).to(device=device, dtype=dtype)
    z_x = sample_inducing_inputs(x_h, spec.num_inducing_h)
    l1_module = _fit_local_svgp_predictor(
        X=x_l1,
        y=y_l1,
        num_inducing=spec.num_inducing_l1,
        pretrain_steps=spec.pretrain_steps_l1,
        pretrain_lr=spec.pretrain_lr,
        batch_size=spec.batch_size,
    )
    l2_module = _fit_local_svgp_predictor(
        X=x_l2,
        y=y_l2,
        num_inducing=spec.num_inducing_l2,
        pretrain_steps=spec.pretrain_steps_l2,
        pretrain_lr=spec.pretrain_lr,
        batch_size=spec.batch_size,
    )
    with torch.no_grad():
        l1_mean, _ = l1_module.predict_f(z_x)
        l2_mean, _ = l2_module.predict_f(z_x)
    return torch.cat([z_x, l1_mean.unsqueeze(-1), l2_mean.unsqueeze(-1)], dim=-1)


def build_dag_vi_model(
    *, data: GMGPRealData, spec: DAGVIRunSpec, device: torch.device, dtype: torch.dtype
) -> DAGVIModel:
    input_dim = int(data.x_union.shape[1])
    x_l1 = torch.from_numpy(data.node_inputs["l1"]).to(device=device, dtype=dtype)
    x_l2 = torch.from_numpy(data.node_inputs["l2"]).to(device=device, dtype=dtype)
    dag = DAG({"x": [], "l1": ["x"], "l2": ["x"], "h": ["x", "l1", "l2"]})
    gp_modules = {
        "l1": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_l1, spec.num_inducing_l1),
        ),
        "l2": MeanFieldSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_l2, spec.num_inducing_l2),
        ),
    }
    h_kernel = MultifidelityFusionKernel(
        block_names=("x", "l1", "l2"),
        block_dims=(input_dim, 1, 1),
        input_blocks=("x",),
        ard=True,
    )
    h_inducing = initialise_h_inducing_inputs(
        data=data, spec=spec, device=device, dtype=dtype
    )
    gp_modules["h"] = MeanFieldSVGP(h_kernel, h_inducing)
    likelihoods = {
        "l1": GaussianLikelihood(init_variance=0.0001, eps=0.0001),
        "l2": GaussianLikelihood(init_variance=0.0001, eps=0.0001),
        "h": GaussianLikelihood(init_variance=0.0001, eps=0.0001),
    }
    return DAGVIModel(
        dag=dag,
        root_dims={"x": input_dim},
        gp_modules=gp_modules,
        likelihoods=likelihoods,
    )


def build_dag_svi_model(
    *,
    data: GMGPRealData,
    spec: DAGSVIRunSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> DAGSVIModel:
    input_dim = int(data.x_union.shape[1])
    x_l1 = torch.from_numpy(data.node_inputs["l1"]).to(device=device, dtype=dtype)
    x_l2 = torch.from_numpy(data.node_inputs["l2"]).to(device=device, dtype=dtype)
    dag = DAG({"x": [], "l1": ["x"], "l2": ["x"], "h": ["x", "l1", "l2"]})
    gp_modules = {
        "l1": StructuredNodeSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_l1, spec.num_inducing_l1),
        ),
        "l2": StructuredNodeSVGP(
            ConcatFusionKernel(input_dim=input_dim, base_kernel="rbf", ard=True),
            sample_inducing_inputs(x_l2, spec.num_inducing_l2),
        ),
    }
    h_kernel = MultifidelityFusionKernel(
        block_names=("x", "l1", "l2"),
        block_dims=(input_dim, 1, 1),
        input_blocks=("x",),
        ard=True,
    )
    h_inducing = initialise_h_inducing_inputs(
        data=data, spec=spec, device=device, dtype=dtype
    )
    gp_modules["h"] = StructuredNodeSVGP(h_kernel, h_inducing)
    likelihoods = {
        "l1": GaussianLikelihood(init_variance=0.0001, eps=0.0001),
        "l2": GaussianLikelihood(init_variance=0.0001, eps=0.0001),
        "h": GaussianLikelihood(init_variance=0.0001, eps=0.0001),
    }
    model = DAGSVIModel(
        dag=dag,
        root_dims={"x": input_dim},
        gp_modules=gp_modules,
        likelihoods=likelihoods,
        observed_nodes=("l1", "l2", "h"),
    )
    return model


def freeze_inducing_locations(model: DAGVIModel | DAGSVIModel) -> None:
    for module in model.gp_modules.values():
        _set_module_inducing_trainable(module, False)


def configure_inducing_trainability(
    model: DAGVIModel | DAGSVIModel, *, spec: DAGVIRunSpec | DAGSVIRunSpec
) -> None:
    if spec.learn_inducing:
        return
    freeze_inducing_locations(model)


def _pretrain_steps(spec: DAGVIRunSpec, node: str) -> int:
    return {
        "l1": int(spec.pretrain_steps_l1),
        "l2": int(spec.pretrain_steps_l2),
        "h": int(spec.pretrain_steps_h),
    }[node]


def pretrain_dag_vi_model(
    model: DAGVIModel,
    data: GMGPRealData,
    *,
    spec: DAGVIRunSpec,
    callback: ProgressCallback | None = None,
    log_every: int = 50,
    hook: DAGVITrainingHook | None = None,
    h_continue_steps: int = 0,
    h_continue_log_every: int = 500,
) -> list[dict[str, float | int | str]]:
    reference = next(model.parameters())
    train_data = prepare_dataset(
        data.train_data, root_dims=model.root_dims, reference=reference
    )
    root_values = train_data.roots
    latent_values: dict[str, Tensor] = {}
    trace: list[dict[str, float | int | str]] = []
    for node in model.dag.topo:
        if node in model.root_dims or node not in train_data.y:
            continue
        steps = _pretrain_steps(spec, node)
        if node == "h":
            steps += h_continue_steps
        if steps <= 0:
            continue
        Xw = assemble_node_inputs(
            model.dag,
            node=node,
            root_dims=model.root_dims,
            root_values=root_values,
            latent_values=latent_values,
        )
        y, mask = train_data.get_observations(node)
        obs_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if int(obs_idx.numel()) == 0:
            continue
        module = model.gp_modules[node]
        likelihood = model.likelihood(node)
        if node != "h" or spec.h_inducing_init != "random":
            with torch.no_grad():
                _copy_module_inducing_inputs(
                    module, sample_inducing_inputs(Xw[obs_idx], module.num_inducing)
                )
        num_data_node = int(obs_idx.shape[0])
        batch_size = min(int(spec.batch_size), num_data_node)
        params = [
            param
            for param in list(module.parameters()) + list(likelihood.parameters())
            if param.requires_grad
        ]
        optimizer = torch.optim.Adam(params, lr=float(spec.pretrain_lr))
        log_every = max(1, int(log_every))
        for step in range(steps):
            if batch_size >= num_data_node:
                batch = obs_idx
            else:
                batch = obs_idx[
                    torch.randperm(num_data_node, device=reference.device)[:batch_size]
                ]
            optimizer.zero_grad(set_to_none=True)
            elbo = _svgp_regression_elbo(
                module, Xw[batch], y[batch], likelihood, num_data=num_data_node
            )
            (-elbo).backward()
            optimizer.step()
            event = {
                "phase": "pretrain",
                "node": node,
                "step": step + 1,
                "elbo": float(elbo.detach().cpu()),
            }
            interval = (
                h_continue_log_every
                if node == "h" and step + 1 > spec.pretrain_steps_h
                else log_every
            )
            if (step + 1) % interval == 0 or step + 1 == steps:
                trace.append(event)
                if callback is not None:
                    callback(event)
                if hook is not None:
                    hook(event, model)
        with torch.no_grad():
            mean, _ = module.predict_f(Xw)
        latent_values[node] = mean.unsqueeze(-1)
    return trace


def _progress_callback(
    progress_path: Path | None,
) -> tuple[ProgressCallback, Callable[[], None]]:
    if progress_path is None:
        return (lambda _event: None, lambda: None)
    handle = progress_path.open("w", encoding="ascii")

    def _json_safe(event: dict[str, object]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in event.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return payload

    def callback(event: dict[str, float | int | str]) -> None:
        handle.write(json.dumps(_json_safe(event)) + "\n")
        handle.flush()

    return (callback, handle.close)


def _combine_callbacks(*callbacks: ProgressCallback | None) -> ProgressCallback | None:
    active = [callback for callback in callbacks if callback is not None]
    if not active:
        return None

    def wrapped(event: dict[str, float | int | str]) -> None:
        for callback in active:
            callback(event)

    return wrapped


def fit_dag_vi_model(
    data: GMGPRealData,
    *,
    spec: DAGVIRunSpec,
    dtype: torch.dtype,
    device: torch.device,
    progress_path: Path | None = None,
    callback: ProgressCallback | None = None,
    pretrain_log_every: int = 50,
    hook: DAGVITrainingHook | None = None,
) -> tuple[DAGVIModel, float, list[dict[str, float | int | str]], list[float]]:
    set_all_seeds(spec.seed)
    model = build_dag_vi_model(data=data, spec=spec, device=device, dtype=dtype).to(
        device=device, dtype=dtype
    )
    configure_inducing_trainability(model, spec=spec)
    progress_callback, close_progress = _progress_callback(progress_path)
    combined_callback = _combine_callbacks(progress_callback, callback)
    try:
        start = time.perf_counter()
        pretrain_trace = (
            pretrain_dag_vi_model(
                model,
                data,
                spec=spec,
                callback=combined_callback,
                log_every=pretrain_log_every,
                hook=hook,
            )
            if spec.use_warmup
            else []
        )

        def joint_callback(event: dict[str, float | int | str]) -> None:
            if combined_callback is not None:
                combined_callback(event)
            if hook is not None:
                hook(event, model)

        joint_history = fit_dag_vi(
            model,
            data.train_data,
            DAGVIFitConfig(
                steps=int(spec.joint_steps),
                joint_lr=float(spec.joint_lr),
                batch_size=int(spec.batch_size),
                num_mc_samples=int(spec.num_mc_samples),
                likelihood_lr_scale=float(spec.likelihood_lr_scale),
                seed=int(spec.seed),
            ),
            callback=joint_callback,
        )
        elapsed = time.perf_counter() - start
    finally:
        close_progress()
    model.eval()
    return (model, float(elapsed), pretrain_trace, joint_history)


def dag_vi_spec_from_dag_svi_spec(spec: DAGSVIRunSpec) -> DAGVIRunSpec:
    return DAGVIRunSpec(
        seed=int(spec.seed),
        num_inducing_l1=int(spec.num_inducing_l1),
        num_inducing_l2=int(spec.num_inducing_l2),
        num_inducing_h=int(spec.num_inducing_h),
        pretrain_steps_l1=int(spec.pretrain_steps_l1),
        pretrain_steps_l2=int(spec.pretrain_steps_l2),
        pretrain_steps_h=int(spec.pretrain_steps_h),
        pretrain_lr=float(spec.pretrain_lr),
        joint_steps=int(spec.joint_mf_warmstart_steps),
        joint_lr=float(spec.joint_lr),
        likelihood_lr_scale=float(spec.likelihood_lr_scale),
        batch_size=int(spec.batch_size),
        num_mc_samples=int(spec.num_mc_samples),
        pred_samples=int(spec.pred_samples),
        learn_inducing=bool(spec.learn_inducing),
        use_warmup=bool(spec.use_warmup),
        h_inducing_init=spec.h_inducing_init,
    )


def _pack_block_diagonal_posterior(
    model: DAGSVIModel,
    *,
    q_means: dict[str, Tensor],
    q_roots: dict[str, Tensor],
) -> None:
    with torch.no_grad():
        model.posterior.q_mean.zero_()
        for idx, node in enumerate(model.posterior.structure.order):
            sl = model.posterior.block_slice(node)
            model.posterior.q_mean[sl].copy_(q_means[node])
            cov = q_roots[node] @ q_roots[node].transpose(0, 1)
            precision = torch.linalg.inv(cov)
            precision_root = torch.linalg.cholesky(precision)
            raw_block = torch.tril(precision_root)
            if model.posterior.diag_parametrization == "softplus-diag":
                diag = torch.diagonal(raw_block)
                raw_block = (
                    raw_block - torch.diag(diag) + torch.diag(inv_softplus(diag))
                )
            model.posterior.raw_diag_blocks[idx].copy_(raw_block)
        for block in model.posterior.raw_offdiag_blocks.values():
            block.zero_()


def _copy_dag_vi_state_to_dag_svi_model(
    mf_model: DAGVIModel, structured_model: DAGSVIModel
) -> None:
    q_means: dict[str, Tensor] = {}
    q_roots: dict[str, Tensor] = {}
    with torch.no_grad():
        for node in structured_model.dag.non_roots:
            mf_module = mf_model.gp_modules[node]
            structured_module = structured_model.gp_modules[node]
            _copy_module_inducing_inputs(structured_module, mf_module.Z.detach())
            structured_module.kernel.load_state_dict(mf_module.kernel.state_dict())
            structured_model.likelihood(node).load_state_dict(
                mf_model.likelihood(node).state_dict()
            )
            q_means[node] = mf_module.q_mean.detach().clone()
            q_roots[node] = mf_module.q_root().detach().clone()
    _pack_block_diagonal_posterior(structured_model, q_means=q_means, q_roots=q_roots)


def initialise_dag_svi_from_dag_vi(
    structured_model: DAGSVIModel,
    data: GMGPRealData,
    *,
    spec: DAGSVIRunSpec,
    callback: ProgressCallback | None = None,
    pretrain_log_every: int = 50,
    hook: DAGVITrainingHook | None = None,
) -> tuple[list[dict[str, float | int | str]], list[float]]:
    if not spec.use_warmup and int(spec.joint_mf_warmstart_steps) <= 0:
        return ([], [])
    mf_spec = dag_vi_spec_from_dag_svi_spec(spec)
    reference = next(structured_model.parameters())
    mf_model = build_dag_vi_model(
        data=data, spec=mf_spec, device=reference.device, dtype=reference.dtype
    )
    mf_model.to(device=reference.device, dtype=reference.dtype)
    configure_inducing_trainability(mf_model, spec=mf_spec)
    pretrain_trace = (
        pretrain_dag_vi_model(
            mf_model,
            data,
            spec=mf_spec,
            callback=callback,
            log_every=pretrain_log_every,
            hook=hook,
        )
        if spec.use_warmup
        else []
    )

    def mf_callback(event: dict[str, float | int | str]) -> None:
        payload = dict(event)
        if payload.get("phase") == "joint":
            payload["phase"] = "joint_mf"
        if callback is not None:
            callback(payload)
        if hook is not None:
            hook(payload, mf_model)

    joint_history: list[float] = []
    if int(spec.joint_mf_warmstart_steps) > 0:
        joint_history = fit_dag_vi(
            mf_model,
            data.train_data,
            DAGVIFitConfig(
                steps=int(spec.joint_mf_warmstart_steps),
                joint_lr=float(spec.joint_lr),
                batch_size=int(spec.batch_size),
                num_mc_samples=int(spec.num_mc_samples),
                likelihood_lr_scale=float(spec.likelihood_lr_scale),
                seed=int(spec.seed),
            ),
            callback=mf_callback,
        )
    _copy_dag_vi_state_to_dag_svi_model(mf_model, structured_model)
    return (pretrain_trace, joint_history)


def fit_dag_svi_model(
    data: GMGPRealData,
    *,
    spec: DAGSVIRunSpec,
    dtype: torch.dtype,
    device: torch.device,
    progress_path: Path | None = None,
    callback: ProgressCallback | None = None,
    pretrain_log_every: int = 50,
    hook: DAGSVITrainingHook | None = None,
) -> tuple[
    DAGSVIModel,
    float,
    list[dict[str, float | int | str]],
    list[float],
    list[float],
]:
    set_all_seeds(spec.seed)
    model = build_dag_svi_model(data=data, spec=spec, device=device, dtype=dtype).to(
        device=device, dtype=dtype
    )
    configure_inducing_trainability(model, spec=spec)
    progress_callback, close_progress = _progress_callback(progress_path)
    combined_callback = _combine_callbacks(progress_callback, callback)
    try:
        start = time.perf_counter()
        pretrain_trace, joint_mf_history = initialise_dag_svi_from_dag_vi(
            model,
            data,
            spec=spec,
            callback=combined_callback,
            pretrain_log_every=pretrain_log_every,
            hook=hook,
        )

        def joint_callback(event: dict[str, float | int | str]) -> None:
            if combined_callback is not None:
                combined_callback(event)
            if hook is not None:
                hook(event, model)

        joint_history = fit_dag_svi(
            model,
            data.train_data,
            DAGSVIFitConfig(
                steps=int(spec.joint_steps),
                joint_lr=float(spec.joint_lr),
                posterior_lr_scale=float(spec.posterior_lr_scale),
                batch_size=int(spec.batch_size),
                num_mc_samples=int(spec.num_mc_samples),
                likelihood_lr_scale=float(spec.likelihood_lr_scale),
                seed=int(spec.seed),
            ),
            callback=joint_callback,
        )
        elapsed = time.perf_counter() - start
    finally:
        close_progress()
    model.eval()
    return (model, float(elapsed), pretrain_trace, joint_mf_history, joint_history)
