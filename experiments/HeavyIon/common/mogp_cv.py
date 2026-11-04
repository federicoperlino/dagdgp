"""Latent MO-GP prediction on the cross-validation partitions."""

from __future__ import annotations

from dataclasses import replace

import gpytorch
import torch

from dagdgp import DAGDataset
from experiments.HeavyIon.common.data import GMGPRealData
from experiments.HeavyIon.common.metrics import summarise_scalar_samples
from experiments.HeavyIon.models.mogp import TASKS


def subset(dataset: DAGDataset, keep: torch.Tensor) -> DAGDataset:
    return DAGDataset(
        roots={k: v[keep] for k, v in dataset.roots.items()},
        y={k: v[keep] for k, v in dataset.y.items()},
        mask={k: v[keep] for k, v in dataset.mask.items()},
    )


def prepare_fold(data: GMGPRealData, *, joint: bool):
    # Only rows with an observed training target enter the inducing pool and ELBO.
    keep = torch.stack([data.train_data.mask[n] for n in TASKS]).any(dim=0)
    adapted = replace(
        data,
        train_data=subset(data.train_data, keep),
        x_union=data.x_union[keep.numpy()].copy(),
    )
    if joint:
        h_mask = data.test_data.mask["h"]
        adapted = replace(
            adapted,
            x_test=data.x_union[h_mask.numpy()].copy(),
            y_test=data.test_data.y["h"][h_mask].numpy().copy(),
        )
        val_keep = torch.stack([data.test_data.mask[n] for n in TASKS]).any(dim=0)
        validation = subset(data.test_data, val_keep)
    else:
        count = len(data.y_test)
        validation = DAGDataset(
            roots={"x": torch.from_numpy(data.x_test)},
            y={
                "l1": torch.zeros(count, dtype=torch.float64),
                "l2": torch.zeros(count, dtype=torch.float64),
                "h": torch.from_numpy(data.y_test),
            },
            mask={
                "l1": torch.zeros(count, dtype=torch.bool),
                "l2": torch.zeros(count, dtype=torch.bool),
                "h": torch.ones(count, dtype=torch.bool),
            },
        )
    return adapted, validation


def posterior_payload(model, dataset, *, device, dtype, num_samples=1024):
    x = dataset.roots["x"].to(device=device, dtype=dtype)
    with (
        torch.no_grad(),
        gpytorch.settings.cholesky_jitter(1e-5),
        gpytorch.settings.max_cholesky_size(4096),
    ):
        dist = model(x)
        draws = dist.rsample(torch.Size([num_samples])).detach().cpu()
        means, variances = dist.mean.detach().cpu(), dist.variance.detach().cpu()
    if not torch.isfinite(draws).all():
        raise FloatingPointError("Non-finite MO-GP posterior samples")
    node_samples = {n: draws[:, :, i] for i, n in enumerate(TASKS)}
    return {
        "num_samples": num_samples,
        "roots": {k: v.cpu() for k, v in dataset.roots.items()},
        "y": {k: v.cpu() for k, v in dataset.y.items()},
        "mask": {k: v.cpu() for k, v in dataset.mask.items()},
        "node_samples": node_samples,
        "summary": {n: summarise_scalar_samples(v) for n, v in node_samples.items()},
        "analytic_summary": {
            n: {"mean": means[:, i], "variance": variances[:, i]}
            for i, n in enumerate(TASKS)
        },
        "posterior_type": "joint latent function posterior; likelihood noise excluded",
    }
