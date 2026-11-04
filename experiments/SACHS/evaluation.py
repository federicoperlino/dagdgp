"""Predictive samples and reported Sachs scores."""

from __future__ import annotations

import random

import gpytorch
import numpy as np
import torch

from config import Config, NODES
from data import SachsData
from models import LMCMultitaskGP, set_seed


def exact_mogp_row_samples(
    model: LMCMultitaskGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    test_x: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((num_samples, len(test_x), len(NODES)), dtype=np.float64)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-5):
        for row in range(len(test_x)):
            predictive = likelihood(model(test_x[row : row + 1]))
            mean = predictive.mean[0].detach().cpu().numpy()
            covariance = predictive.covariance_matrix.detach().cpu().numpy()
            covariance = 0.5 * (covariance + covariance.T)
            jitter = 1e-10
            while True:
                try:
                    root = np.linalg.cholesky(
                        covariance + jitter * np.eye(len(NODES))
                    )
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
                    if jitter > 1e-3:
                        raise
            standard_normal = rng.standard_normal((num_samples, len(NODES)))
            samples[:, row, :] = mean + standard_normal @ root.T
    return samples


def metric_block(
    y: np.ndarray, samples: np.ndarray, train_values: np.ndarray
) -> dict[str, float]:
    predictive_mean = samples.mean(axis=0)
    first = np.mean(np.abs(samples - y[None, :, :]), axis=0)
    ordered = np.sort(samples, axis=0)
    coefficients = (
        2 * np.arange(samples.shape[0]) - samples.shape[0] + 1
    ).reshape(-1, 1, 1)
    crps = first - np.sum(coefficients * ordered, axis=0) / samples.shape[0] ** 2

    scale_mean = train_values.mean(axis=0)
    scale_std = np.maximum(train_values.std(axis=0), 1e-12)
    y_scaled = (y - scale_mean) / scale_std
    samples_scaled = (samples - scale_mean[None, None, :]) / scale_std[None, None, :]
    energy_first = np.linalg.norm(
        samples_scaled - y_scaled[None, :, :], axis=-1
    ).mean(axis=0)
    energy_second = np.linalg.norm(
        samples_scaled[0::2] - samples_scaled[1::2], axis=-1
    ).mean(axis=0)
    return {
        "rmse": float(np.sqrt(np.mean((predictive_mean - y) ** 2))),
        "crps": float(crps.mean()),
        "energy_standardised": float((energy_first - 0.5 * energy_second).mean()),
    }


def dag_predictions(model, dataset, data: SachsData, cfg: Config, seed: int) -> np.ndarray:
    device = next(model.parameters()).device
    indices = np.flatnonzero(data.test)
    predictive_seed = 271_828 + 1_000 * seed
    if cfg.protocol != "scalability":
        set_seed(predictive_seed)
        with torch.no_grad():
            samples = model.sample_predictive_likelihood(
                dataset, num_samples=cfg.predictive_samples,
                indices=torch.as_tensor(indices, dtype=torch.long, device=device),
                sample_latent_prior=False,
            )
        return np.stack([samples[n].squeeze(-1).cpu().numpy() for n in NODES], axis=-1)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    was_training = model.training
    chunks = []
    try:
        set_seed(predictive_seed)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), 128):
                rows = indices[start:start + 128]
                samples = model.sample_predictive_likelihood(
                    dataset, num_samples=cfg.predictive_samples,
                    indices=torch.as_tensor(rows, dtype=torch.long, device=device),
                    sample_latent_prior=False,
                )
                chunk = np.stack([samples[n].squeeze(-1).cpu().numpy() for n in NODES], axis=-1)
                chunks.append(chunk + data.offsets[rows][None, :, :])
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        model.train(was_training)
    return np.concatenate(chunks, axis=1)


def evaluate(data: SachsData, samples: np.ndarray) -> dict[str, float]:
    return metric_block(
        data.targets.loc[data.test, list(NODES)].to_numpy(np.float64),
        samples,
        data.targets.loc[data.train, list(NODES)].to_numpy(np.float64),
    )
