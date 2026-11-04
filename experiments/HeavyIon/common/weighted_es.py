"""Energy score with equal scaling for the three fidelity blocks."""

from __future__ import annotations

from typing import Any

import torch

NODES = ("l1", "l2", "h")


def weighted_energy_score(
    train_payload: dict[str, Any],
    validation_payload: dict[str, Any],
) -> dict[str, Any]:
    block_sizes: dict[str, int] = {}
    sigmas: dict[str, float] = {}
    y_blocks: list[torch.Tensor] = []
    sample_blocks: list[torch.Tensor] = []

    for node in NODES:
        train_mask = train_payload["mask"][node].to(dtype=torch.bool)
        y_train = train_payload["y"][node][train_mask].to(dtype=torch.float64)
        sigma = float(y_train.std(unbiased=False).item())
        sigmas[node] = sigma

        validation_mask = validation_payload["mask"][node].to(dtype=torch.bool)
        y_validation = validation_payload["y"][node][validation_mask].to(
            dtype=torch.float64
        )
        samples_validation = validation_payload["node_samples"][node][
            :, validation_mask
        ].to(dtype=torch.float64)
        block_sizes[node] = int(y_validation.numel())

        scale = float((block_sizes[node] * sigma**2) ** -0.5)
        y_blocks.append(y_validation * scale)
        sample_blocks.append(samples_validation * scale)

    y_concat = torch.cat(y_blocks, dim=0)
    samples_concat = torch.cat(sample_blocks, dim=1)
    term_observation = torch.sqrt(((samples_concat - y_concat) ** 2).sum(dim=1)).mean()
    term_pairwise = 0.5 * torch.cdist(samples_concat, samples_concat, p=2).mean()
    score = float((term_observation - term_pairwise).item())

    return {
        "weighted_es": score,
        "block_sizes": block_sizes,
        "training_sigmas": sigmas,
        "num_samples": int(samples_concat.shape[0]),
    }
