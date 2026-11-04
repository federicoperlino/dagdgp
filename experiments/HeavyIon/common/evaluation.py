"""Validation callbacks and final scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.HeavyIon.common.metrics import compute_metrics
from experiments.HeavyIon.common.real_outputs import posterior_payload

NODES = ("l1", "l2", "h")


class ValidationTrace:
    """Evaluate and record validation metrics during training."""

    def __init__(self, data, *, device, spec, joint, smoke=False):
        self.data, self.device, self.joint = data, device, joint
        self.samples = 8 if smoke else 256
        self.every = 1 if smoke else 500
        self.totals = {
            "joint": spec.joint_steps,
            "joint_mf": getattr(spec, "joint_mf_warmstart_steps", 0),
        }
        self.records = []

    def __call__(self, event, model):
        phase, step = event["phase"], event["step"]
        if phase != "pretrain" and step % self.every and step != self.totals.get(phase):
            return
        payload = posterior_payload(
            model,
            self.data.test_data.to(device=self.device, dtype=torch.float64),
            num_samples=self.samples,
        )
        if self.joint:
            scores = _evaluate_payload(
                payload,
                baselines={n: float(self.data.node_targets[n].mean()) for n in NODES},
            )
            record = {
                f"validation_{metric}_{node}": value
                for node, values in scores.items()
                for metric, value in values.items()
            }
        else:
            summary = payload["summary"]["h"]
            scores = compute_metrics(
                self.data.y_test,
                summary["mean"].numpy(),
                summary["variance"].numpy(),
                self.data.baseline_mean,
            )
            record = {f"validation_{metric}": value for metric, value in scores.items()}
        self.records.append(dict(event, **record))


def _macro_average(metrics_by_node: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([metrics_by_node[node][metric] for node in NODES]))
        for metric in ("rmse", "n_rmse", "crps")
    }


def _evaluate_payload(
    payload: dict[str, Any],
    *,
    baselines: dict[str, float],
) -> dict[str, dict[str, float]]:
    metrics_by_node: dict[str, dict[str, float]] = {}
    for node in NODES:
        y = payload["y"][node].numpy()
        mask = payload["mask"][node].numpy().astype(bool)
        if int(mask.sum()) == 0:
            raise ValueError(
                f"No observed points found for node '{node}' in evaluation payload."
            )
        mean = payload["summary"][node]["mean"].numpy()
        var = payload["summary"][node]["variance"].numpy()
        metrics_by_node[node] = compute_metrics(
            y[mask], mean[mask], var[mask], float(baselines[node])
        )
    metrics_by_node["macro"] = _macro_average(metrics_by_node)
    return metrics_by_node
