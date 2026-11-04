"""Loader for the prepared HeavyIon benchmark snapshot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dagdgp import DAGDataset

SNAPSHOT_NAME = "gmgp_real_dataset.npz"
_env_real_data_dir = os.environ.get("GMGP_REAL_DATA_DIR")
if _env_real_data_dir:
    DEFAULT_REAL_DATA_DIR = Path(_env_real_data_dir)
else:
    DEFAULT_REAL_DATA_DIR = None


@dataclass(frozen=True)
class GMGPRealData:
    train_data: DAGDataset
    test_data: DAGDataset
    x_union: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    baseline_mean: float
    node_inputs: dict[str, np.ndarray]
    node_targets: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _row_key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(row, dtype=np.float64), 12))


def _overlap_size(x_left: np.ndarray, x_right: np.ndarray) -> int:
    left = {_row_key(row) for row in x_left}
    right = {_row_key(row) for row in x_right}
    return len(left & right)


def _is_subset(x_small: np.ndarray, x_large: np.ndarray) -> bool:
    large = {_row_key(row) for row in x_large}
    return all((_row_key(row) in large for row in x_small))


def _summarise_targets(targets: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for node, values in targets.items():
        summary[node] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _stable_row_union(*arrays: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for array in arrays:
        for row in array:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(np.asarray(row, dtype=np.float64).copy())
    return np.vstack(rows)


def _masked_targets(
    x_union: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {_row_key(row): idx for idx, row in enumerate(x_union)}
    y = np.zeros(x_union.shape[0], dtype=np.float64)
    mask = np.zeros(x_union.shape[0], dtype=bool)
    for row, value in zip(x_obs, y_obs):
        idx = lookup[_row_key(row)]
        y[idx] = float(value)
        mask[idx] = True
    return (y, mask)


def _resolve_snapshot_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "data_dir must be provided explicitly or via GMGP_REAL_DATA_DIR."
        )
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / SNAPSHOT_NAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected prepared HeavyIon snapshot at '{candidate}'."
        )
    return candidate


def build_real_data(
    *,
    data_dir: Path | None = DEFAULT_REAL_DATA_DIR,
    include_test_h_in_train: bool = False,
) -> GMGPRealData:
    snapshot_path = _resolve_snapshot_path(data_dir)
    with np.load(snapshot_path, allow_pickle=False) as payload:
        x_test = np.asarray(payload["x_test"], dtype=np.float64)
        y_test = np.asarray(payload["y_test"], dtype=np.float64)
        x_l1 = np.asarray(payload["x_l1"], dtype=np.float64)
        x_l2 = np.asarray(payload["x_l2"], dtype=np.float64)
        x_h = np.asarray(payload["x_h"], dtype=np.float64)
        y_l1 = np.asarray(payload["y_l1"], dtype=np.float64)
        y_l2 = np.asarray(payload["y_l2"], dtype=np.float64)
        y_h = np.asarray(payload["y_h"], dtype=np.float64)
    if include_test_h_in_train:
        x_h_all = np.vstack([x_h, x_test])
        y_h_all = np.concatenate([y_h, y_test])
    else:
        x_h_all = x_h
        y_h_all = y_h
    x_union = _stable_row_union(x_l1, x_l2, x_h_all)
    y_l1_union, mask_l1 = _masked_targets(x_union, x_l1, y_l1)
    y_l2_union, mask_l2 = _masked_targets(x_union, x_l2, y_l2)
    y_h_union, mask_h = _masked_targets(x_union, x_h_all, y_h_all)
    train_data = DAGDataset(
        roots={"x": torch.from_numpy(x_union)},
        y={
            "l1": torch.from_numpy(y_l1_union),
            "l2": torch.from_numpy(y_l2_union),
            "h": torch.from_numpy(y_h_union),
        },
        mask={
            "l1": torch.from_numpy(mask_l1),
            "l2": torch.from_numpy(mask_l2),
            "h": torch.from_numpy(mask_h),
        },
    )
    test_data = DAGDataset(roots={"x": torch.from_numpy(x_test)}, y={})
    metadata: dict[str, Any] = {
        "snapshot_path": str(snapshot_path),
        "include_test_h_in_train": bool(include_test_h_in_train),
        "input_dim": int(x_union.shape[1]),
        "train_sizes": {
            "l1": int(x_l1.shape[0]),
            "l2": int(x_l2.shape[0]),
            "h": int(x_h_all.shape[0]),
            "union": int(x_union.shape[0]),
        },
        "test_size": int(x_test.shape[0]),
        "nestedness": {
            "h_subset_l1": bool(_is_subset(x_h_all, x_l1)),
            "h_subset_l2": bool(_is_subset(x_h_all, x_l2)),
            "l1_l2_overlap": int(_overlap_size(x_l1, x_l2)),
            "train_test_overlap": int(_overlap_size(x_union, x_test)),
        },
        "target_summary": _summarise_targets(
            {"l1": y_l1, "l2": y_l2, "h": y_h, "test": y_test}
        ),
        "files": {"snapshot": SNAPSHOT_NAME},
    }
    baseline_mean = float(np.mean(np.concatenate([y_l1, y_l2, y_h_all])))
    return GMGPRealData(
        train_data=train_data,
        test_data=test_data,
        x_union=x_union,
        x_test=x_test,
        y_test=y_test,
        baseline_mean=baseline_mean,
        node_inputs={"l1": x_l1, "l2": x_l2, "h": x_h_all},
        node_targets={"l1": y_l1, "l2": y_l2, "h": y_h_all},
        metadata=metadata,
    )


__all__ = ["DEFAULT_REAL_DATA_DIR", "GMGPRealData", "SNAPSHOT_NAME", "build_real_data"]
