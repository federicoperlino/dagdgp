"""Fixed repeated and pooled cross-validation partitions."""

from __future__ import annotations

import copy

import numpy as np
import torch

from dagdgp import DAGDataset
from experiments.HeavyIon.common.data import GMGPRealData

NODES = ("l1", "l2", "h")


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def make_fold_data(
    full: GMGPRealData, *, val_indices: np.ndarray, fold: int
) -> GMGPRealData:
    y = {node: values.clone() for node, values in full.train_data.y.items()}
    mask = {node: values.clone() for node, values in full.train_data.mask.items()}
    roots = {name: values.clone() for name, values in full.train_data.roots.items()}

    h_mask = mask["h"].clone().to(dtype=torch.bool)
    val_tensor = torch.as_tensor(val_indices, dtype=torch.long)
    h_mask[val_tensor] = False
    mask["h"] = h_mask

    y_h_union = _tensor_to_numpy(y["h"])
    train_h_indices = np.flatnonzero(_tensor_to_numpy(h_mask).astype(bool))
    x_val = full.x_union[val_indices].copy()
    y_val = y_h_union[val_indices].copy()
    x_h_train = full.x_union[train_h_indices].copy()
    y_h_train = y_h_union[train_h_indices].copy()

    metadata = copy.deepcopy(full.metadata)
    metadata["test_size"] = int(x_val.shape[0])
    metadata["train_sizes"] = dict(metadata["train_sizes"])
    metadata["train_sizes"]["h"] = int(x_h_train.shape[0])
    metadata["target_summary"] = copy.deepcopy(metadata["target_summary"])
    metadata["target_summary"]["h"] = _summary(y_h_train)
    metadata["target_summary"]["validation_h"] = _summary(y_val)
    metadata["cv"] = {
        "type": "nested_high_fidelity_masking",
        "fold": int(fold),
        "validation_union_indices": [int(idx) for idx in val_indices.tolist()],
        "train_h_count": int(x_h_train.shape[0]),
        "validation_h_count": int(x_val.shape[0]),
        "low_fidelity_masks_unchanged": True,
    }

    return GMGPRealData(
        train_data=DAGDataset(roots=roots, y=y, mask=mask),
        test_data=DAGDataset(roots={"x": torch.from_numpy(x_val)}, y={}),
        x_union=full.x_union.copy(),
        x_test=x_val,
        y_test=y_val,
        baseline_mean=float(np.mean(y_h_train)),
        node_inputs={
            "l1": full.node_inputs["l1"].copy(),
            "l2": full.node_inputs["l2"].copy(),
            "h": x_h_train,
        },
        node_targets={
            "l1": full.node_targets["l1"].copy(),
            "l2": full.node_targets["l2"].copy(),
            "h": y_h_train,
        },
        metadata=metadata,
    )


def make_joint_fold_data(
    full: GMGPRealData,
    *,
    val_indices_by_node: dict[str, np.ndarray],
    fold: int,
) -> GMGPRealData:
    y_train = {node: values.clone() for node, values in full.train_data.y.items()}
    train_mask = {node: values.clone() for node, values in full.train_data.mask.items()}
    roots = {name: values.clone() for name, values in full.train_data.roots.items()}

    val_mask: dict[str, torch.Tensor] = {}
    train_indices_by_node: dict[str, np.ndarray] = {}
    validation_targets: dict[str, np.ndarray] = {}
    train_targets: dict[str, np.ndarray] = {}
    train_inputs: dict[str, np.ndarray] = {}
    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}

    for node in NODES:
        node_train_mask = train_mask[node].clone().to(dtype=torch.bool)
        node_val_mask = torch.zeros_like(node_train_mask, dtype=torch.bool)
        val_indices = np.asarray(val_indices_by_node[node], dtype=np.int64)
        val_tensor = torch.as_tensor(val_indices, dtype=torch.long)
        node_train_mask[val_tensor] = False
        node_val_mask[val_tensor] = True
        train_mask[node] = node_train_mask
        val_mask[node] = node_val_mask

        y_union = _tensor_to_numpy(y_train[node])
        node_train_indices = np.flatnonzero(
            _tensor_to_numpy(node_train_mask).astype(bool)
        )
        train_indices_by_node[node] = node_train_indices
        validation_targets[node] = y_union[val_indices].copy()
        train_targets[node] = y_union[node_train_indices].copy()
        train_inputs[node] = full.x_union[node_train_indices].copy()
        train_counts[node] = int(node_train_indices.shape[0])
        validation_counts[node] = int(val_indices.shape[0])

    metadata = copy.deepcopy(full.metadata)
    metadata["test_size"] = int(sum(validation_counts.values()))
    metadata["train_sizes"] = dict(metadata["train_sizes"])
    metadata["target_summary"] = copy.deepcopy(metadata["target_summary"])
    for node in NODES:
        metadata["train_sizes"][node] = train_counts[node]
        metadata["target_summary"][node] = _summary(train_targets[node])
        metadata["target_summary"][f"validation_{node}"] = _summary(
            validation_targets[node]
        )
    metadata["cv"] = {
        "type": "joint_multifidelity_masking",
        "fold": int(fold),
        "validation_union_indices": {
            node: [int(idx) for idx in val_indices_by_node[node].tolist()]
            for node in NODES
        },
        "train_counts": train_counts,
        "validation_counts": validation_counts,
    }

    validation_y = {node: values.clone() for node, values in full.train_data.y.items()}
    validation_data = DAGDataset(roots=roots, y=validation_y, mask=val_mask)

    return GMGPRealData(
        train_data=DAGDataset(roots=roots, y=y_train, mask=train_mask),
        test_data=validation_data,
        x_union=full.x_union.copy(),
        x_test=full.x_union.copy(),
        y_test=validation_targets["h"].copy(),
        baseline_mean=float(np.mean(train_targets["h"])),
        node_inputs=train_inputs,
        node_targets=train_targets,
        metadata=metadata,
    )
