"""Load Sachs measurements and the fixed train/test splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dagdgp import DAGDataset
from config import CONFIGS, NODES, OBSERVED, ROOTS, SEEDS

DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SachsData:
    targets: pd.DataFrame
    inputs: pd.DataFrame
    train: np.ndarray
    test: np.ndarray
    offsets: np.ndarray | None
    split_hash: str


def load_data(protocol: str, seed: int, data_dir: Path | None = None) -> SachsData:
    if protocol not in CONFIGS or seed not in SEEDS:
        raise ValueError("Select a Sachs protocol and a split seed from 0 to 4.")
    data_dir = DIRECTORY / "private" / "data" if data_dir is None else data_dir
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    names = sorted(manifest["data_sha256"])
    if protocol != "scalability":
        names = ["cd3cd28_icam2.csv"]
    frames, conditions = [], []
    for name in names:
        path = data_dir / name
        if sha256(path.read_bytes()).hexdigest() != manifest["data_sha256"][name]:
            raise ValueError(f"Data checksum mismatch: {name}")
        frame = pd.read_csv(path)
        if set(frame.columns) != set(ROOTS) | set(NODES):
            raise ValueError(f"Unexpected columns in {name}")
        if protocol == "scalability":
            frame = frame.loc[:, [*ROOTS, *NODES]]
        frames.append(frame)
        conditions.append(np.full(len(frame), Path(name).stem, dtype=object))
    raw = pd.concat(frames, ignore_index=True)
    expected = 11_672 if protocol == "scalability" else 902
    if len(raw) != expected or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Unexpected Sachs data dimensions or non-finite measurements.")
    logged = np.log(raw + 1e-3)
    split_path = DIRECTORY / "splits" / f"{protocol}_{seed}.npz"
    split_hash = sha256(split_path.read_bytes()).hexdigest()
    if split_hash != manifest["split_sha256"][split_path.name]:
        raise ValueError(f"Split checksum mismatch: {split_path.name}")
    with np.load(split_path, allow_pickle=False) as split:
        train, test = split["train"], split["test"]
    if (
        train.dtype != bool or test.dtype != bool
        or train.shape != (expected,) or test.shape != (expected,)
        or not np.array_equal(train, ~test)
    ):
        raise ValueError("Invalid train/test masks.")
    inputs, offsets = logged, None
    if protocol == "scalability":
        condition = np.concatenate(conditions)
        training = logged.loc[train].copy()
        training.insert(0, "condition", condition[train])
        means = training.groupby("condition", sort=True).mean()
        row_means = means.loc[list(condition), logged.columns].to_numpy(np.float64)
        inputs = logged.copy()
        inputs.loc[:, :] = logged.to_numpy(np.float64) - row_means
        offsets = means.loc[list(condition), list(NODES)].to_numpy(np.float64)
    return SachsData(logged, inputs, train, test, offsets, split_hash)


def make_dataset(data: SachsData, device: torch.device) -> DAGDataset:
    roots = {
        node: torch.as_tensor(
            data.inputs[node].to_numpy(np.float64).copy(), device=device
        ).unsqueeze(-1)
        for node in ROOTS
    }
    values, masks = {}, {}
    for node in NODES:
        observed = np.ones(len(data.inputs), dtype=bool)
        if node not in OBSERVED:
            observed[data.test] = False
        y = data.inputs[node].to_numpy(np.float64).copy()
        y[~observed] = 0.0
        values[node] = torch.as_tensor(y, device=device)
        masks[node] = torch.as_tensor(observed, device=device)
    return DAGDataset(roots=roots, y=values, mask=masks)
