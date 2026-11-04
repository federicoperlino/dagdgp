"""Configuration and runtime helpers for the real-data runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DAGVIRunSpec:
    seed: int = 0
    num_inducing_l1: int = 200
    num_inducing_l2: int = 200
    num_inducing_h: int = 25
    pretrain_steps_l1: int = 2000
    pretrain_steps_l2: int = 2000
    pretrain_steps_h: int = 1000
    pretrain_lr: float = 0.01
    joint_steps: int = 10000
    joint_lr: float = 0.01
    likelihood_lr_scale: float = 0.1
    batch_size: int = 512
    num_mc_samples: int = 10
    pred_samples: int = 1024
    learn_inducing: bool = False
    use_warmup: bool = True
    h_inducing_init: str = "predmean"


@dataclass(frozen=True)
class DAGSVIRunSpec:
    seed: int = 0
    num_inducing_l1: int = 200
    num_inducing_l2: int = 200
    num_inducing_h: int = 25
    pretrain_steps_l1: int = 2000
    pretrain_steps_l2: int = 2000
    pretrain_steps_h: int = 1000
    pretrain_lr: float = 0.01
    joint_steps: int = 10000
    joint_lr: float = 0.01
    posterior_lr_scale: float = 1.0
    likelihood_lr_scale: float = 0.1
    joint_mf_warmstart_steps: int = 5000
    batch_size: int = 512
    num_mc_samples: int = 10
    pred_samples: int = 1024
    learn_inducing: bool = False
    use_warmup: bool = True
    h_inducing_init: str = "predmean"


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key.startswith("cuda:"):
        return torch.device(key)
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if key != "auto":
        return torch.device(name)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_index = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if int(free_bytes) > best_free:
            best_index = idx
            best_free = int(free_bytes)
    return torch.device(f"cuda:{best_index}")
