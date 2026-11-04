"""Configurations for the three Sachs experiments."""

from __future__ import annotations

from dataclasses import dataclass

ROOTS = ("PKC", "Plcg")
NODES = ("Raf", "Mek", "PIP2", "PIP3", "Erk", "Akt", "PKA", "P38", "Jnk")
OBSERVED = {"PKA", "Raf"}
PARENTS = {
    "PKC": (),
    "PKA": ("PKC",),
    "Raf": ("PKC", "PKA"),
    "Mek": ("PKC", "PKA", "Raf"),
    "Erk": ("PKA", "Mek"),
    "Akt": ("PKA", "Erk"),
    "P38": ("PKC", "PKA"),
    "Jnk": ("PKC", "PKA"),
    "Plcg": (),
    "PIP3": ("Plcg",),
    "PIP2": ("Plcg", "PIP3"),
}
MODELS = ("dag_vi", "dag_svi", "mogp")
SEEDS = tuple(range(5))
MODEL_SEED = 0


@dataclass(frozen=True)
class Config:
    protocol: str
    num_inducing: int
    joint_steps: int
    mogp_steps: int
    pretrain_steps: int
    pretrain_batch_size: int | None
    batch_size: int | None
    unfreeze_step: int
    std_ddof: int
    num_mc_samples: int = 5
    predictive_samples: int = 128
    joint_lr: float = 1e-3
    mogp_lr: float = 1e-2


CONFIGS = {
    "interpolation": Config(
        protocol="interpolation",
        num_inducing=32,
        joint_steps=10_000,
        mogp_steps=19_000,
        pretrain_steps=1_000,
        pretrain_batch_size=None,
        batch_size=None,
        unfreeze_step=4_000,
        std_ddof=0,
    ),
    "scalability": Config(
        protocol="scalability",
        num_inducing=64,
        joint_steps=30_000,
        mogp_steps=30_000,
        pretrain_steps=1_000,
        pretrain_batch_size=512,
        batch_size=1_024,
        unfreeze_step=1_000,
        std_ddof=1,
    ),
    "extrapolation": Config(
        protocol="extrapolation",
        num_inducing=32,
        joint_steps=10_000,
        mogp_steps=10_000,
        pretrain_steps=0,
        pretrain_batch_size=None,
        batch_size=None,
        unfreeze_step=4_000,
        std_ddof=0,
    ),
}
