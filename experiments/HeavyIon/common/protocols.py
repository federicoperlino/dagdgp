"""Training budgets and frozen partitions for the four paper protocols."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.HeavyIon.common.data import build_real_data
from experiments.HeavyIon.common.real_config import DAGVIRunSpec, DAGSVIRunSpec
from experiments.HeavyIon.common.splits import make_fold_data, make_joint_fold_data

HERE = Path(__file__).resolve().parents[1]
PUBLISHED_PRETRAIN_STEPS = 500
PUBLISHED_JOINT_STEPS = 12000
PUBLISHED_WARMSTART_STEPS = 2000
PUBLISHED_TOTAL_UPDATES = 3 * PUBLISHED_PRETRAIN_STEPS + PUBLISHED_JOINT_STEPS
PUBLISHED_MODELS = [
    "mf_dgp_l1",
    "mf_dgp_l2",
    "dgp_x",
    "dgp",
    "mogp",
    "dag_vi",
    "dag_svi",
]


def load_data(protocol, *, split=0, fold=0):
    if not (HERE / "private/data/gmgp_real_dataset.npz").is_file():
        raise FileNotFoundError(
            "HeavyIon data are not distributed with the public code. "
            "Place the prepared dataset in HeavyIon/private/data/; see README.md."
        )
    full = build_real_data(
        data_dir=HERE / "private/data", include_test_h_in_train=protocol == "joint"
    )
    partitions = json.loads((HERE / "splits/partitions.json").read_text())
    if protocol == "published":
        return full
    if protocol == "repeated":
        indices = np.asarray(partitions["repeated"][str(split)][fold], dtype=np.int64)
        return make_fold_data(full, val_indices=indices, fold=fold)
    indices = {
        n: np.asarray(v[fold], dtype=np.int64) for n, v in partitions["joint"].items()
    }
    return make_joint_fold_data(full, val_indices_by_node=indices, fold=fold)


def dag_spec(protocol, model, *, seed=0, variant="predmean", smoke=False):
    low_steps, h_steps, h_count = {
        "published": (PUBLISHED_PRETRAIN_STEPS, PUBLISHED_PRETRAIN_STEPS, 25),
        "repeated": (2000, 0, 20),
        "joint": (3000, 1000, 90),
    }[protocol]
    common = dict(
        seed=seed,
        pretrain_steps_l1=low_steps,
        pretrain_steps_l2=low_steps,
        pretrain_steps_h=h_steps,
        num_inducing_h=h_count,
        h_inducing_init=variant,
        learn_inducing=model in {"dgp", "dgp_x"},
    )
    if model == "dag_svi":
        warmstart = PUBLISHED_WARMSTART_STEPS if protocol == "published" else 2000
        total_joint = PUBLISHED_JOINT_STEPS if protocol == "published" else 12000
        spec = DAGSVIRunSpec(
            **common,
            joint_steps=total_joint - warmstart,
            joint_mf_warmstart_steps=warmstart,
        )
    else:
        pretrained_nodes = {"dgp": 2, "mf_dgp_l1": 2, "mf_dgp_l2": 2, "dgp_x": 0}
        steps = PUBLISHED_JOINT_STEPS if protocol == "published" else 12000
        if protocol == "published" and model in pretrained_nodes:
            steps = (
                PUBLISHED_TOTAL_UPDATES
                - pretrained_nodes[model] * PUBLISHED_PRETRAIN_STEPS
            )
        spec = DAGVIRunSpec(**common, joint_steps=steps)
    if smoke:
        changes = dict(
            pretrain_steps_l1=2,
            pretrain_steps_l2=2,
            pretrain_steps_h=2 if h_steps else 0,
            joint_steps=2,
            num_mc_samples=2,
            pred_samples=8,
        )
        if model == "dag_svi":
            changes["joint_mf_warmstart_steps"] = 2
        spec = replace(spec, **changes)
    return spec
