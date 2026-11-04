"""Command-line options shared by the experiment entry points."""

import argparse
from pathlib import Path

import torch

from experiments.HeavyIon.common.protocols import HERE
from experiments.HeavyIon.common.published import PUBLISHED_SEEDS
from experiments.HeavyIon.common.real_config import resolve_device


def arguments(protocol, models):
    parser = argparse.ArgumentParser(description=f"HeavyIon {protocol} experiment")
    parser.add_argument("--models", nargs="+", choices=models, default=models)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two updates per phase; writes to smoke_results.",
    )
    if protocol == "published":
        parser.add_argument(
            "--seeds",
            nargs="+",
            type=int,
            choices=PUBLISHED_SEEDS,
            default=list(PUBLISHED_SEEDS),
        )
    else:
        parser.add_argument(
            "--folds",
            nargs="+",
            type=int,
            choices=range(5 if protocol == "repeated" else 10),
        )
    if protocol == "repeated":
        parser.add_argument(
            "--splits", nargs="+", type=int, choices=range(10), default=list(range(10))
        )
    if protocol == "joint":
        parser.add_argument(
            "--variants",
            nargs="+",
            choices=("predmean", "random"),
            default=["predmean", "random"],
        )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.device = resolve_device(args.device)
    args.output = (
        args.output or HERE / ("smoke_results" if args.smoke else "results") / protocol
    )
    if args.smoke:
        for name in ("seeds", "splits"):
            if hasattr(args, name):
                setattr(args, name, getattr(args, name)[:1])
    if hasattr(args, "folds") and args.folds is None:
        args.folds = (
            [0] if args.smoke else list(range(5 if protocol == "repeated" else 10))
        )
    return args
