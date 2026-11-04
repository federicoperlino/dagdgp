#!/usr/bin/env python3
"""Run the Sachs interpolation, scalability, and extrapolation experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import gpytorch
import numpy as np
import torch

from dagdgp import FitConfig, fit
from config import CONFIGS, MODEL_SEED, MODELS, NODES, OBSERVED, ROOTS, SEEDS, Config
from data import DIRECTORY, SachsData, load_data, make_dataset
from evaluation import dag_predictions, evaluate, exact_mogp_row_samples
from models import (
    build_dag_model,
    build_mogp,
    initialise_dag_svi_from_dag_vi,
    nodewise_pretrain,
    set_seed,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_dag(model, dataset, cfg: Config) -> float:
    options = FitConfig(
        steps=cfg.joint_steps,
        joint_lr=cfg.joint_lr,
        posterior_lr_scale=1.0,
        inducing_unfreeze_step=cfg.unfreeze_step,
        likelihood_unfreeze_step=cfg.unfreeze_step,
        batch_size=dataset.num_data if cfg.batch_size is None else cfg.batch_size,
        num_mc_samples=cfg.num_mc_samples,
        likelihood_lr_scale=1.0,
        optimizer="adam",
        lr_schedule="none",
        seed=MODEL_SEED,
    )
    device = next(model.parameters()).device
    synchronize(device)
    started = time.perf_counter()
    fit(model, dataset, options)
    synchronize(device)
    return time.perf_counter() - started


def run_mogp(data: SachsData, cfg: Config, seed: int, device: torch.device):
    set_seed(MODEL_SEED)
    x = torch.as_tensor(data.inputs.loc[:, ROOTS].to_numpy(np.float64).copy(), device=device)
    values = data.inputs.loc[:, NODES].to_numpy(np.float64).copy()
    values[np.ix_(data.test, np.asarray([n not in OBSERVED for n in NODES]))] = np.nan
    y = torch.as_tensor(values, device=device)
    test_x = x[torch.as_tensor(data.test, device=device)]
    model, likelihood = build_mogp(x, data.train, cfg)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([*model.parameters(), *likelihood.parameters()], lr=cfg.mogp_lr)
    objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x))
    synchronize(device)
    started = time.perf_counter()
    with gpytorch.settings.cholesky_jitter(1e-5), gpytorch.settings.observation_nan_policy("mask"):
        for _ in range(cfg.mogp_steps):
            if cfg.protocol == "scalability":
                batch = torch.randperm(len(x), device=device)
                batch_x, batch_y = x[batch], y[batch]
            else:
                batch_x, batch_y = x, y
            optimizer.zero_grad(set_to_none=True)
            loss = -objective(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MO-GP loss.")
            loss.backward()
            optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    samples = exact_mogp_row_samples(
        model, likelihood, test_x, num_samples=cfg.predictive_samples,
        seed=271_828 + 1_000 * seed,
    )
    if data.offsets is not None:
        samples = samples + data.offsets[data.test][None, :, :]
    return evaluate(data, samples), elapsed


def run_split(
    cfg: Config,
    seed: int,
    models: list[str],
    output: Path,
    device: torch.device,
    data_dir: Path | None,
) -> None:
    data = load_data(cfg.protocol, seed, data_dir)
    pending = []
    for family in models:
        path = output / cfg.protocol / f"seed_{seed}" / f"{family}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["config"] != asdict(cfg) or saved["split_sha256"] != data.split_hash:
                raise ValueError(f"Existing result has a different configuration: {path}")
            if (saved["protocol"], saved["seed"], saved["model"]) != (
                cfg.protocol, seed, family
            ):
                raise ValueError(f"Existing result has inconsistent identifiers: {path}")
            if not all(
                np.isfinite(saved["metrics"][key])
                for key in ("rmse", "crps", "energy_standardised")
            ):
                raise ValueError(f"Existing result has invalid scores: {path}")
            if cfg.protocol != "extrapolation":
                elapsed = saved.get("training_seconds", float("nan"))
                if not np.isfinite(elapsed) or elapsed < 0:
                    raise ValueError(f"Existing result has an invalid training time: {path}")
            print(f"Complete: {cfg.protocol}, seed {seed}, {family}", flush=True)
        else:
            pending.append(family)
    if not pending:
        return
    dataset = make_dataset(data, device)
    shared_state = None
    if cfg.pretrain_steps and any(m != "mogp" for m in pending):
        model = build_dag_model("dag_vi", dataset, data.train, cfg, device)
        nodewise_pretrain(model, dataset, cfg)
        shared_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model
    for family in pending:
        if family == "mogp":
            metrics, elapsed = run_mogp(data, cfg, seed, device)
        else:
            model = build_dag_model(family, dataset, data.train, cfg, device)
            if shared_state is not None:
                if family == "dag_vi":
                    model.load_state_dict(shared_state, strict=True)
                else:
                    initialise_dag_svi_from_dag_vi(model, shared_state)
            elapsed = train_dag(model, dataset, cfg)
            metrics = evaluate(data, dag_predictions(model, dataset, data, cfg, seed))
            del model
        result = {
            "protocol": cfg.protocol, "seed": seed, "model": family,
            "config": asdict(cfg), "split_sha256": data.split_hash,
            "metrics": metrics,
            "hardware": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "software": {
                "torch": torch.__version__,
                "gpytorch": gpytorch.__version__,
                "numpy": np.__version__,
            },
        }
        if cfg.protocol != "extrapolation":
            result["training_seconds"] = elapsed
        path = output / cfg.protocol / f"seed_{seed}" / f"{family}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
        print(f"Saved: {path}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY / "reproduction")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use two training steps and eight predictive samples to check execution.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    protocols = list(CONFIGS) if args.protocol == "all" else [args.protocol]
    output = args.output_dir / "smoke" if args.smoke else args.output_dir
    for protocol in protocols:
        cfg = CONFIGS[protocol]
        if args.smoke:
            cfg = replace(
                cfg,
                joint_steps=2,
                mogp_steps=2,
                pretrain_steps=min(2, cfg.pretrain_steps),
                unfreeze_step=1,
                predictive_samples=8,
            )
        for seed in dict.fromkeys(args.seeds):
            run_split(cfg, seed, list(dict.fromkeys(args.models)), output, device, args.data_dir)


if __name__ == "__main__":
    main()
