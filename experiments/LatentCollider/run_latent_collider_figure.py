#!/usr/bin/env python3
"""Run the latent collider figure experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import run_latent_collider as base


OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "figure_summary.json"
PLOT_DATA_PATH = OUTPUT_DIR / "figure_seed2_plot_data.pt"
TRAIN_SEED = 2


def _config() -> base.ExperimentConfig:
    return base.ExperimentConfig(
        steps=base.STEPS,
        figure_seed=TRAIN_SEED,
        fit_seed=TRAIN_SEED,
        child_lengthscale=base.CHILD_LENGTHSCALE,
        noise_std=base.NOISE_STD,
        m_root=base.M_ROOT,
        child_grid_m1=base.CHILD_GRID,
        child_grid_m2=base.CHILD_GRID,
        num_mc_samples=base.NUM_MC_SAMPLES,
        posterior_samples=base.POSTERIOR_SAMPLES_FIG,
        child_kernel_mode="sum",
        child_input_connection=False,
    )


def main() -> None:
    if base.DEVICE.type != "cpu":
        raise RuntimeError("This script requires DAGDGP_DEVICE=cpu.")

    config = _config()
    kernel_ctor = base.kernel_ctor_from_config(config)
    dataset, truth = base.generate_dataset(config, seed=TRAIN_SEED, kernel_ctor=kernel_ctor)
    mf_model, structured_model = base.fit_both(config, dataset, kernel_ctor=kernel_ctor)

    torch.manual_seed(TRAIN_SEED)
    mf_metrics = base.global_metrics(
        mf_model,
        dataset,
        truth,
        num_samples=base.POSTERIOR_SAMPLES_EVAL,
        noise_std=config.noise_std,
    )
    torch.manual_seed(TRAIN_SEED)
    mf_samples_eval = base.sample_all_latents(
        mf_model,
        dataset,
        num_samples=base.POSTERIOR_SAMPLES_EVAL,
    )

    torch.manual_seed(TRAIN_SEED)
    structured_metrics = base.global_metrics(
        structured_model,
        dataset,
        truth,
        num_samples=base.POSTERIOR_SAMPLES_EVAL,
        noise_std=config.noise_std,
    )
    torch.manual_seed(TRAIN_SEED)
    structured_samples_eval = base.sample_all_latents(
        structured_model,
        dataset,
        num_samples=base.POSTERIOR_SAMPLES_EVAL,
    )

    torch.manual_seed(TRAIN_SEED + 1)
    mf_samples_fig = base.sample_all_latents(
        mf_model,
        dataset,
        num_samples=base.POSTERIOR_SAMPLES_FIG,
    )
    torch.manual_seed(TRAIN_SEED + 1)
    structured_samples_fig = base.sample_all_latents(
        structured_model,
        dataset,
        num_samples=base.POSTERIOR_SAMPLES_FIG,
    )

    torch.manual_seed(TRAIN_SEED + 2)
    x1_dense, A_mf = base.line_samples_for_A(mf_model, config)
    torch.manual_seed(TRAIN_SEED + 2)
    _, A_structured = base.line_samples_for_A(structured_model, config)
    torch.manual_seed(TRAIN_SEED + 3)
    x2_dense, B_mf = base.line_samples_for_B(mf_model, config)
    torch.manual_seed(TRAIN_SEED + 3)
    _, B_structured = base.line_samples_for_B(structured_model, config)

    coords, _ = base.select_anchor_cases_from_observed_c(truth, config)
    plot_data = {
        "config": config,
        "truth": truth,
        "mf_metrics": mf_metrics,
        "structured_metrics": structured_metrics,
        "mf_samples_eval": mf_samples_eval,
        "structured_samples_eval": structured_samples_eval,
        "mf_samples_fig": mf_samples_fig,
        "structured_samples_fig": structured_samples_fig,
        "line_draws": {
            "x1_dense": x1_dense,
            "x2_dense": x2_dense,
            "A_mf": A_mf,
            "A_structured": A_structured,
            "B_mf": B_mf,
            "B_structured": B_structured,
        },
    }
    summary = {
        "config": asdict(config),
        "train_seed": TRAIN_SEED,
        "mf": mf_metrics,
        "structured": structured_metrics,
        "figure_coords": [list(item) for item in coords],
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(plot_data, PLOT_DATA_PATH)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {PLOT_DATA_PATH}")
    print(
        f"DAG-VI RMSE={mf_metrics['rmse_c']:.3f}, NLL={mf_metrics['nll_c']:.3f}; "
        f"DAG-SVI RMSE={structured_metrics['rmse_c']:.3f}, "
        f"NLL={structured_metrics['nll_c']:.3f}"
    )


if __name__ == "__main__":
    main()
