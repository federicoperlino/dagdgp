#!/usr/bin/env python3
"""Render the appendix latent-collider figures from the saved plot data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.patches import Circle


OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_DATA_PATH = OUTPUT_DIR / "figure_seed2_plot_data.pt"
APPENDIX_DIR = OUTPUT_DIR / "appendix_res"
EXPLAIN_PDF = APPENDIX_DIR / "explaining_away_geometry.pdf"
COMPOSITION_PDF = APPENDIX_DIR / "compositional_uncertainty.pdf"

FINAL_COORDS = [
    (5, 9),
    (9, 1),
    (2, 1),
    (2, 6),
    (10, 7),
    (7, 6),
    (7, 3),
    (6, 1),
    (4, 7),
    (13, 1),
]

DAG_VI_COLOR = "#0072B2"
DAG_SVI_COLOR = "#E69F00"
DAG_VI_LABEL = "DAG-VI"
DAG_SVI_LABEL = "DAG-SVI"
EXPLAIN_BACKGROUND_LIGHT = "#edf1f5"
EXPLAIN_BACKGROUND_DARK = "#c3cbd4"
DISPLAY_POINTS = 450
DISK_RADIUS_FRAC = 0.078
POINT_SIZE = 6.2
ALPHA_UNDER = 0.0
ALPHA_MAIN = 0.48


class ExperimentConfig:
    pass


class TruthData:
    pass


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "contour.negative_linestyle": "solid",
    }
)


def load_plot_data(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def blend_with_white(color: str, amount: float) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    mixed = (1.0 - amount) * rgb + amount * np.ones(3, dtype=float)
    return mcolors.to_hex(mixed, keep_alpha=False)


def darken(color: str, amount: float) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    darker = (1.0 - amount) * rgb
    return mcolors.to_hex(np.clip(darker, 0.0, 1.0), keep_alpha=False)


def smooth_1d(values: np.ndarray, sigma: float = 2.2) -> np.ndarray:
    if sigma <= 0.0:
        return values.copy()
    radius = max(3, int(round(3.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def finite_diff(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    diff = np.empty_like(values)
    diff[1:-1] = (values[2:] - values[:-2]) / (grid[2:] - grid[:-2])
    diff[0] = (values[1] - values[0]) / (grid[1] - grid[0])
    diff[-1] = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
    return diff


def truth_jacobian_at_cases(truth: TruthData, coords: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    x1 = truth.x1_train.detach().cpu().numpy()
    x2 = truth.x2_train.detach().cpu().numpy()
    a = truth.a_train_unique.detach().cpu().numpy()
    b = truth.b_train_unique.detach().cpu().numpy()
    da = finite_diff(a, x1)
    db = finite_diff(b, x2)
    dA = np.array([da[i] for i, _ in coords], dtype=float)
    dB = np.array([db[j] for _, j in coords], dtype=float)
    return dA, dB


def safe_divisor(values: np.ndarray) -> np.ndarray:
    abs_values = np.abs(values)
    positive = abs_values[abs_values > 1e-8]
    floor = max(1e-3, 0.2 * np.median(positive) if positive.size else 1e-3)
    out = np.sign(values) * np.maximum(abs_values, floor)
    out[out == 0.0] = floor
    return out


def center_cloud(cloud: np.ndarray) -> np.ndarray:
    return cloud - cloud.mean(axis=0, keepdims=True)


def transformed_clouds(
    samples: dict[str, torch.Tensor],
    truth: TruthData,
    config: ExperimentConfig,
    coords: list[tuple[int, int]],
    dA_dx1: np.ndarray,
    dB_dx2: np.ndarray,
) -> list[np.ndarray]:
    A = samples["A"].squeeze(-1).detach().cpu().numpy()
    B = samples["B"].squeeze(-1).detach().cpu().numpy()
    dA_dx1 = safe_divisor(dA_dx1)
    dB_dx2 = safe_divisor(dB_dx2)

    out: list[np.ndarray] = []
    for k, (i, j) in enumerate(coords):
        flat = i * config.n2 + j
        a_draw = A[:, flat]
        b_draw = B[:, flat]
        dx1 = (a_draw - a_draw.mean()) / dA_dx1[k]
        dx2 = (b_draw - b_draw.mean()) / dB_dx2[k]
        out.append(center_cloud(np.stack([dx1, dx2], axis=1)))
    return out


def per_cloud_display_scales(all_clouds: list[np.ndarray], truth: TruthData, target_frac: float = 0.062) -> list[float]:
    span = min(
        float(truth.x1_train.max() - truth.x1_train.min()),
        float(truth.x2_train.max() - truth.x2_train.min()),
    )
    target = target_frac * span
    scales: list[float] = []
    for cloud in all_clouds:
        radii = np.sqrt((cloud**2).sum(axis=1))
        q90 = float(np.quantile(radii, 0.90))
        scales.append(1.0 if q90 < 1e-10 else target / q90)
    return scales


def contour_levels(values: np.ndarray, n_levels: int = 13) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    return np.linspace(lo, hi, n_levels)


def selected_case_corr(
    samples: dict[str, torch.Tensor],
    coords: list[tuple[int, int]],
    config: ExperimentConfig,
) -> list[float]:
    A = samples["A"].squeeze(-1).detach().cpu().numpy()
    B = samples["B"].squeeze(-1).detach().cpu().numpy()
    corrs: list[float] = []
    for i, j in coords:
        idx = i * config.n2 + j
        corrs.append(float(np.corrcoef(A[:, idx], B[:, idx])[0, 1]))
    return corrs


def sort_coords_for_display(truth: TruthData, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    x1 = truth.x1_train.detach().cpu().numpy()
    x2 = truth.x2_train.detach().cpu().numpy()
    return sorted(coords, key=lambda ij: (-float(x2[ij[1]]), float(x1[ij[0]])))


def draw_cloud_background(ax: plt.Axes, x0: float, y0: float, radius: float) -> None:
    ax.add_patch(Circle((x0, y0), radius=radius, facecolor="white", edgecolor="none", alpha=0.78, zorder=2.4))


def draw_cloud(
    ax: plt.Axes,
    pts: np.ndarray,
    *,
    color: str,
    point_size: float,
    alpha_under: float,
    alpha_main: float,
    style: str,
) -> None:
    if style == "ring":
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size * 1.55,
            facecolors="none",
            edgecolors=color,
            alpha=alpha_main,
            linewidths=0.88,
            marker="o",
            zorder=3.85,
        )
        return

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=point_size * 1.45,
        color="white",
        alpha=alpha_under,
        linewidths=0,
        marker="o",
        zorder=3.2,
    )
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=point_size,
        color=color,
        alpha=alpha_main,
        linewidths=0,
        marker="o",
        zorder=3.6,
    )


def render_explaining_away_geometry(plot_data: dict[str, object]) -> None:
    truth = plot_data["truth"]
    config = plot_data["config"]
    coords = sort_coords_for_display(truth, FINAL_COORDS)
    mf_corrs = selected_case_corr(plot_data["mf_samples_fig"], coords, config)
    structured_corrs = selected_case_corr(plot_data["structured_samples_fig"], coords, config)
    dA_dx1, dB_dx2 = truth_jacobian_at_cases(truth, coords)
    mf_clouds = transformed_clouds(plot_data["mf_samples_fig"], truth, config, coords, dA_dx1, dB_dx2)
    structured_clouds = transformed_clouds(plot_data["structured_samples_fig"], truth, config, coords, dA_dx1, dB_dx2)
    mf_scales = per_cloud_display_scales(mf_clouds, truth)
    structured_scales = per_cloud_display_scales(structured_clouds, truth)

    X1d = truth.x1_dense.detach().cpu().numpy()
    X2d = truth.x2_dense.detach().cpu().numpy()
    Csurf = truth.c_dense_truth_mean.detach().cpu().numpy()
    c_train = truth.c_train.reshape(config.n1, config.n2).detach().cpu().numpy()
    levels = contour_levels(Csurf, n_levels=13)

    fig, ax = plt.subplots(1, 1, figsize=(10.9, 8.15), constrained_layout=True)
    ax.contour(X1d, X2d, Csurf.T, levels=levels, colors=EXPLAIN_BACKGROUND_LIGHT, linewidths=0.88, linestyles="solid", alpha=0.90, zorder=1.0)
    ax.contour(X1d, X2d, Csurf.T, levels=levels[::2], colors=EXPLAIN_BACKGROUND_DARK, linewidths=1.08, linestyles="solid", alpha=0.84, zorder=1.1)

    x1_train = truth.x1_train.detach().cpu().numpy()
    x2_train = truth.x2_train.detach().cpu().numpy()
    span = min(float(X1d.max() - X1d.min()), float(X2d.max() - X2d.min()))
    disk_radius = DISK_RADIUS_FRAC * span
    n_available = mf_clouds[0].shape[0] if mf_clouds else 0
    keep_idx = None if DISPLAY_POINTS >= n_available else np.linspace(0, n_available - 1, DISPLAY_POINTS, dtype=int)

    for (i, j), mf_cloud, structured_cloud, mf_scale, structured_scale in zip(
        coords,
        mf_clouds,
        structured_clouds,
        mf_scales,
        structured_scales,
        strict=True,
    ):
        x0 = float(x1_train[i])
        y0 = float(x2_train[j])
        draw_cloud_background(ax, x0, y0, radius=disk_radius)
        ax.contour(X1d, X2d, Csurf.T, levels=[float(c_train[i, j])], colors=["#5B6168"], linewidths=1.85, linestyles=["solid"], alpha=0.95, zorder=2.2)

        st_pts = np.column_stack([x0 + structured_scale * structured_cloud[:, 0], y0 + structured_scale * structured_cloud[:, 1]])
        mf_pts = np.column_stack([x0 + mf_scale * mf_cloud[:, 0], y0 + mf_scale * mf_cloud[:, 1]])
        show_st = st_pts if keep_idx is None else st_pts[keep_idx]
        show_mf = mf_pts if keep_idx is None else mf_pts[keep_idx]

        draw_cloud(ax, show_mf, color=DAG_VI_COLOR, point_size=POINT_SIZE * 1.05, alpha_under=ALPHA_UNDER * 1.05, alpha_main=ALPHA_MAIN * 0.84, style="filled")
        draw_cloud(ax, show_st, color=DAG_SVI_COLOR, point_size=POINT_SIZE * 1.12, alpha_under=ALPHA_UNDER * 0.95, alpha_main=ALPHA_MAIN * 0.80, style="ring")
        ax.scatter(x0, y0, s=18, color="#111111", alpha=0.96, linewidths=0, zorder=4.55)

    handles = [
        plt.Line2D([0], [0], color=DAG_VI_COLOR, lw=0.0, marker="o", markersize=6.5, markerfacecolor=DAG_VI_COLOR, markeredgecolor="white", markeredgewidth=0.55, label=DAG_VI_LABEL),
        plt.Line2D([0], [0], color=DAG_SVI_COLOR, lw=0.0, marker="o", markersize=6.5, markerfacecolor="none", markeredgecolor=DAG_SVI_COLOR, markeredgewidth=1.0, label=DAG_SVI_LABEL),
        plt.Line2D([0], [0], color="#5B6168", lw=1.8, linestyle="-", label=r"True $w_3$ level set"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.53),
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        borderpad=0.5,
        labelspacing=0.45,
        handlelength=2.55,
        fontsize=13.0,
    )
    legend.get_frame().set_edgecolor("0.84")
    legend.get_frame().set_linewidth(0.9)

    corr_text = (
        r"avg $|\mathrm{posterior\ corr}(w_1,w_2)|$" + "\n"
        + f"{DAG_VI_LABEL}: {np.mean(np.abs(mf_corrs)):.2f}\n"
        + f"{DAG_SVI_LABEL}: {np.mean(np.abs(structured_corrs)):.2f}"
    )
    ax.text(
        0.03,
        0.97,
        corr_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "none", "alpha": 0.90},
        fontsize=12.0,
    )

    ax.set_title("Explaining Away in Input Space", fontsize=15.5, pad=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(float(X1d.min()), float(X1d.max()))
    ax.set_ylim(float(X2d.min()), float(X2d.max()))
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(EXPLAIN_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_method(ax: plt.Axes, x: np.ndarray, draws: np.ndarray, color: str, label: str) -> None:
    q025 = smooth_1d(np.quantile(draws, 0.025, axis=0))
    q975 = smooth_1d(np.quantile(draws, 0.975, axis=0))
    q25 = smooth_1d(np.quantile(draws, 0.25, axis=0))
    q75 = smooth_1d(np.quantile(draws, 0.75, axis=0))
    mean = smooth_1d(draws.mean(axis=0))
    idx = np.linspace(0, draws.shape[0] - 1, min(12, draws.shape[0]), dtype=int)
    band_outer = blend_with_white(color, 0.18)
    band_inner = blend_with_white(color, 0.08)
    line_color = darken(color, 0.12)

    ax.fill_between(x, q025, q975, color=band_outer, alpha=0.12, linewidth=0)
    ax.fill_between(x, q25, q75, color=band_inner, alpha=0.18, linewidth=0)
    for k in idx:
        ax.plot(x, draws[k], color=band_inner, alpha=0.035, linewidth=0.72)
    ax.plot(x, mean, color=line_color, linewidth=2.8, label=label)


def render_compositional_uncertainty(plot_data: dict[str, object]) -> None:
    x1_dense = to_numpy(plot_data["line_draws"]["x1_dense"])
    x2_dense = to_numpy(plot_data["line_draws"]["x2_dense"])
    A_mf = to_numpy(plot_data["line_draws"]["A_mf"])
    A_svi = to_numpy(plot_data["line_draws"]["A_structured"])
    B_mf = to_numpy(plot_data["line_draws"]["B_mf"])
    B_svi = to_numpy(plot_data["line_draws"]["B_structured"])

    fig, axes = plt.subplots(2, 1, figsize=(6.9, 8.15), constrained_layout=True)

    for ax, x, draws_vi, draws_svi, title, stats_text, ylabel, text_x, text_y, text_va in [
        (
            axes[0],
            x1_dense,
            A_mf,
            A_svi,
            r"Latent $w_1$",
            rf"{DAG_VI_LABEL}: posterior var$(w_1)$ = {plot_data['mf_metrics']['var_a']:.2e}" + "\n"
            + rf"{DAG_SVI_LABEL}: posterior var$(w_1)$ = {plot_data['structured_metrics']['var_a']:.2e}",
            r"$w_1$",
            0.03,
            0.92,
            "top",
        ),
        (
            axes[1],
            x2_dense,
            B_mf,
            B_svi,
            r"Latent $w_2$",
            rf"{DAG_VI_LABEL}: posterior var$(w_2)$ = {plot_data['mf_metrics']['var_b']:.2e}" + "\n"
            + rf"{DAG_SVI_LABEL}: posterior var$(w_2)$ = {plot_data['structured_metrics']['var_b']:.2e}",
            r"$w_2$",
            0.03,
            0.10,
            "bottom",
        ),
    ]:
        draw_method(ax, x, draws_vi, DAG_VI_COLOR, DAG_VI_LABEL)
        draw_method(ax, x, draws_svi, DAG_SVI_COLOR, DAG_SVI_LABEL)
        ax.axhline(0.0, color="0.88", linewidth=0.8, zorder=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.text(
            text_x,
            text_y,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va=text_va,
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "none", "alpha": 0.86},
            fontsize=13.0,
        )

    axes[0].set_xlabel(r"$x_1$")
    axes[1].set_xlabel(r"$x_2$")
    handles = [
        plt.Line2D([0], [0], color=DAG_VI_COLOR, lw=2.6, label=DAG_VI_LABEL),
        plt.Line2D([0], [0], color=DAG_SVI_COLOR, lw=2.6, label=DAG_SVI_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.31, -0.005), fontsize=12.6)
    fig.savefig(COMPOSITION_PDF, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = load_plot_data(PLOT_DATA_PATH)
    render_explaining_away_geometry(plot_data)
    render_compositional_uncertainty(plot_data)
    print(EXPLAIN_PDF)
    print(COMPOSITION_PDF)


if __name__ == "__main__":
    main()
