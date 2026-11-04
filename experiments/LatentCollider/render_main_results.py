#!/usr/bin/env python3
"""Render the main-text latent-collider figures from the saved plot data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import chi2, norm


OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_DATA_PATH = OUTPUT_DIR / "figure_seed2_plot_data.pt"
MAIN_DIR = OUTPUT_DIR / "main_res"
TRUE_CHILD_PDF = MAIN_DIR / "true_child.pdf"
JOINT_SAMPLE_PDF = MAIN_DIR / "joint_sample.pdf"
ABSOLUTE_ERROR_PDF = MAIN_DIR / "absolute_error.pdf"
HERO_CASE = (5, 6)

TEXT_COLOR = "#28313B"
MUTED_TEXT = "#607080"
AXIS_COLOR = "#D1D8E0"
CONTOUR_LINE = "#D6D6D6"
MATCH_LINE = "#2B2B2B"
POINT_COLOR = "#111111"


class ExperimentConfig:
    pass


class TruthData:
    pass


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9.2,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.usetex": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_plot_data(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=MUTED_TEXT, labelsize=7.6, length=2.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)


def soft_child_cmap():
    return plt.get_cmap("coolwarm")


def fit_gaussian(samples_a: np.ndarray, samples_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array([samples_a.mean(), samples_b.mean()]), np.cov(samples_a, samples_b)


def ellipse(mu: np.ndarray, cov: np.ndarray, prob: float, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    k = chi2.ppf(prob, df=2)
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-14)
    a, b = np.sqrt(w[1] * k), np.sqrt(w[0] * k)
    v1, v2 = V[:, 1], V[:, 0]
    t = np.linspace(0.0, 2.0 * np.pi, n)
    xy = mu[:, None] + a * np.cos(t) * v1[:, None] + b * np.sin(t) * v2[:, None]
    return xy[0], xy[1]


def render_true_child(plot_data: dict[str, object]) -> None:
    config = plot_data["config"]
    truth = plot_data["truth"]
    n1 = int(config.n1)
    n2 = int(config.n2)
    x1 = truth.x1_train.detach().cpu().numpy()
    x2 = truth.x2_train.detach().cpu().numpy()
    c_grid = truth.c_train.reshape(n1, n2).detach().cpu().numpy()

    i, j = HERO_CASE
    x1_star = float(x1[i])
    x2_star = float(x2[j])
    c_star = float(c_grid[i, j])

    fig = plt.figure(figsize=(6.3, 5.5))
    ax = fig.add_axes([0.10, 0.11, 0.74, 0.80])
    levels = np.linspace(float(c_grid.min()), float(c_grid.max()), 14)
    mappable = ax.contourf(x1, x2, c_grid.T, levels=levels, cmap=soft_child_cmap(), antialiased=True, zorder=0)
    ax.contour(x1, x2, c_grid.T, levels=levels, colors=CONTOUR_LINE, linewidths=1.0, zorder=1)
    ax.contour(x1, x2, c_grid.T, levels=[c_star], colors=MATCH_LINE, linewidths=2.8, zorder=2)

    ax.scatter([x1_star], [x2_star], s=260, color="white", linewidths=0, zorder=3)
    ax.scatter([x1_star], [x2_star], s=88, color=POINT_COLOR, linewidths=0, zorder=4)
    ax.text(
        x1_star + 0.10,
        x2_star + 0.07,
        r"$x^\ast$",
        fontsize=15.5,
        color=TEXT_COLOR,
        weight="bold",
        zorder=5,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.86},
    )

    ax.set_xlabel(r"$x_1$", fontsize=13.0, color=TEXT_COLOR)
    ax.set_ylabel(r"$x_2$", fontsize=13.0, color=TEXT_COLOR)
    ax.set_xlim(float(x1.min()), float(x1.max()))
    ax.set_ylim(float(x2.min()), float(x2.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.tick_params(colors=MUTED_TEXT, labelsize=12.2, length=3.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    cax = fig.add_axes([0.86, 0.11, 0.030, 0.80])
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical", format="%g")
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5])
    cbar.set_label(r"$w_3$ value", labelpad=4, fontsize=12.0, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=10.2, colors=MUTED_TEXT, length=2.5)
    cbar.outline.set_edgecolor(AXIS_COLOR)

    fig.savefig(TRUE_CHILD_PDF, dpi=280, bbox_inches="tight")
    plt.close(fig)


def render_joint_sample(plot_data: dict[str, object]) -> None:
    config = plot_data["config"]
    idx = int(HERO_CASE[0]) * int(config.n2) + int(HERO_CASE[1])

    vi_a = plot_data["mf_samples_fig"]["A"].squeeze(-1)[:, idx].detach().cpu().numpy()
    vi_b = plot_data["mf_samples_fig"]["B"].squeeze(-1)[:, idx].detach().cpu().numpy()
    svi_a = plot_data["structured_samples_fig"]["A"].squeeze(-1)[:, idx].detach().cpu().numpy()
    svi_b = plot_data["structured_samples_fig"]["B"].squeeze(-1)[:, idx].detach().cpu().numpy()

    mu_pr = np.array([0.0, 0.0])
    cov_pr = np.eye(2)
    mu_vi, cov_vi = fit_gaussian(vi_a, vi_b)
    mu_svi, cov_svi = fit_gaussian(svi_a, svi_b)

    sig_pr1 = float(np.sqrt(cov_pr[0, 0]))
    sig_vi1 = float(np.sqrt(cov_vi[0, 0]))
    sig_svi1 = float(np.sqrt(cov_svi[0, 0]))
    sig_pr2 = float(np.sqrt(cov_pr[1, 1]))
    sig_vi2 = float(np.sqrt(cov_vi[1, 1]))
    sig_svi2 = float(np.sqrt(cov_svi[1, 1]))

    c_pr = "#BBBBBB"
    c_vi = "#4477AA"
    c_svi = "#CC6677"
    halo = 4.5
    probs = [0.90, 0.68, 0.50]
    lw_pr = [0.70, 1.00, 1.45]
    lw_post = [1.10, 1.60, 2.20]
    xlim = (-3.0, 2.5)
    ylim = (-2.8, 3.6)

    fig = plt.figure(figsize=(4.5, 4.2))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[3.8, 1.1],
        height_ratios=[1.1, 3.8],
        hspace=0.04,
        wspace=0.04,
        left=0.11,
        right=0.97,
        bottom=0.10,
        top=0.97,
    )

    ax_j = fig.add_subplot(gs[1, 0])
    ax_tx = fig.add_subplot(gs[0, 0], sharex=ax_j)
    ax_ry = fig.add_subplot(gs[1, 1], sharey=ax_j)
    ax_lg = fig.add_subplot(gs[0, 1])
    ax_lg.axis("off")

    ax_j.set_xlim(*xlim)
    ax_j.set_ylim(*ylim)
    ax_j.set_aspect("equal")
    ax_j.tick_params(direction="out")
    for v in np.arange(-2.5, 2.5, 0.5):
        ax_j.axvline(v, color="#f0f0f0", lw=0.4, zorder=0)
    for v in np.arange(-2.5, 3.5, 0.5):
        ax_j.axhline(v, color="#f0f0f0", lw=0.4, zorder=0)
    ax_j.axhline(0.0, color="#e0e0e0", lw=0.6, zorder=0)
    ax_j.axvline(0.0, color="#e0e0e0", lw=0.6, zorder=0)
    ax_j.spines["top"].set_visible(False)
    ax_j.spines["right"].set_visible(False)

    x99, y99 = ellipse(mu_pr, cov_pr, 0.99)
    ax_j.fill(x99, y99, color=c_pr, alpha=0.07, zorder=1)
    for prob, lw in zip(probs, lw_pr, strict=True):
        x, y = ellipse(mu_pr, cov_pr, prob)
        ax_j.plot(x, y, color=c_pr, lw=lw, ls=(0, (5, 3.5)), alpha=0.65, zorder=2)

    xf, yf = ellipse(mu_svi, cov_svi, 0.50)
    ax_j.fill(xf, yf, color=c_svi, alpha=0.13, zorder=3)
    for prob, lw in zip(probs, lw_post, strict=True):
        x, y = ellipse(mu_svi, cov_svi, prob)
        ax_j.plot(x, y, color="white", lw=lw + halo, zorder=4, solid_capstyle="round")
        ax_j.plot(x, y, color=c_svi, lw=lw, zorder=5, solid_capstyle="round")

    ax_j.plot(*mu_vi, "o", ms=9, color=c_vi, zorder=10, markeredgecolor="white", markeredgewidth=1.0)
    ax_j.plot(*mu_pr, "o", ms=5, color=c_pr, zorder=11, markeredgecolor="white", markeredgewidth=0.6)
    ax_j.plot(*mu_svi, "s", ms=4.5, color=c_svi, zorder=11, markeredgecolor="white", markeredgewidth=0.6)
    ax_j.annotate(
        "",
        xy=mu_svi + np.array([0.01, 0.07]),
        xytext=mu_pr,
        arrowprops=dict(
            arrowstyle="->",
            lw=1.1,
            color="#666",
            linestyle=(0, (4, 2.5)),
            connectionstyle="arc3,rad=-0.30",
        ),
        zorder=9,
    )
    ax_j.set_xlabel(r"$F_{w_1}(x_1^*)$", labelpad=3)
    ax_j.set_ylabel(r"$F_{w_2}(x_2^*)$", rotation=0, labelpad=16)

    grid1 = np.linspace(xlim[0], xlim[1], 800)
    pdf_pr1 = norm.pdf(grid1, mu_pr[0], sig_pr1)
    pdf_svi1 = norm.pdf(grid1, mu_svi[0], sig_svi1)
    pdf_vi1 = norm.pdf(grid1, mu_vi[0], sig_vi1)
    ylim_tx = float(norm.pdf(mu_svi[0], mu_svi[0], sig_svi1) * 1.65)

    ax_tx.fill_between(grid1, pdf_pr1, alpha=0.07, color=c_pr)
    ax_tx.plot(grid1, pdf_pr1, color=c_pr, lw=0.9, ls=(0, (5, 3.5)), alpha=0.75)
    ax_tx.fill_between(grid1, np.clip(pdf_svi1, 0.0, ylim_tx), alpha=0.14, color=c_svi)
    ax_tx.plot(grid1, np.clip(pdf_svi1, 0.0, ylim_tx), color=c_svi, lw=1.5)
    ax_tx.fill_between(grid1, np.clip(pdf_vi1, 0.0, ylim_tx), alpha=0.14, color=c_vi)
    ax_tx.plot(grid1, np.clip(pdf_vi1, 0.0, ylim_tx), color=c_vi, lw=1.5)

    vi_peak1 = float(norm.pdf(mu_vi[0], mu_vi[0], sig_vi1))
    ratio1 = vi_peak1 / float(norm.pdf(mu_pr[0], mu_pr[0], sig_pr1))
    bw = 0.045
    bd = 0.018
    y_brk = ylim_tx * 0.985
    for dy in [0.0, -0.055 * ylim_tx]:
        y = y_brk + dy
        ax_tx.plot([mu_vi[0] - bw, mu_vi[0] + bw], [y - bd, y + bd], color="white", lw=3.5, solid_capstyle="round", zorder=9)
        ax_tx.plot([mu_vi[0] - bw, mu_vi[0] + bw], [y - bd, y + bd], color=c_vi, lw=1.3, solid_capstyle="round", zorder=10)
    ax_tx.text(
        mu_vi[0] + bw + 0.04,
        y_brk - 0.03 * ylim_tx,
        f"×{ratio1:.0f}",
        ha="left",
        va="center",
        fontsize=6,
        color=c_vi,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
    )
    ax_tx.set_ylim(0.0, ylim_tx)
    ax_tx.set_xlim(*xlim)
    ax_tx.axis("off")

    grid2 = np.linspace(ylim[0], ylim[1], 800)
    pdf_pr2 = norm.pdf(grid2, mu_pr[1], sig_pr2)
    pdf_svi2 = norm.pdf(grid2, mu_svi[1], sig_svi2)
    pdf_vi2 = norm.pdf(grid2, mu_vi[1], sig_vi2)
    ylim_ry = float(norm.pdf(mu_svi[1], mu_svi[1], sig_svi2) * 1.65)

    ax_ry.fill_betweenx(grid2, np.clip(pdf_pr2, 0.0, ylim_ry), alpha=0.07, color=c_pr)
    ax_ry.plot(np.clip(pdf_pr2, 0.0, ylim_ry), grid2, color=c_pr, lw=0.9, ls=(0, (5, 3.5)), alpha=0.75)
    ax_ry.fill_betweenx(grid2, np.clip(pdf_svi2, 0.0, ylim_ry), alpha=0.14, color=c_svi)
    ax_ry.plot(np.clip(pdf_svi2, 0.0, ylim_ry), grid2, color=c_svi, lw=1.5)
    ax_ry.fill_betweenx(grid2, np.clip(pdf_vi2, 0.0, ylim_ry), alpha=0.14, color=c_vi)
    ax_ry.plot(np.clip(pdf_vi2, 0.0, ylim_ry), grid2, color=c_vi, lw=1.5)

    vi_peak2 = float(norm.pdf(mu_vi[1], mu_vi[1], sig_vi2))
    ratio2 = vi_peak2 / float(norm.pdf(mu_pr[1], mu_pr[1], sig_pr2))
    bh = 0.045
    x_brk = ylim_ry * 0.985
    for dx in [0.0, -0.055 * ylim_ry]:
        x = x_brk + dx
        ax_ry.plot([x - bd, x + bd], [mu_vi[1] - bh, mu_vi[1] + bh], color="white", lw=3.5, solid_capstyle="round", zorder=9)
        ax_ry.plot([x - bd, x + bd], [mu_vi[1] - bh, mu_vi[1] + bh], color=c_vi, lw=1.3, solid_capstyle="round", zorder=10)
    ax_ry.text(
        x_brk - 0.03 * ylim_ry,
        mu_vi[1] - bh - 0.06,
        f"×{ratio2:.0f}",
        ha="center",
        va="top",
        fontsize=6,
        color=c_vi,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
    )
    ax_ry.set_xlim(0.0, ylim_ry * 1.12)
    ax_ry.set_ylim(*ylim)
    ax_ry.axis("off")

    handles = [
        Line2D([0], [0], color=c_pr, lw=1.2, ls=(0, (5, 3.5)), alpha=0.70, label="Prior"),
        Line2D([0], [0], color=c_vi, lw=0, marker="o", ms=6.5, markeredgecolor="white", markeredgewidth=0.5, label="VI posterior"),
        Line2D([0], [0], color=c_svi, lw=2.0, label="SVI posterior"),
    ]
    ax_lg.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.42, 0.45),
        frameon=True,
        framealpha=0.95,
        edgecolor="#d5d5d5",
        handlelength=1.5,
        borderpad=0.65,
        labelspacing=0.55,
    )

    fig.savefig(JOINT_SAMPLE_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_absolute_error(plot_data: dict[str, object]) -> None:
    config = plot_data["config"]
    truth = plot_data["truth"]
    x1 = truth.x1_train.detach().cpu().numpy()
    x2 = truth.x2_train.detach().cpu().numpy()
    c_true = truth.c_train.reshape(config.n1, config.n2).detach().cpu().numpy()
    c_vi = plot_data["mf_samples_eval"]["C"].squeeze(-1).mean(dim=0).reshape(config.n1, config.n2).detach().cpu().numpy()
    c_svi = plot_data["structured_samples_eval"]["C"].squeeze(-1).mean(dim=0).reshape(config.n1, config.n2).detach().cpu().numpy()

    err_vi = np.abs(c_vi - c_true)
    err_svi = np.abs(c_svi - c_true)
    err_max = float(max(err_vi.max(), err_svi.max()))
    err_levels = np.linspace(0.0, err_max, 15)

    fig = plt.figure(figsize=(5.55, 3.15))
    ax_vi = fig.add_axes([0.08, 0.12, 0.34, 0.52])
    ax_svi = fig.add_axes([0.49, 0.12, 0.34, 0.52])

    m_vi = ax_vi.contourf(x1, x2, err_vi.T, levels=err_levels, cmap="magma", extend="max")
    ax_vi.contour(x1, x2, err_vi.T, levels=err_levels[1::2], colors="#f2e9e4", linewidths=0.52, alpha=0.35)
    ax_vi.set_title("DAG-VI", color=TEXT_COLOR, fontsize=11.3, pad=4)
    ax_vi.set_ylabel(r"$x_2$")
    ax_vi.set_xlabel(r"$x_1$")
    style_axis(ax_vi)

    ax_svi.contourf(x1, x2, err_svi.T, levels=err_levels, cmap="magma", extend="max")
    ax_svi.contour(x1, x2, err_svi.T, levels=err_levels[1::2], colors="#f2e9e4", linewidths=0.52, alpha=0.35)
    ax_svi.set_title("DAG-SVI", color=TEXT_COLOR, fontsize=11.3, pad=4)
    ax_svi.set_ylabel(r"$x_2$")
    ax_svi.set_xlabel(r"$x_1$")
    style_axis(ax_svi)

    for ax in (ax_vi, ax_svi):
        ax.set_xlim(float(x1.min()), float(x1.max()))
        ax.set_ylim(float(x2.min()), float(x2.max()))
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_yticks([-1.0, 0.0, 1.0])

    cax_err = fig.add_axes([0.22, 0.81, 0.46, 0.045])
    cbar_err = fig.colorbar(m_vi, cax=cax_err, orientation="horizontal", format="%g")
    cbar_err.set_ticks([0.0, 0.1, 0.2, 0.3])
    cbar_err.set_label("Absolute error", color=TEXT_COLOR, labelpad=3)
    cbar_err.ax.tick_params(labelsize=7, colors=MUTED_TEXT, length=2.0)
    cbar_err.outline.set_edgecolor(AXIS_COLOR)

    fig.savefig(ABSOLUTE_ERROR_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = load_plot_data(PLOT_DATA_PATH)
    render_true_child(plot_data)
    render_joint_sample(plot_data)
    render_absolute_error(plot_data)
    print(TRUE_CHILD_PDF)
    print(JOINT_SAMPLE_PDF)
    print(ABSOLUTE_ERROR_PDF)


if __name__ == "__main__":
    main()
