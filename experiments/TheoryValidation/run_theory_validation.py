#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "theory_experiments.pdf"

PAPER = {
    "navy": "#223B5E",
    "teal": "#218F95",
    "terracotta": "#D96B51",
    "purple": "#6C5B7B",
    "grid": "#CFCBC3",
}

ArrayLike = Union[np.ndarray, float]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
            "grid.color": PAPER["grid"],
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.7)


def se_corr_from_sqdist(dist2: ArrayLike, lengthscale: float) -> np.ndarray:
    dist2 = np.asarray(dist2, dtype=float)
    return np.exp(-0.5 * dist2 / (lengthscale**2))


@dataclass(frozen=True)
class KernelFamily:
    name: str = "se"
    lengthscale: float = 1.0

    def corr(self, sqdist: ArrayLike) -> np.ndarray:
        if self.name != "se":
            raise ValueError(f"Unsupported kernel family {self.name!r}.")
        return se_corr_from_sqdist(sqdist, self.lengthscale)


def sample_gaussian_contrasts(variance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    variance = np.maximum(np.asarray(variance, dtype=float), 0.0)
    return rng.normal(loc=0.0, scale=np.sqrt(variance), size=variance.shape)


def circulant_parents(width: int, indegree: int) -> np.ndarray:
    if indegree > width:
        raise ValueError("indegree cannot exceed width.")
    offsets = np.arange(indegree, dtype=int)
    parents = np.empty((width, indegree), dtype=int)
    for j in range(width):
        parents[j] = (j + offsets) % width
    return parents


@dataclass
class IndegreeExperimentConfig:
    depth: int = 50
    width: int = 128
    indegree: int = 1
    draws: int = 1500
    tau2: float = 1.0
    parent_kernel: KernelFamily = field(default_factory=lambda: KernelFamily("se", 2.0))
    root_kernel: KernelFamily = field(default_factory=lambda: KernelFamily("se", 0.8))
    delta_x: float = 1.0
    fusion_mode: str = "product"
    seed: int = 0

    def root_corr(self) -> float:
        return float(self.root_kernel.corr(self.delta_x**2))


def simulate_indegree_block(cfg: IndegreeExperimentConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    root_corr = cfg.root_corr()
    root_var = 2.0 * cfg.tau2 * (1.0 - root_corr)

    deltas = np.empty((cfg.depth, cfg.draws, cfg.width), dtype=float)
    deltas[0] = sample_gaussian_contrasts(root_var * np.ones((cfg.draws, cfg.width)), rng)
    parent_idx = circulant_parents(cfg.width, cfg.indegree)

    for ell in range(1, cfg.depth):
        prev = deltas[ell - 1]
        parent_delta_sq = prev[:, parent_idx] ** 2
        parent_sq = np.sum(parent_delta_sq, axis=2)
        parent_corr = cfg.parent_kernel.corr(parent_sq)
        if cfg.fusion_mode == "product":
            corr = parent_corr
        else:
            raise ValueError(f"Unsupported fusion mode {cfg.fusion_mode!r}.")
        variance = 2.0 * cfg.tau2 * (1.0 - corr)
        deltas[ell] = sample_gaussian_contrasts(variance, rng)

    node_second_moment = np.mean(deltas**2, axis=1)
    return {"C_common": np.mean(node_second_moment, axis=1)}


@dataclass
class OutdegreeExperimentConfig:
    depth: int = 9
    draws: int = 700
    branching_factor: int = 2
    tau2: float = 1.0
    kernel: KernelFamily = field(default_factory=lambda: KernelFamily("se", 1.0))
    root_delta: float = 1.0
    threshold: float = 0.45
    seed: int = 0

    def root_variance(self) -> float:
        root_corr = float(self.kernel.corr(self.root_delta**2))
        return float(2.0 * self.tau2 * (1.0 - root_corr))


def simulate_outdegree_block(cfg: OutdegreeExperimentConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    M = np.empty((cfg.depth + 1, cfg.draws), dtype=float)
    root_var = cfg.root_variance()

    for draw_idx in range(cfg.draws):
        current = sample_gaussian_contrasts(np.asarray([root_var], dtype=float), rng)
        M[0, draw_idx] = float(np.max(np.abs(current)))
        for ell in range(1, cfg.depth + 1):
            variance = 2.0 * cfg.tau2 * (1.0 - cfg.kernel.corr(current**2))
            current = sample_gaussian_contrasts(np.repeat(variance, cfg.branching_factor), rng)
            M[ell, draw_idx] = float(np.max(np.abs(current)))

    exceedance = (M > float(cfg.threshold)).astype(float)
    survival = np.minimum.accumulate(exceedance, axis=0)
    return {"survival_probability": np.mean(survival, axis=1)}


@dataclass
class PosteriorRefreshConfig:
    depth: int = 4
    draws: int = 20000
    epsilon: float = 0.35
    source_prior_var: float = 1.0
    observation_noise_var: float = 0.20
    observed_gap: float = 1.0
    route_variance: float = 0.60
    carry_variance: float = 0.02
    route_kernel: KernelFamily = field(default_factory=lambda: KernelFamily("se", 0.55))
    carry_kernel: KernelFamily = field(default_factory=lambda: KernelFamily("se", 0.55))
    seed: int = 0


def gaussian_tail_prob(epsilon: float, variance: float) -> float:
    variance = float(variance)
    if variance <= 0:
        return 0.0 if epsilon > 0 else 1.0
    return float(2.0 * ndtr(-epsilon / math.sqrt(variance)))


def gaussian_refresh_posterior(*, prior_var: float, observation_noise_var: float, observed_gap: float) -> tuple[float, float]:
    denom = float(prior_var + 2.0 * observation_noise_var)
    mean = float(prior_var / denom * observed_gap)
    var = float(2.0 * observation_noise_var * prior_var / denom)
    return mean, var


def gaussian_refresh_source_strength(mean: float, var: float, epsilon: float) -> float:
    if var <= 0.0:
        return float(abs(mean) > epsilon)
    std = math.sqrt(var)
    return float(ndtr((-epsilon - mean) / std) + (1.0 - ndtr((epsilon - mean) / std)))


def route_retention_step_probability(*, epsilon: float, route_variance: float, route_kernel: KernelFamily) -> float:
    retained_corr = float(route_kernel.corr(epsilon**2))
    step_var = 2.0 * route_variance * (1.0 - retained_corr)
    return gaussian_tail_prob(epsilon, step_var)


def simulate_posterior_refresh_panel(cfg: PosteriorRefreshConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    mean0, var0 = gaussian_refresh_posterior(
        prior_var=cfg.source_prior_var,
        observation_noise_var=cfg.observation_noise_var,
        observed_gap=cfg.observed_gap,
    )
    q_source = gaussian_refresh_source_strength(mean0, var0, cfg.epsilon)
    beta = route_retention_step_probability(
        epsilon=cfg.epsilon,
        route_variance=cfg.route_variance,
        route_kernel=cfg.route_kernel,
    )

    depths = np.arange(cfg.depth + 1, dtype=int)
    chain_source = rng.normal(loc=mean0, scale=math.sqrt(var0), size=cfg.draws)
    chain_prob = np.empty(cfg.depth + 1, dtype=float)
    chain_prob[0] = float(np.mean(np.abs(chain_source) > cfg.epsilon))
    chain_state = chain_source.copy()
    for h in range(1, cfg.depth + 1):
        var = 2.0 * cfg.route_variance * (1.0 - cfg.route_kernel.corr(chain_state**2))
        chain_state = sample_gaussian_contrasts(var, rng)
        chain_prob[h] = float(np.mean(np.abs(chain_state) > cfg.epsilon))

    source1 = rng.normal(loc=mean0, scale=math.sqrt(var0), size=cfg.draws)
    source2 = rng.normal(loc=mean0, scale=math.sqrt(var0), size=cfg.draws)
    dag_prob = np.empty(cfg.depth + 1, dtype=float)
    dag_prob[0] = float(np.mean(np.maximum(np.abs(source1), np.abs(source2)) > cfg.epsilon))

    branch1_states = [source1.copy()]
    branch2_states = [source2.copy()]
    current1 = source1.copy()
    current2 = source2.copy()
    for _ in range(1, cfg.depth):
        var1 = 2.0 * cfg.route_variance * (1.0 - cfg.route_kernel.corr(current1**2))
        var2 = 2.0 * cfg.route_variance * (1.0 - cfg.route_kernel.corr(current2**2))
        current1 = sample_gaussian_contrasts(var1, rng)
        current2 = sample_gaussian_contrasts(var2, rng)
        branch1_states.append(current1.copy())
        branch2_states.append(current2.copy())

    for h in range(1, cfg.depth + 1):
        route_state1 = branch1_states[h - 1]
        route_state2 = branch2_states[h - 1]
        var1 = (
            2.0 * cfg.route_variance * (1.0 - cfg.route_kernel.corr(route_state1**2))
            + 2.0 * cfg.carry_variance * (1.0 - cfg.carry_kernel.corr(source2**2))
        )
        var2 = (
            2.0 * cfg.route_variance * (1.0 - cfg.route_kernel.corr(route_state2**2))
            + 2.0 * cfg.carry_variance * (1.0 - cfg.carry_kernel.corr(source1**2))
        )
        target1 = sample_gaussian_contrasts(var1, rng)
        target2 = sample_gaussian_contrasts(var2, rng)
        dag_prob[h] = float(np.mean(np.maximum(np.abs(target1), np.abs(target2)) > cfg.epsilon))

    chain_bound = np.asarray([q_source * (beta**h) for h in depths], dtype=float)
    dag_bound = np.asarray([1.0 - (1.0 - q_source * (beta**h)) ** 2 for h in depths], dtype=float)
    return {
        "depths": depths,
        "chain_prob": chain_prob,
        "dag_prob": dag_prob,
        "chain_bound": chain_bound,
        "dag_bound": dag_bound,
    }


def rbf(da: np.ndarray | float, db: np.ndarray | float, sigma: float, ell: float) -> np.ndarray:
    da = np.asarray(da, dtype=float)
    db = np.asarray(db, dtype=float)
    return sigma**2 * np.exp(-((da - db) ** 2) / (2.0 * ell**2))


def sample_joint_gp(Kaa: np.ndarray, Kab: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sqrt_Kaa = np.sqrt(np.maximum(Kaa, 0.0))
    z1 = rng.standard_normal(Kaa.shape[0])
    fa = sqrt_Kaa * z1

    ratio = np.where(Kaa > 1e-12, Kab / Kaa, 0.0)
    cond_var = np.maximum(Kaa - Kab**2 / np.maximum(Kaa, 1e-12), 0.0)
    z2 = rng.standard_normal(Kaa.shape[0])
    fb = ratio * fa + np.sqrt(cond_var) * z2
    return fa, fb


def simulate_proper_dagdgp(*, L: int, W: int, s: int, sigma: float, ell: float, delta_r: float, rng: np.random.Generator) -> np.ndarray:
    Fa = np.zeros(W)
    Fb = np.full(W, delta_r)

    M = np.empty(L, dtype=float)
    M[0] = np.abs(Fa - Fb).max()

    for t in range(1, L):
        sep_mask = np.zeros(W, dtype=bool)
        if s > 0:
            sep_mask[rng.choice(W, size=min(s, W), replace=False)] = True

        parents = rng.integers(0, W, size=W)
        Fp_a = Fa[parents]
        Fp_b = Fb[parents]

        Kaa = sigma**2 * np.ones(W)
        Kab = rbf(Fp_a, Fp_b, sigma, ell)

        root_kaa = sigma**2
        root_kab = float(rbf(0.0, delta_r, sigma, ell))
        Kaa[sep_mask] += root_kaa
        Kab[sep_mask] += root_kab

        Fa, Fb = sample_joint_gp(Kaa, Kab, rng)
        M[t] = np.abs(Fa - Fb).max()

    return M


def compute_theorem1_panel_data(*, fast: bool) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(0)

    L = 60
    W = 16
    sigma = 1.0
    ell = 1.0
    delta_r = 0.3
    n_runs = 700 if fast else 1500
    epsilon = 0.30
    s_grid = np.arange(0, 11, dtype=int)
    tail_start = 30

    v_star = 2.0 * sigma**2 * (1.0 - np.exp(-delta_r**2 / (2.0 * ell**2)))
    p_eps = float(2.0 * ndtr(-epsilon / np.sqrt(v_star)))

    empirical_mean = []
    empirical_std = []
    for s in s_grid:
        trajs = np.empty((n_runs, L), dtype=float)
        for r in range(n_runs):
            trajs[r] = simulate_proper_dagdgp(
                L=L,
                W=W,
                s=int(s),
                sigma=sigma,
                ell=ell,
                delta_r=delta_r,
                rng=rng,
            )
        persistent = np.mean(trajs[:, tail_start:] > epsilon, axis=1)
        empirical_mean.append(float(np.mean(persistent)))
        empirical_std.append(float(np.std(persistent)))

    return {
        "p_eps": np.asarray(p_eps),
        "s_grid": s_grid,
        "empirical_mean": np.asarray(empirical_mean),
        "empirical_std": np.asarray(empirical_std),
    }


def make_figure(*, fast: bool) -> plt.Figure:
    theorem1 = compute_theorem1_panel_data(fast=fast)

    depth = 50
    draws_indegree = 1400 if fast else 2400
    k_values = [1, 2, 4, 8]
    indegree_curves: dict[int, np.ndarray] = {}
    for i, k in enumerate(k_values):
        indegree_curves[k] = simulate_indegree_block(
            IndegreeExperimentConfig(
                depth=depth,
                width=128,
                indegree=k,
                draws=draws_indegree,
                parent_kernel=KernelFamily("se", 2.0),
                root_kernel=KernelFamily("se", 0.8),
                delta_x=1.0,
                fusion_mode="product",
                seed=200 + i,
            )
        )["C_common"]

    threshold = 0.45
    out_depths = np.arange(1, 10)
    b_values = [1, 2, 3, 5]
    outdegree_depths = [3, 6, 9]
    branching_curves: dict[int, np.ndarray] = {}
    for i, b in enumerate(b_values):
        branching_curves[b] = simulate_outdegree_block(
            OutdegreeExperimentConfig(
                depth=int(out_depths[-1]),
                draws=220 if fast else 700,
                branching_factor=int(b),
                tau2=1.0,
                kernel=KernelFamily("se", 1.0),
                root_delta=1.0,
                threshold=threshold,
                seed=600 + 13 * i,
            )
        )["survival_probability"]

    refresh = simulate_posterior_refresh_panel(
        PosteriorRefreshConfig(
            depth=4 if fast else 6,
            draws=20000 if fast else 50000,
            epsilon=0.35,
            source_prior_var=1.0,
            observation_noise_var=0.20,
            observed_gap=1.0,
            route_variance=0.60,
            carry_variance=0.02,
            route_kernel=KernelFamily("se", 0.55),
            carry_kernel=KernelFamily("se", 0.55),
            seed=1041,
        )
    )

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 3.5), constrained_layout=True)

    ax = axes[0]
    add_axis_style(ax)
    s_grid = theorem1["s_grid"]
    p_eps = float(theorem1["p_eps"])
    s_smooth = np.linspace(float(s_grid[0]), float(s_grid[-1]), 250)
    lower_bound = 1.0 - (1.0 - p_eps) ** s_smooth
    ax.plot(s_smooth, lower_bound, color=PAPER["navy"], linewidth=1.6, label=r"lower bound $1-(1-p_\varepsilon)^s$")
    ax.errorbar(
        s_grid,
        theorem1["empirical_mean"],
        yerr=theorem1["empirical_std"],
        fmt="o",
        color="black",
        markersize=4.2,
        capsize=2.2,
        linewidth=1.0,
        elinewidth=0.9,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=r"empirical $\hat{\rho}_\varepsilon$",
    )
    ax.set_title(r"$\mathbf{a.}$   Separating nodes prevent prior collapse")
    ax.set_xlabel(r"Separating nodes per slice $s$")
    ax.set_ylabel(r"Persistent fraction $\hat{\rho}_\varepsilon$")
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks(range(0, 11, 2))
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.75", borderpad=0.35)

    ax = axes[1]
    add_axis_style(ax)
    indegree_colors = [PAPER["navy"], PAPER["purple"], PAPER["teal"], PAPER["terracotta"]]
    for color, k in zip(indegree_colors, k_values):
        ax.plot(out_depths, indegree_curves[k][out_depths], marker="o", linewidth=1.8, markersize=3.8, color=color, label=fr"$k={k}$")
    ax.set_yscale("log")
    ax.set_title(r"$\mathbf{b.}$   Effect of indegree, $k$, under product fusion")
    ax.set_xlabel(r"Depth $\ell$")
    ax.set_ylabel(r"Expected squared contrast $C_\ell$")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="0.75", loc="lower left", ncol=2, borderpad=0.35)

    ax = axes[2]
    add_axis_style(ax)
    outdegree_colors = [PAPER["navy"], PAPER["purple"], PAPER["terracotta"]]
    x_b = np.asarray(b_values, dtype=float)
    for color, depth_value in zip(outdegree_colors, outdegree_depths):
        ys = np.array([branching_curves[int(b)][depth_value] for b in b_values], dtype=float)
        ax.plot(x_b, ys, marker="o", color=color, linewidth=1.8, markersize=4.0, label=fr"$L={depth_value}$")
    ax.set_xlabel(r"Branching factor $b$")
    ax.set_ylabel(r"Empirical survival probability")
    ax.set_xlim(float(x_b[0]) - 0.1, float(x_b[-1]) + 0.1)
    ax.set_xticks(b_values)
    ax.set_ylim(-0.02, 0.62)
    ax.set_title(r"$\mathbf{c.}$   Effect of outdegree")
    ax.legend(title=r"Depth, $L$", frameon=True, framealpha=0.95, edgecolor="0.75", loc="upper left", bbox_to_anchor=(0.02, 0.98), borderpad=0.35)

    ax = axes[3]
    add_axis_style(ax)
    depths = refresh["depths"]
    ax.plot(depths, refresh["chain_prob"], marker="o", linewidth=1.8, markersize=4.2, color=PAPER["navy"], label="Chain empirical")
    ax.plot(depths, refresh["chain_bound"], "--", linewidth=1.6, color=PAPER["navy"], alpha=0.82, label="Chain theorem")
    ax.plot(depths, refresh["dag_prob"], marker="o", linewidth=1.8, markersize=4.2, color=PAPER["terracotta"], label="Disjoint-route DAG empirical")
    ax.plot(depths, refresh["dag_bound"], "--", linewidth=1.6, color=PAPER["terracotta"], alpha=0.82, label="Disjoint-route DAG theorem")
    ax.set_xlabel(r"Distance from observed antichain $h$")
    ax.set_ylabel(r"Predictive $\widehat{\Pi}_{\ell_0\to \ell_0+h}(M_{\ell_0+h}>\varepsilon)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"$\mathbf{d.}$   Refresh from Gaussian intermediate observations")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="0.75", loc="upper right", bbox_to_anchor=(1.05, 0.98), borderpad=0.35)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the main theory validation figure.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast and args.full:
        raise ValueError("Use either --fast or --full, not both.")
    fast = bool(args.fast)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_plot_style()
    fig = make_figure(fast=fast)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
