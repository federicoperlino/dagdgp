"""Normalized local Shapley shares from the four published-split DAG-SVI fits."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.HeavyIon.common.published import PUBLISHED_SEEDS

MAIN_FIGURE_WIDTH_IN = 512.4 / 72.0
MAIN_FIGURE_HEIGHT_IN = 282.0 / 72.0
PAPER_COLORS = {"l1": "#1b9e77", "l2": "#d95f02", "grid": "#d8d4cd"}
HERE = Path(__file__).resolve().parents[1]


def _knn_conditional_mean_variance_1d(
    h: np.ndarray, x: np.ndarray, *, num_neighbors: int
) -> float:
    sample_count = h.shape[0]
    k = int(max(2, min(num_neighbors, sample_count)))
    distances = np.abs(x[:, None] - x[None, :])
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    conditional_mean = h[indices].mean(axis=1)
    return float(np.var(conditional_mean, ddof=0))


def _knn_conditional_mean_variance_2d(
    h: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    num_neighbors: int,
) -> float:
    sample_count = h.shape[0]
    k = int(max(2, min(num_neighbors, sample_count)))
    design = np.stack([x1, x2], axis=1)
    design = (design - design.mean(axis=0, keepdims=True)) / np.maximum(
        design.std(axis=0, keepdims=True), 1e-12
    )
    squared_norms = (design * design).sum(axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * design @ design.T
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    conditional_mean = h[indices].mean(axis=1)
    return float(np.var(conditional_mean, ddof=0))


def _knn_predict_2d(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    query_x1: np.ndarray,
    query_x2: np.ndarray,
    *,
    num_neighbors: int,
) -> np.ndarray:
    train = np.stack([x1, x2], axis=1).astype(np.float64)
    query = np.stack([query_x1, query_x2], axis=1).astype(np.float64)
    train_mean = train.mean(axis=0, keepdims=True)
    train_std = np.maximum(train.std(axis=0, keepdims=True), 1e-12)
    train_stdzd = (train - train_mean) / train_std
    query_stdzd = (query - train_mean) / train_std

    train_sq = np.sum(train_stdzd * train_stdzd, axis=1, keepdims=True)
    query_sq = np.sum(query_stdzd * query_stdzd, axis=1, keepdims=True)
    distances = query_sq + train_sq.T - 2.0 * query_stdzd @ train_stdzd.T
    distances = np.maximum(distances, 0.0)

    sample_count = train.shape[0]
    k = int(max(3, min(num_neighbors, sample_count)))
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    local_distances = np.take_along_axis(distances, indices, axis=1)
    local_targets = y[indices]

    local_scale = np.maximum(local_distances[:, -1:], 1e-9)
    weights = np.exp(-0.5 * local_distances / local_scale)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sum(weights * local_targets, axis=1)


def _mean_local_share_l2(
    h: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    *,
    num_neighbors: int,
) -> np.ndarray:
    num_points = h.shape[0]
    baseline_predictions = _knn_predict_2d(
        l1, l2, h, l1, l2, num_neighbors=num_neighbors
    )
    baseline = float(np.mean(baseline_predictions))

    query_l1 = np.repeat(l1, num_points)
    query_l2 = np.tile(l2, num_points)
    v1_grid = _knn_predict_2d(
        l1, l2, h, query_l1, query_l2, num_neighbors=num_neighbors
    ).reshape(num_points, num_points)

    query_l1 = np.tile(l1, num_points)
    query_l2 = np.repeat(l2, num_points)
    v2_grid = _knn_predict_2d(
        l1, l2, h, query_l1, query_l2, num_neighbors=num_neighbors
    ).reshape(num_points, num_points)

    v1 = v1_grid.mean(axis=1)
    v2 = v2_grid.mean(axis=1)
    v12 = baseline_predictions
    phi1 = 0.5 * ((v1 - baseline) + (v12 - v2))
    phi2 = 0.5 * ((v2 - baseline) + (v12 - v1))
    denom = np.maximum(np.abs(phi1) + np.abs(phi2), 1e-12)
    return np.abs(phi2) / denom


def _variance_local_share_l2(
    h: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    *,
    num_neighbors: int,
) -> np.ndarray:
    num_points = h.shape[1]
    phi1 = np.empty(num_points, dtype=np.float64)
    phi2 = np.empty(num_points, dtype=np.float64)
    for point_index in range(num_points):
        h_point = h[:, point_index]
        l1_point = l1[:, point_index]
        l2_point = l2[:, point_index]
        v1 = _knn_conditional_mean_variance_1d(
            h_point, l1_point, num_neighbors=num_neighbors
        )
        v2 = _knn_conditional_mean_variance_1d(
            h_point, l2_point, num_neighbors=num_neighbors
        )
        v12 = _knn_conditional_mean_variance_2d(
            h_point, l1_point, l2_point, num_neighbors=num_neighbors
        )
        phi1[point_index] = 0.5 * (v1 + v12 - v2)
        phi2[point_index] = 0.5 * (v2 + v12 - v1)
    denom = np.maximum(np.abs(phi1) + np.abs(phi2), 1e-12)
    return np.abs(phi2) / denom


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.7,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "grid.color": PAPER_COLORS["grid"],
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_shapley_boxplots(
    mean_l2: np.ndarray, var_l2: np.ndarray, *, output_path: Path
) -> None:
    _set_plot_style()
    mean_l1 = 1.0 - mean_l2
    var_l1 = 1.0 - var_l2

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(MAIN_FIGURE_WIDTH_IN, MAIN_FIGURE_HEIGHT_IN),
        constrained_layout=True,
    )
    positions = [1.0, 2.0, 4.0, 5.0]
    values = [mean_l1, mean_l2, var_l1, var_l2]
    colors = [
        PAPER_COLORS["l1"],
        PAPER_COLORS["l2"],
        PAPER_COLORS["l1"],
        PAPER_COLORS["l2"],
    ]
    labels = [r"$L_1$", r"$L_2$", r"$L_1$", r"$L_2$"]

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9},
        boxprops={"linewidth": 0.9},
        flierprops={"marker": "", "markersize": 0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.82)

    rng = np.random.default_rng(12345)
    for pos, point_values, color in zip(positions, values, colors):
        jitter = rng.uniform(-0.11, 0.11, size=point_values.shape[0])
        ax.scatter(
            np.full(point_values.shape[0], pos) + jitter,
            point_values,
            s=15,
            color=color,
            alpha=0.32,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xlim(0.45, 5.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized Shapley share")
    ax.grid(axis="y")
    ax.text(
        1.5,
        -0.16,
        "Predictive Mean",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        4.5,
        -0.16,
        "Posterior Variance",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    fig.savefig(output_path)
    plt.close(fig)


def compute_shares(samples):
    mean, variance = [], []
    for seed in range(len(samples["h"])):
        h, l1, l2 = (samples[n][seed] for n in ("h", "l1", "l2"))
        mean.append(
            _mean_local_share_l2(h.mean(0), l1.mean(0), l2.mean(0), num_neighbors=10)
        )
        variance.append(_variance_local_share_l2(h, l1, l2, num_neighbors=8))
    return np.mean(mean, axis=0), np.mean(variance, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=HERE / "results/published",
        help="Published-split run directory containing DAG-SVI fits for seeds 0--3.",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "figures/local_shapley_share_boxplots.pdf"
    )
    args = parser.parse_args()
    import torch

    samples = {n: [] for n in ("l1", "l2", "h")}
    for seed in PUBLISHED_SEEDS:
        payload = torch.load(
            args.runs / f"seed_{seed}/dag_svi/posterior_test.pt",
            map_location="cpu",
            weights_only=True,
        )
        for n in samples:
            samples[n].append(payload["node_samples"][n].numpy())
    mean, variance = compute_shares(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_shapley_boxplots(mean, variance, output_path=args.output)
    np.savez_compressed(
        args.output.with_suffix(".npz"), mean_l2=mean, variance_l2=variance
    )


if __name__ == "__main__":
    main()
