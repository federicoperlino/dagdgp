"""Plot results comparing H-only training with joint training after H pretraining."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parents[1]


def collect_curves(counterfactual, joint):
    curves = {}
    for fold in range(10):
        red = json.loads(
            (counterfactual / f"fold_{fold}/validation_trace.json").read_text()
        )
        blue = json.loads(
            (joint / f"predmean/fold_{fold}/dag_vi/validation_trace.json").read_text()
        )
        start = [
            e
            for e in blue
            if e["phase"] == "pretrain" and e["node"] == "h" and e["step"] == 1000
        ]
        branches = {
            "h": [
                e
                for e in red
                if e["phase"] == "pretrain" and e["node"] == "h" and e["step"] >= 1000
            ],
            "joint": start + [e for e in blue if e["phase"] == "joint"],
        }
        curves[str(fold)] = {
            b: {
                "steps": [
                    e["step"] + (1000 if e["phase"] == "joint" else 0) for e in events
                ],
                **{
                    m: [100 * e[f"validation_{m}_h"] for e in events]
                    for m in ("rmse", "crps")
                },
            }
            for b, events in branches.items()
        }
    return curves


def mean_curves(curves):
    if set(curves) != set(map(str, range(10))):
        raise ValueError("The figure requires the same ten folds in both branches")
    expected = list(range(1000, 13001, 500))
    for record in curves.values():
        for branch in ("h", "joint"):
            if record[branch]["steps"] != expected:
                raise ValueError("Missing or mismatched continuation checkpoints")
        for metric in ("rmse", "crps"):
            np.testing.assert_allclose(
                record["h"][metric][0], record["joint"][metric][0], rtol=1e-5
            )
    return {
        b: {
            "steps": expected,
            **{
                m: np.mean([r[b][m] for r in curves.values()], axis=0).tolist()
                for m in ("rmse", "crps")
            },
        }
        for b in ("h", "joint")
    }


def render(curves, output):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("rmse", "crps")):
        for branch, color, label in (
            ("h", "#C44E52", "Continue H only (LF frozen)"),
            ("joint", "#4C72B0", "Joint training"),
        ):
            x = np.asarray(curves[branch]["steps"], dtype=float)
            y = np.asarray(curves[branch][metric], dtype=float)
            grid = np.unique(
                np.concatenate([np.linspace(a, b, 32) for a, b in zip(x[:-1], x[1:])])
            )
            ax.plot(
                grid,
                PchipInterpolator(x, y, extrapolate=False)(grid),
                color=color,
                lw=2.4,
                label=label,
                linestyle="--" if branch == "h" else "-",
            )
        ax.spines["left"].set_linestyle((0, (3, 3)))
        ax.spines["left"].set_color("#858B93")
        ax.spines["left"].set_linewidth(1.0)
        for offset in (0.018, 0.032):
            ax.plot(
                [offset - 0.004, offset + 0.004],
                [-0.012, 0.012],
                transform=ax.transAxes,
                color="#535860",
                lw=1.1,
                clip_on=False,
                zorder=6,
            )
        ax.set_xlabel("Updates since the start of H pretraining")
        ax.set_ylabel(f"Validation {metric.upper()} on H ($\\times 10^{{-2}}$)")
        ax.set_xlim(1000, 13200)
        ax.set_xticks(
            [1000, 4000, 7000, 10000, 13000], ["1k", "4k", "7k", "10k", "13k"]
        )
        ax.set_axisbelow(True)
        ax.grid(alpha=0.16, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=HERE / "results",
        help="Root containing counterfactual/ and joint/ runs.",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "figures/pretraining_counterfactual.pdf"
    )
    args = parser.parse_args()
    curves = collect_curves(args.runs / "counterfactual", args.runs / "joint")
    means = mean_curves(curves)
    render(means, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(means, indent=2) + "\n")


if __name__ == "__main__":
    main()
