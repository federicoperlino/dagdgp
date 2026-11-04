"""Build the five paper tables from completed experiment runs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.HeavyIon.common.published import PUBLISHED_SEEDS

HERE = Path(__file__).resolve().parents[1]
LABELS = {
    "mf_dgp_l1": "MF-DGP (L1)",
    "mf_dgp_l2": "MF-DGP (L2)",
    "dgp_x": r"2-DGP $(X)$",
    "dgp": r"2-DGP $(X,L_1,L_2)$",
    "mogp": "MO-GP",
    "dgmgp": "d-GMGP",
    "dag_vi": "DAG-VI",
    "dag_svi": "DAG-SVI",
}
BASELINES = [
    (r"High-fidelity GP \citep{ji2024}", "5.49", "0.46", "3.54"),
    (r"KO-path \citep{kennedy2000}", "3.48", "0.66", "1.99"),
    (r"KO-misspecified \citep{kennedy2000}", "3.95", "0.71", "2.30"),
    (r"NARGP \citep{perdikaris2017}", "3.66", "0.73", "2.13"),
    (r"r-GMGP \citep{ji2024}", "2.92", "0.72", "1.64"),
    (r"d-GMGP \citep{ji2024}", "2.17", "0.79", "1.34"),
]


def aggregate(rows):
    result = {}
    for protocol, variant, models in [
        (
            "published",
            "",
            ["mf_dgp_l1", "mf_dgp_l2", "dgp_x", "dgp", "mogp", "dag_vi", "dag_svi"],
        ),
        (
            "repeated",
            "",
            (
                ["dgmgp"]
                if any(
                    r["protocol"] == "repeated" and r["model"] == "dgmgp" for r in rows
                )
                else []
            )
            + ["mogp", "dag_vi", "dag_svi"],
        ),
        ("joint", "predmean", ["mogp", "dag_vi", "dag_svi"]),
        ("joint", "random", ["mogp", "dag_vi", "dag_svi"]),
    ]:
        key = protocol + ("_" + variant if variant else "")
        result[key] = {}
        for model in models:
            selected = [
                r
                for r in rows
                if r["protocol"] == protocol
                and r["model"] == model
                and (protocol != "published" or int(r["seed"]) in PUBLISHED_SEEDS)
                and (
                    protocol != "joint"
                    or r["variant"] == ("shared" if model == "mogp" else variant)
                )
            ]
            expected = (
                len(PUBLISHED_SEEDS)
                if protocol == "published"
                else 50 if protocol == "repeated" else 10
            )
            if len(selected) != expected:
                raise ValueError(
                    f"{key}/{model}: expected {expected} fits, found {len(selected)}"
                )
            identifiers = [(r["seed"], r["split"], r["fold"]) for r in selected]
            if len(set(identifiers)) != expected:
                raise ValueError(f"Duplicate fit in {key}/{model}")
            normalized = {(int(s), int(r or 0), int(f or 0)) for s, r, f in identifiers}
            required = (
                {(s, 0, 0) for s in PUBLISHED_SEEDS}
                if protocol == "published"
                else {(0, s, f) for s in range(10) for f in range(5)}
                if protocol == "repeated"
                else {(0, 0, f) for f in range(10)}
            )
            if normalized != required:
                raise ValueError(
                    f"Unexpected seed or fold identifiers in {key}/{model}"
                )
            metrics = (
                ["weighted_es", "rmse", "crps", "n_rmse"]
                if protocol == "joint"
                else ["rmse", "n_rmse", "crps"]
            )
            stats = {}
            for metric in metrics:
                values = np.array([float(r[metric]) for r in selected])
                if protocol == "repeated":
                    values = np.array(
                        [
                            np.mean(
                                [
                                    float(r[metric])
                                    for r in selected
                                    if int(r["split"]) == s
                                ]
                            )
                            for s in range(10)
                        ]
                    )
                    if any(
                        sum(int(r["split"]) == s for r in selected) != 5
                        for s in range(10)
                    ):
                        raise ValueError("Every repetition must contain five folds")
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {metric} in {key}/{model}")
                stats[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
            result[key][model] = stats
    appendix = result["published"]
    result["published"] = {m: appendix[m] for m in ("dgp", "mogp", "dag_vi", "dag_svi")}
    result["published_appendix"] = {
        m: appendix[m] for m in ("mf_dgp_l1", "mf_dgp_l2", "dgp_x")
    }
    return result


def collect_runs(root):
    rows = []
    for p in root.rglob("metrics.json"):
        config = json.loads(p.with_name("config.json").read_text())
        if config.get("protocol") == "published":
            if int(config["seed"]) not in PUBLISHED_SEEDS:
                continue
        if config.get("smoke"):
            raise ValueError("Smoke results cannot be used for paper tables")
        scores = json.loads(p.read_text())
        if config.get("model") == "dgmgp":
            scores = scores["test_h"]
        row = {
            k: config.get(k, "")
            for k in ("protocol", "variant", "model", "seed", "split", "fold")
        }
        row.update(scores)
        rows.append(row)
    return rows


def render(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for table, models in summary.items():
        joint = table.startswith("joint")
        headers = (
            [
                r"Weighted ES $\downarrow$",
                r"H RMSE ($\times10^{-2}$) $\downarrow$",
                r"H CRPS ($\times10^{-2}$) $\downarrow$",
                r"H N-RMSE $\uparrow$",
            ]
            if joint
            else [
                r"RMSE ($\times10^{-2}$) $\downarrow$",
                r"N-RMSE $\uparrow$",
                r"CRPS ($\times10^{-2}$) $\downarrow$",
            ]
        )
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{l" + "c" * len(headers) + "}",
            r"\toprule",
            "Model & " + " & ".join(headers) + r" \\",
            r"\midrule",
        ]
        if table == "published":
            lines += [" & ".join(row) + r" \\" for row in BASELINES] + [r"\midrule"]
        for model, metrics in models.items():
            cells = []
            for metric, stats in metrics.items():
                factor, decimals = (
                    (100, 2)
                    if metric in ("rmse", "crps")
                    else (1, 4 if metric == "weighted_es" else 3)
                )
                mean, std = stats["mean"] * factor, stats["std"] * factor
                best = (max if metric == "n_rmse" else min)(
                    m[metric]["mean"] for m in models.values()
                )
                text = f"{mean:.{decimals}f}"
                if stats["mean"] == best and table != "published_appendix":
                    text = r"\mathbf{" + text + "}"
                cells.append("$" + text + rf" \pm {std:.{decimals}f}$")
            label = (
                "MO-GP (LMC-SVGP)"
                if table == "published_appendix" and model == "mogp"
                else LABELS[model]
            )
            label += r"$^\dagger$" if table == "published" else ""
            lines.append(label + " & " + " & ".join(cells) + r" \\")
        captions = {
            "published_appendix": r"Additional baselines on the published split (75 test points), mean\,$\pm$\,population std over four seeds (0--3). Gaussian CRPS includes likelihood noise. Models reported in Table~\ref{tab:heavy_ion_results} are omitted.",
            "published": r"Published split (75 test points). Top block from \citep{ji2024}; $^\dagger$ours, mean\,$\pm$\,population std over four seeds (0--3). Our Gaussian CRPS uses predictive moments including likelihood noise.",
            "repeated": r"High-fidelity five-fold cross-validation with 200 observations at each lower fidelity and 25 at high fidelity. Metrics are averaged over five folds, then reported as mean\,$\pm$\,std over ten repetitions.",
            "joint_predmean": r"Pooled joint 10-fold cross-validation with \texttt{predmean} initialization. Mean\,$\pm$\,std across folds.",
            "joint_random": r"Pooled joint 10-fold cross-validation with \texttt{random} initialization. Mean\,$\pm$\,std across folds.",
        }
        labels = {
            "published": "tab:heavy_ion_results",
            "published_appendix": "tab:heavy_ion_additional_baselines",
            "repeated": "tab:heavy_ion_results_cv",
            "joint_predmean": "tab:joint_cv_energy_h_metrics_predmean",
            "joint_random": "tab:joint_cv_energy_h_metrics",
        }
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{" + captions[table] + "}",
            r"\label{" + labels[table] + "}",
            r"\end{table}",
        ]
        (output / f"{table}.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=HERE / "results",
        help="Root containing published/, repeated/ and joint/ runs.",
    )
    parser.add_argument("--output", type=Path, default=HERE / "tables")
    args = parser.parse_args()
    rows = collect_runs(args.runs)
    render(aggregate(rows), args.output)


if __name__ == "__main__":
    main()
