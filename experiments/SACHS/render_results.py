#!/usr/bin/env python3
"""Aggregate the five Sachs splits and write the two paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from config import CONFIGS, MODELS, SEEDS

DIRECTORY = Path(__file__).resolve().parent
SCORES = ("rmse", "crps", "energy_standardised")
LABELS = {"dag_vi": "DAG-VI", "dag_svi": "DAG-SVI", "mogp": "MO-GP"}


def load_runs(runs_dir: Path) -> list[dict]:
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    records = []
    for protocol, cfg in CONFIGS.items():
        for seed in SEEDS:
            for model in MODELS:
                path = runs_dir / protocol / f"seed_{seed}" / f"{model}.json"
                if not path.is_file():
                    raise ValueError(f"Missing run: {path}")
                result = json.loads(path.read_text())
                if result["config"] != asdict(cfg):
                    raise ValueError(f"Run does not use the paper configuration: {path}")
                if (result["protocol"], result["seed"], result["model"]) != (
                    protocol, seed, model
                ):
                    raise ValueError(f"Inconsistent run identifiers: {path}")
                if result["split_sha256"] != manifest["split_sha256"][f"{protocol}_{seed}.npz"]:
                    raise ValueError(f"Run uses a different split: {path}")
                records.append({
                    "protocol": protocol, "seed": seed, "model": model,
                    **{key: result["metrics"][key] for key in SCORES},
                    "training_seconds": result.get("training_seconds"),
                })
    return records


def aggregate(records: list[dict]) -> dict[str, list[dict]]:
    expected = {(p, s, m) for p in CONFIGS for s in SEEDS for m in MODELS}
    actual = [(r["protocol"], r["seed"], r["model"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Expected exactly five splits per model for each of the three protocols."
        )
    tables = {}
    for protocol, cfg in CONFIGS.items():
        rows = []
        for model in MODELS:
            runs = [r for r in records if r["protocol"] == protocol and r["model"] == model]
            row = {"model": model}
            keys = SCORES if protocol == "extrapolation" else (*SCORES, "training_seconds")
            for key in keys:
                values = [float(r[key]) for r in runs]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite {key}: {protocol}, {model}")
                label = "time_minutes" if key == "training_seconds" else key
                if key == "training_seconds":
                    if any(v < 0 for v in values):
                        raise ValueError("Training times must be nonnegative.")
                    values = [v / 60 for v in values]
                row[f"{label}_mean"] = statistics.mean(values)
                std = statistics.stdev if cfg.std_ddof else statistics.pstdev
                row[f"{label}_std"] = std(values)
            rows.append(row)
        tables[protocol] = rows
    return tables


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cells(row: dict, rows: list[dict], timed: bool) -> str:
    result = []
    for metric in (*SCORES, "time_minutes") if timed else SCORES:
        precision = 1 if metric == "time_minutes" else 3
        mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
        value = f"{mean:.{precision}f}"
        if mean == min(r[f"{metric}_mean"] for r in rows):
            value = r"\mathbf{" + value + "}"
        result.append(f"${value}\\pm{std:.{precision}f}$")
    return " & ".join(result)


def write_tables(tables: dict[str, list[dict]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for protocol, rows in tables.items():
        write_csv(output / f"{protocol}.csv", rows)
    end_row = r" \\"
    lines = [
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"$N$ & Model & RMSE $\downarrow$ & CRPS $\downarrow$ "
        r"& ES $\downarrow$ & Time (min) $\downarrow$" + end_row,
        r"\midrule",
    ]
    for protocol, n in [("interpolation", "902"), ("scalability", r"$11{,}672$")]:
        rows = tables[protocol]
        for i, row in enumerate(rows):
            model = row["model"]
            batch = (
                "1024 mini-batch"
                if protocol == "scalability" and model != "mogp"
                else "full batch"
            )
            lines.append(
                f"{n if i == 0 else ''} & {LABELS[model]} ({batch}) "
                f"& {cells(row, rows, True)}" + end_row
            )
        if protocol == "interpolation":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "main_table.tex").write_text("\n".join(lines) + "\n")
    rows = tables["extrapolation"]
    lines = [
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & RMSE $\downarrow$ & CRPS $\downarrow$ & ES $\downarrow$" + end_row,
        r"\midrule",
        *[f"{LABELS[row['model']]} & {cells(row, rows, False)}" + end_row for row in rows],
        r"\bottomrule", r"\end{tabular}",
    ]
    (output / "extrapolation_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", type=Path, default=DIRECTORY / "reproduction",
        help="Directory containing the 45 trained model runs.",
    )
    parser.add_argument("--output-dir", type=Path, help="Destination for the aggregate CSV and LaTeX tables.")
    args = parser.parse_args()
    output = args.output_dir or args.runs_dir / "tables"
    write_tables(aggregate(load_runs(args.runs_dir)), output)
    print(f"Tables written to {output}")


if __name__ == "__main__":
    main()
