"""Pooled 10-fold cross-validation with configurable inducing initialization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.HeavyIon.common.cli import arguments
from experiments.HeavyIon.common.protocols import load_data
from experiments.HeavyIon.common.runner import run_fit


def main():
    args = arguments("joint", ["mogp", "dag_vi", "dag_svi"])
    for fold in args.folds:
        data = load_data("joint", fold=fold)
        for model in args.models:
            for variant in ["shared"] if model == "mogp" else args.variants:
                run_fit(
                    data,
                    protocol="joint",
                    model_name=model,
                    variant=variant,
                    fold=fold,
                    output=args.output / variant / f"fold_{fold}" / model,
                    device=args.device,
                    smoke=args.smoke,
                )


if __name__ == "__main__":
    main()
