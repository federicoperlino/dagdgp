"""Fit all seven models for the main and appendix published-split tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.HeavyIon.common.cli import arguments
from experiments.HeavyIon.common.protocols import PUBLISHED_MODELS, load_data
from experiments.HeavyIon.common.runner import run_fit


def main():
    args = arguments("published", PUBLISHED_MODELS)
    data = load_data("published")
    for seed in args.seeds:
        for model in args.models:
            run_fit(
                data,
                protocol="published",
                model_name=model,
                seed=seed,
                device=args.device,
                output=args.output / f"seed_{seed}" / model,
                smoke=args.smoke,
            )


if __name__ == "__main__":
    main()
