"""Ten repetitions of five-fold validation on the 25 high-fidelity cases."""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from experiments.HeavyIon.common.cli import arguments
from experiments.HeavyIon.common.protocols import HERE, load_data
from experiments.HeavyIon.common.real_config import set_all_seeds
from experiments.HeavyIon.common.runner import run_fit


def main():
    args = arguments("repeated", ["dgmgp", "mogp", "dag_vi", "dag_svi"])
    if "dgmgp" in args.models and not (HERE / "private/dgmgp.py").is_file():
        raise FileNotFoundError(
            "d-GMGP is a local extension. Provide private/dgmgp.py or select --models mogp dag_vi dag_svi."
        )
    full = load_data("published")
    if "dgmgp" in args.models:
        from experiments.HeavyIon.private.dgmgp import (
            DGMGPConfig,
            fit_low_fidelity,
            run_fold,
        )

        if not args.smoke and (
            args.splits != list(range(10)) or args.folds != list(range(5))
        ):
            raise ValueError(
                "d-GMGP uses a continuous optimization RNG stream; run its ten repetitions together."
            )

        config = (
            DGMGPConfig(initial_max_iters=2, restarts=0, restart_max_iters=2)
            if args.smoke
            else DGMGPConfig()
        )
        set_all_seeds(0)
        parents = fit_low_fidelity(full, config)
        dgmgp_rng = np.random.get_state()
    for split in args.splits:
        for fold in args.folds:
            data = load_data("repeated", split=split, fold=fold)
            for model in args.models:
                dest = args.output / f"split_{split}" / f"fold_{fold}" / model
                if model == "dgmgp":
                    if dest.exists():
                        raise FileExistsError(dest)
                    np.random.set_state(dgmgp_rng)
                    run_fold(
                        fold_id=fold,
                        val_indices=np.asarray(
                            data.metadata["cv"]["validation_union_indices"]
                        ),
                        output_dir=dest,
                        data=full,
                        m1=parents[0],
                        m2=parents[1],
                        config=config,
                    )
                    dgmgp_rng = np.random.get_state()
                    from experiments.HeavyIon.common.real_outputs import write_json

                    write_json(
                        dest / "config.json",
                        dict(
                            protocol="repeated",
                            model="dgmgp",
                            seed=0,
                            split=split,
                            fold=fold,
                            smoke=args.smoke,
                            variant="",
                            **asdict(config),
                        ),
                    )
                else:
                    run_fit(
                        data,
                        protocol="repeated",
                        model_name=model,
                        split=split,
                        fold=fold,
                        output=dest,
                        device=args.device,
                        smoke=args.smoke,
                    )


if __name__ == "__main__":
    main()
