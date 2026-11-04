"""Continue H pretraining while keeping the two parent models fixed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dataclasses import asdict

import torch

from experiments.HeavyIon.common.cli import arguments
from experiments.HeavyIon.common.evaluation import ValidationTrace
from experiments.HeavyIon.common.protocols import dag_spec, load_data
from experiments.HeavyIon.common.real_config import set_all_seeds
from experiments.HeavyIon.common.real_outputs import cpu_state_dict, write_json
from experiments.HeavyIon.models.dag import (
    build_dag_vi_model,
    configure_inducing_trainability,
    pretrain_dag_vi_model,
)


def main():
    args = arguments("counterfactual", ["dag_vi"])
    for fold in args.folds:
        dest = args.output / f"fold_{fold}"
        dest.mkdir(parents=True, exist_ok=False)
        data = load_data("joint", fold=fold)
        spec = dag_spec("joint", "dag_vi", variant="predmean", smoke=args.smoke)
        set_all_seeds(spec.seed)
        model = build_dag_vi_model(
            data=data, spec=spec, device=args.device, dtype=torch.float64
        ).to(device=args.device, dtype=torch.float64)
        configure_inducing_trainability(model, spec=spec)
        trace = ValidationTrace(
            data, device=args.device, spec=spec, joint=True, smoke=args.smoke
        )
        parents = {}

        def hook(event, current):
            trace(event, current)
            if event["node"] == "h" and event["step"] == spec.pretrain_steps_h:
                parents.update(
                    {
                        k: v.detach().clone()
                        for k, v in current.state_dict().items()
                        if ".l1." in k or ".l2." in k
                    }
                )

        pretrain_dag_vi_model(
            model,
            data,
            spec=spec,
            hook=hook,
            log_every=1 if args.smoke else 100,
            h_continue_steps=2 if args.smoke else 12000,
            h_continue_log_every=1 if args.smoke else 500,
        )
        assert parents and all(
            torch.equal(value, model.state_dict()[key])
            for key, value in parents.items()
        )
        write_json(
            dest / "config.json",
            dict(
                spec=asdict(spec),
                fold=fold,
                smoke=args.smoke,
                additional_h_steps=2 if args.smoke else 12000,
                optimizer_reset=False,
            ),
        )
        write_json(dest / "validation_trace.json", trace.records)
        torch.save(cpu_state_dict(model), dest / "model.pt")


if __name__ == "__main__":
    main()
