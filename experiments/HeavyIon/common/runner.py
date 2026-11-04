"""Shared fitting and scoring for the HeavyIon experiments."""

from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.HeavyIon.common.evaluation import ValidationTrace, _evaluate_payload
from experiments.HeavyIon.common.metrics import compute_metrics
from experiments.HeavyIon.common.protocols import PUBLISHED_TOTAL_UPDATES, dag_spec
from experiments.HeavyIon.common.real_config import set_all_seeds
from experiments.HeavyIon.common.real_outputs import (
    cpu_state_dict,
    posterior_payload,
    write_json,
)
from experiments.HeavyIon.common.weighted_es import weighted_energy_score
from experiments.HeavyIon.models.dag import (
    configure_inducing_trainability,
    fit_dag_vi_model,
    fit_dag_svi_model,
    pretrain_dag_vi_model,
)


def score_h(payload, data, *, likelihood_variance=0.0):
    """Compute H prediction metrics, optionally adding likelihood noise to the variance."""
    mean = payload["summary"]["h"]["mean"].numpy()
    var = payload["summary"]["h"]["variance"].numpy()
    return compute_metrics(
        data.y_test, mean, var + likelihood_variance, data.baseline_mean
    )


def score_joint(train_payload, test_payload, data):
    baselines = {n: float(data.node_targets[n].mean()) for n in ("l1", "l2", "h")}
    metrics = _evaluate_payload(test_payload, baselines=baselines)["h"]
    metrics["weighted_es"] = weighted_energy_score(train_payload, test_payload)[
        "weighted_es"
    ]
    return metrics


def run_fit(
    data,
    *,
    protocol,
    model_name,
    output,
    device,
    seed=0,
    split=0,
    fold=0,
    variant="predmean",
    smoke=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dtype = torch.float64
    trace = None
    joint = protocol == "joint"
    if model_name == "mogp":
        from experiments.HeavyIon.common.mogp_cv import (
            posterior_payload as mogp_payload,
        )
        from experiments.HeavyIon.common.mogp_cv import prepare_fold
        from experiments.HeavyIon.models.mogp import (
            MOGPRunSpec,
            _evaluate_mogp,
            fit_mogp,
        )

        spec = MOGPRunSpec(
            seed=seed,
            steps=2
            if smoke
            else (PUBLISHED_TOTAL_UPDATES if protocol == "published" else 12000),
        )
        adapted, validation = (
            (data, None) if protocol == "published" else prepare_fold(data, joint=joint)
        )
        model, likelihood, _, _, _, train_x, test_x = fit_mogp(
            adapted, spec=spec, device=device, dtype=dtype
        )
        analytic, predictions = _evaluate_mogp(
            data=adapted,
            model=model,
            train_x=train_x,
            test_x=test_x,
            jitter=spec.jitter,
            likelihood=likelihood if protocol == "published" else None,
        )
        if protocol == "published":
            import numpy as np

            np.savez_compressed(output / "predictions.npz", **predictions)
            metrics = analytic["test_h"]
            write_json(output / "metrics_latent.json", analytic["test_h_latent"])
            train = test = None
        else:
            set_all_seeds(81723 + (5 * split if not joint else 0) + fold)
            samples = 8 if smoke else 1024
            train = mogp_payload(
                model,
                adapted.train_data,
                device=device,
                dtype=dtype,
                num_samples=samples,
            )
            test = mogp_payload(
                model, validation, device=device, dtype=dtype, num_samples=samples
            )
            metrics = None
        torch.save(cpu_state_dict(likelihood), output / "likelihood.pt")
    else:
        spec = dag_spec(protocol, model_name, seed=seed, variant=variant, smoke=smoke)
        if protocol != "published":
            trace = ValidationTrace(
                data, device=device, spec=spec, joint=joint, smoke=smoke
            )
        if model_name in {"dgp", "dgp_x", "mf_dgp_l1", "mf_dgp_l2"}:
            from experiments.HeavyIon.common.training import DAGVIFitConfig, fit_dag_vi
            from experiments.HeavyIon.models.dgp import (
                build_input_dgp2_mf_inputs_model,
                build_input_dgp2_model,
                build_single_parent_model,
            )

            set_all_seeds(seed)
            if model_name.startswith("mf_dgp_"):
                model = build_single_parent_model(
                    data,
                    spec,
                    parent=model_name.removeprefix("mf_dgp_"),
                    device=device,
                    dtype=dtype,
                )
            else:
                builder = (
                    build_input_dgp2_model
                    if model_name == "dgp_x"
                    else build_input_dgp2_mf_inputs_model
                )
                model = builder(data, spec, device=device, dtype=dtype)
            model = model.to(device=device, dtype=dtype)
            configure_inducing_trainability(model, spec=spec)
            if model_name != "dgp_x":
                pretrain_dag_vi_model(
                    model,
                    data,
                    spec=replace(spec, pretrain_steps_h=0)
                    if model_name == "dgp"
                    else spec,
                )
            fit_dag_vi(
                model,
                data.train_data,
                DAGVIFitConfig(
                    steps=spec.joint_steps,
                    joint_lr=spec.joint_lr,
                    likelihood_lr_scale=spec.likelihood_lr_scale,
                    batch_size=spec.batch_size,
                    num_mc_samples=spec.num_mc_samples,
                    seed=seed,
                ),
            )
            model.eval()
        else:
            fit = fit_dag_vi_model if model_name == "dag_vi" else fit_dag_svi_model
            model = fit(
                data,
                spec=spec,
                dtype=dtype,
                device=device,
                pretrain_log_every=1 if smoke else 100,
                hook=trace,
            )[0]
        if joint and model_name == "dag_svi":
            train = {
                "y": {k: v.detach().cpu() for k, v in data.train_data.y.items()},
                "mask": {k: v.detach().cpu() for k, v in data.train_data.mask.items()},
            }
        else:
            train = posterior_payload(
                model,
                data.train_data.to(device=device, dtype=dtype),
                num_samples=spec.pred_samples,
            )
        if joint and model_name == "dag_svi":
            replicate_metrics = []
            for repetition in range(1 if smoke else 3):
                set_all_seeds(740000 + 100 * fold + repetition)
                prediction = posterior_payload(
                    model,
                    data.test_data.to(device=device, dtype=dtype),
                    num_samples=spec.pred_samples,
                    joint=True,
                )
                replicate_metrics.append(score_joint(train, prediction, data))
                if repetition == 0:
                    test = prediction
                else:
                    torch.save(prediction, output / f"posterior_test_rep{repetition}.pt")
            metrics = {
                name: sum(row[name] for row in replicate_metrics) / len(replicate_metrics)
                for name in replicate_metrics[0]
            }
        else:
            test = posterior_payload(
                model,
                data.test_data.to(device=device, dtype=dtype),
                num_samples=spec.pred_samples,
            )
            metrics = None
    if test is not None:
        if joint:
            if metrics is None:
                metrics = score_joint(train, test, data)
        else:
            if protocol == "published":
                noise = float(model.likelihood("h").variance.detach().cpu().item())
                write_json(output / "metrics_latent.json", score_h(test, data))
                test["likelihood_variance"] = {"h": noise}
                test["observation_summary"] = {
                    "h": {
                        "mean": test["summary"]["h"]["mean"],
                        "variance": test["summary"]["h"]["variance"] + noise,
                    }
                }
                metrics = score_h(test, data, likelihood_variance=noise)
            else:
                metrics = score_h(test, data)
        torch.save(test, output / "posterior_test.pt")
    torch.save(cpu_state_dict(model), output / "model.pt")
    write_json(
        output / "config.json",
        dict(
            protocol=protocol,
            model=model_name,
            split=split,
            fold=fold,
            variant=variant,
            smoke=smoke,
            device=str(device),
            dtype="float64",
            **asdict(spec),
        ),
    )
    if trace is not None:
        write_json(output / "validation_trace.json", trace.records)
    write_json(output / "metrics.json", metrics)
    return metrics
