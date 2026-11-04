# ELBO scaling with graph width

Compare ELBO evaluation times for DAG-VI and the dense and sparse DAG-SVI
backends as graph width increases at fixed depth. See “ELBO scaling” in the
experimental appendix of the accompanying paper.

Run from the repository root:
```bash
python experiments/KWide3Deep/run_k_wide_3_deep_elbo_scaling.py
```

Outputs: `benchmark.json` and `elbo_eval_scaling.pdf` in this directory.
