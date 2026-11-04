# DAG misspecification

Evaluate the sensitivity of DAG-VI and DAG-SVI to adding or removing
parent edges. See “DAG misspecification” in the experimental appendix
of the accompanying paper.

Run from the repository root:
```bash
python experiments/DAGMisspecification/run_dag_misspecification.py
```

Outputs: `dag_misspecification_RMSE.pdf` and `dag_misspecification_CRPS.pdf`
in this directory.
