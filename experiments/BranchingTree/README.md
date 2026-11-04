# ELBO scaling with tree depth

Compare ELBO evaluation times for DAG-VI and the dense and sparse DAG-SVI
backends as tree depth increases. See “ELBO scaling” in the experimental
appendix of the accompanying paper.

Run from the repository root:
```bash
python experiments/BranchingTree/run_branching_union_elbo_scaling.py
```

Outputs: `benchmark.json` and `elbo_eval_scaling.pdf` in this directory.
