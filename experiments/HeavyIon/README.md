# Heavy-ion collision emulation

Compare DAG-VI, DAG-SVI, and baseline models on multi-fidelity heavy-ion
emulation. See “Heavy-ion collision” in the experimental appendix of the
accompanying paper.

Run from the repository root:
```bash
python experiments/HeavyIon/run_all.py --device cuda:0
```

We assume the data have been prepared with the code provided by the authors of
Ji et al. (2024). The data and preparation code are not redistributed here.
Save the prepared dataset as `private/data/gmgp_real_dataset.npz`, containing
`x_l1`, `y_l1`, `x_l2`, `y_l2`, `x_h`, `y_h`, `x_test`, and `y_test`.

Fixed cross-validation partitions
are included in `splits/partitions.json`, as indices into the loader's stable
union of input rows.
Use `--skip-dgmgp` if the external d-GMGP implementation is unavailable.

Outputs: five tables and two figures under `results/`, together with
run records and model checkpoints.

The published-split tables use seeds 0--3 and Gaussian CRPS including
likelihood noise. See the experimental appendix for evaluation details.

To run only the published split:
```bash
python experiments/HeavyIon/runners/run_published_split.py --device cuda:0
```
