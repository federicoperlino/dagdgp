# DAG-DGP

Code accompanying **Deep Gaussian Processes on Directed Acyclic Graphs**.

Code authors: Federico Perlino and Oliver Hamelijnck.

This repository implements DAG-VI and DAG-SVI and provides the experiments
reported in the paper.

## Installation

Tested with Python 3.11. From the repository root:

```bash
pip install -e .
```

For Figure 1, install the additional dependencies:

```bash
pip install -e '.[figures]'
```

## Repository structure

- `dagdgp/`: models, kernels, and inference and training utilities.
- `experiments/`: experiment scripts and instructions.

## Experiments

Each directory under `experiments/` contains its own README with run
instructions and a reference to the paper. Run commands from the
repository root.
