# Illustrative multi-fidelity example

Reproduce Figure 1, along with ground truth and predictions for the
illustrative multi-fidelity DAG-DGP. See “Illustrative multi-fidelity
DAG-DGP of Figure 1” in the experimental appendix of the accompanying paper.

Run from the repository root:
```bash
pip install -e '.[figures]'
python experiments/Figure1/run_experiment.py
```

The 3D visualization in Figure 1 requires OpenGL support for PyVista/VTK.
Text and formulas require `pdflatex` with `amsmath` and Times fonts.

Outputs: `fig1_dagdgp.pdf`, `fig1_fidelities.pdf`, and
`fig1_uncertainty.pdf` in `figures/`.
