# Sachs flow cytometry

Compare DAG-VI, DAG-SVI, and MO-GP on interpolation, extrapolation, and
scalability. See “Sachs flow cytometry experiment” in the experimental
appendix of the accompanying paper.

Run from the repository root:
```bash
python experiments/SACHS/run_experiment.py --device cuda
python experiments/SACHS/render_results.py
```

Download `sachs.zip` from [Zenodo](https://doi.org/10.5281/zenodo.7681811)
and copy the 14 CSV files from its `Data Files/` directory into `private/data/`
without modifying their contents. The loader verifies their checksums against
`manifest.json`.

Data by [Sachs et al. (2005)](https://doi.org/10.1126/science.1105809), curated
by the Mathematical Research Data Initiative and distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Use `--device cpu` if CUDA is unavailable.

Outputs: run records in `reproduction/` and tables in `reproduction/tables/`.
