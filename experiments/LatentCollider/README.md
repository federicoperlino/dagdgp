# Latent collider

Compare compositional uncertainty and explaining away under DAG-VI and
DAG-SVI. See “Latent-Collider experiment” in the experimental appendix
of the accompanying paper.

Run from the repository root, in this order:
```bash
python experiments/LatentCollider/run_latent_collider_figure.py
python experiments/LatentCollider/render_main_results.py
python experiments/LatentCollider/render_appendix_results.py
```

Outputs: figures in `main_res/` and `appendix_res/`; metrics and plot data
in this directory.
