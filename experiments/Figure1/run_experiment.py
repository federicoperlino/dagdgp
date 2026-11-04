"""Generate the overview, ground-truth, and prediction figures."""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.Figure1 import figures, model
else:
    from . import figures, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory for the three PDF figures (existing PDFs are overwritten).",
    )
    args = parser.parse_args()
    if shutil.which("pdflatex") is None:
        parser.error("pdflatex is required; install LaTeX with the Times font package.")
    if importlib.util.find_spec("pyvista") is None:
        parser.error(
            "Install the figure dependencies with: pip install -e '.[figures]'"
        )

    figures.configure_style()
    with threadpool_limits(limits=1, user_api="blas"):
        print("Fitting the multi-fidelity model...", flush=True)
        posterior = model.fit()
        baseline = figures.gp_on_x(posterior)
        print("Rendering the high-fidelity volume...", flush=True)
        image, anchors = figures.render(posterior)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        figures.compose(posterior, image, anchors, args.output_dir / "fig1_dagdgp.pdf")
        figures.mpl.rcParams["font.size"] = 8
        figures.plot_fidelities(posterior, args.output_dir / "fig1_fidelities.pdf")
        figures.plot_uncertainty(
            posterior, baseline, args.output_dir / "fig1_uncertainty.pdf"
        )
    print(f"Saved three figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
