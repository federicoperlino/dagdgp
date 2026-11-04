"""Run all HeavyIon protocols, then produce the five tables and two figures."""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-dgmgp",
        action="store_true",
        help="Run the public models without the local d-GMGP baseline.",
    )
    args = parser.parse_args()
    if not args.skip_dgmgp and not (HERE / "private/dgmgp.py").is_file():
        parser.error(
            "d-GMGP is a local extension. Use --skip-dgmgp or provide private/dgmgp.py."
        )
    output = (
        args.output or HERE / ("smoke_results" if args.smoke else "results")
    ).resolve()
    for protocol, script in (
        ("published", "run_published_split.py"),
        ("repeated", "run_repeated_cv.py"),
        ("joint", "run_joint_cv.py"),
        ("counterfactual", "run_counterfactual.py"),
    ):
        command = [
            sys.executable,
            "-B",
            str(HERE / "runners" / script),
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--output",
            str(output / protocol),
        ]
        if args.smoke:
            command.append("--smoke")
        if protocol == "repeated" and args.skip_dgmgp:
            command += ["--models", "mogp", "dag_vi", "dag_svi"]
        subprocess.run(command, check=True)
    if not args.smoke:
        for script, source, target in (
            ("render_results.py", output, output / "tables"),
            (
                "render_shapley.py",
                output / "published",
                output / "figures/local_shapley_share_boxplots.pdf",
            ),
            (
                "render_counterfactual.py",
                output,
                output / "figures/pretraining_counterfactual.pdf",
            ),
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "reporting" / script),
                    "--runs",
                    str(source),
                    "--output",
                    str(target),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
