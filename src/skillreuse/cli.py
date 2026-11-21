"""Single entry point of the project: ``skillreuse <command> [options]``.

Each command runs one step of the study. The order below is the order of the pipeline:
download, phase 1 (collect), phase 2 (cohort), phases 3-7 (eda ... interpret), figures.
"""

from __future__ import annotations

import importlib
import platform
import sys

# command -> (module, function, description shown by ``skillreuse --help``)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "check": ("skillreuse.cli", "check", "show the versions of the environment"),
    "download": ("skillreuse.download", "download_main", "download GitSkills (fixed revision)"),
    "collect": ("skillreuse.collect", "collect_main",
                "phase 1: sample, dates of the copies and stars (GitHub API)"),
    "cohort": ("skillreuse.cohort", "cohort_main", "phase 2: authors, split and attributes"),
    "eda": ("skillreuse.study", "main_eda", "phase 3: exploratory analysis, window and rules"),
    "spot": ("skillreuse.study", "main_spot", "phase 4: spot-check of 7 algorithms"),
    "tune": ("skillreuse.study", "main_tune", "phase 4: nested hyperparameter search"),
    "final": ("skillreuse.study", "main_final", "phase 5: final models and reference model"),
    "freeze": ("skillreuse.study", "main_freeze", "phase 5: freeze the models and the threshold"),
    "holdout": ("skillreuse.study", "main_holdout", "phase 6: single evaluation on the holdout"),
    "control": ("skillreuse.study", "main_control",
                "control with popularity at the snapshot (protocol deviation)"),
    "sensitivity": ("skillreuse.study", "main_sensitivity", "phase 7: robustness checks"),
    "interpret": ("skillreuse.study", "main_interpret", "phase 7: model interpretation"),
    "figures": ("skillreuse.report", "main_figures", "figures of the paper"),
    "reproduce": ("skillreuse.study", "main_reproduce",
                  "check the frozen models against the holdout results"),
}


def check() -> None:
    """Print the main versions, to confirm that the environment works."""
    import duckdb
    import pandas as pd
    import pyarrow as pa
    import sklearn

    from skillreuse import __version__

    versions = {
        "skillreuse": __version__,
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "scikit-learn": sklearn.__version__,
    }
    print("Project environment installed successfully.")
    for package, version in versions.items():
        print(f"- {package}: {version}")


def main(argv: list[str] | None = None) -> None:
    """Run the command named by the first argument.

    The module of the command is imported only now, so ``skillreuse --help`` does not
    have to load pandas or scikit-learn. ``sys.argv`` is rewritten so that the
    ``argparse`` of each module sees only its own options.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in COMMANDS:
        print("usage: skillreuse <command> [options]\n")
        for name, (_, _, desc) in COMMANDS.items():
            print(f"  {name:<16}{desc}")
        raise SystemExit(0 if not argv or argv[0] in {"-h", "--help"} else 2)
    name, rest = argv[0], argv[1:]
    module, func, _ = COMMANDS[name]
    sys.argv = [f"skillreuse {name}", *rest]
    getattr(importlib.import_module(module), func)()


if __name__ == "__main__":
    main()
