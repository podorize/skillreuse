"""Comandos de diagnostico do projeto."""

from __future__ import annotations

import platform

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import sklearn

from gitskills_reuse import __version__


def main() -> None:
    """Exibe as versoes centrais para confirmar que o ambiente esta funcional."""
    versions = {
        "skillreuse": __version__,
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "pyarrow": pa.__version__,
        "scikit-learn": sklearn.__version__,
    }

    print("Ambiente do projeto instalado com sucesso.")
    for package, version in versions.items():
        print(f"- {package}: {version}")


if __name__ == "__main__":
    main()
