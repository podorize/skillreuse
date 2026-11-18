"""Ponto de entrada unico: ``skillreuse <subcomando> [opcoes]``."""

from __future__ import annotations

import importlib
import platform
import sys

# subcomando -> (modulo, funcao, descricao)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "check": ("skillreuse.cli", "check", "mostra as versoes do ambiente"),
    "download": ("skillreuse.download", "download_main", "baixa o GitSkills (revisao fixa)"),
    "collect": ("skillreuse.collect", "collect_main",
                "fase 1: amostra, datas das copias e estrelas (API do GitHub)"),
    "cohort": ("skillreuse.cohort", "cohort_main", "fase 2: autores, divisao e atributos"),
    "eda": ("skillreuse.study", "main_eda", "fase 3: EDA, janela e regras"),
    "spot": ("skillreuse.study", "main_spot", "fase 4: spot-checking de 7 algoritmos"),
    "tune": ("skillreuse.study", "main_tune", "fase 4: busca aninhada"),
    "final": ("skillreuse.study", "main_final", "fase 5: modelos finais e referencia"),
    "freeze": ("skillreuse.study", "main_freeze", "fase 5: congelamento e limiar"),
    "holdout": ("skillreuse.study", "main_holdout", "fase 6: avaliacao unica"),
    "control": ("skillreuse.study", "main_control",
                "controle com popularidade no retrato (desvio)"),
    "sensitivity": ("skillreuse.study", "main_sensitivity", "fase 7: sensibilidades"),
    "interpret": ("skillreuse.study", "main_interpret", "fase 7: interpretacao"),
    "figures": ("skillreuse.report", "main_figures", "figuras do artigo"),
    "reproduce": ("skillreuse.study", "main_reproduce", "confere modelos congelados no holdout"),
}


def check() -> None:
    """Exibe as versoes centrais para confirmar que o ambiente esta funcional."""
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
    print("Ambiente do projeto instalado com sucesso.")
    for package, version in versions.items():
        print(f"- {package}: {version}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in COMMANDS:
        print("uso: skillreuse <subcomando> [opcoes]\n")
        for name, (_, _, desc) in COMMANDS.items():
            print(f"  {name:<16}{desc}")
        raise SystemExit(0 if not argv or argv[0] in {"-h", "--help"} else 2)
    name, rest = argv[0], argv[1:]
    module, func, _ = COMMANDS[name]
    sys.argv = [f"skillreuse {name}", *rest]
    getattr(importlib.import_module(module), func)()


if __name__ == "__main__":
    main()
