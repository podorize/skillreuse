"""Ponto de entrada unico: ``skillreuse <subcomando> [opcoes]``."""

from __future__ import annotations

import importlib
import platform
import sys

# subcomando -> (modulo, funcao, descricao)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "check": ("skillreuse.cli", "check", "mostra as versoes do ambiente"),
    "download": ("skillreuse.data", "download_main", "baixa GitSkills (revisao fixa)"),
    "build": ("skillreuse.data", "build_main", "agrega repository_count e amostra 200 mil"),
    "features": ("skillreuse.features", "features_main", "atributos e divisao externa"),
    "target": ("skillreuse.models", "prepare_target_main", "alvo de contagem (reused=1)"),
    "target-eda": ("skillreuse.report", "target_eda_main", "EDA do alvo"),
    "benchmark-views": ("skillreuse.benchmark", "views_main", "Ridge: skill/repo/hybrid"),
    "benchmark": ("skillreuse.benchmark", "benchmark_main", "6 regressores agrupados"),
    "tune": ("skillreuse.tuning", "tune_main", "busca aninhada + repeticoes + testes pareados"),
    "controls": ("skillreuse.controls", "controls_main", "controles de atalho de coleta"),
    "owner": ("skillreuse.owner", "owner_main", "alvo por donos, contexto, auditoria e EDA"),
    "owner-models": ("skillreuse.owner_models", "owner_models_main", "modelos por donos"),
    "owner-importance": ("skillreuse.owner_models", "importance_main", "permutacao (limpo)"),
    "primary": ("skillreuse.owner_models", "primary_main", "protocolo principal (H1-H4)"),
    "evaluate": ("skillreuse.benchmark", "evaluate_main", "avaliacao externa (RF e GB)"),
    "explain": ("skillreuse.explain", "explain_main", "explicabilidade do Ridge hibrido"),
    "tail-audit": ("skillreuse.explain", "tail_main", "auditoria da cauda (Ridge)"),
    "audit-text": ("skillreuse.explain", "audit_text_main", "artefatos textuais"),
    "report": ("skillreuse.report", "report_main", "tabelas e figuras do relatorio"),
}


def check() -> None:
    """Exibe as versoes centrais para confirmar que o ambiente esta funcional."""
    import duckdb
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import sklearn

    from skillreuse import __version__

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
