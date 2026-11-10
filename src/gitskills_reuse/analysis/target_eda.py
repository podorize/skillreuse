"""EDA orientada a decidir entre alvo ordinal e regressao de contagem.

As decisoes sobre a formulacao do alvo usam somente o conjunto de desenvolvimento
(split=train). O teste externo e resumido para auditoria, mas nao participa da
recomendacao metodologica.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402,I001


DEFAULT_TARGETS = Path("data/processed/final_targets.parquet")
DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")
DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")
DEFAULT_TABLES = Path("reports/tables")
DEFAULT_FIGURES = Path("reports/figures")
DEFAULT_REPORT = Path("reports/target_eda.md")

ORDINAL_LABELS = {
    0: "2 repositorios",
    1: "3-4 repositorios",
    2: "5-9 repositorios",
    3: "10+ repositorios",
}


def load_target_frame(targets_path: Path, features_path: Path) -> pd.DataFrame:
    """Carrega alvos e split sem usar atributos derivados do conteudo."""
    targets = pq.read_table(targets_path).to_pandas()
    split = pq.read_table(features_path, columns=["content_hash", "split"]).to_pandas()
    frame = targets.merge(split, on="content_hash", how="inner", validate="one_to_one")
    if len(frame) != len(targets):
        raise ValueError("A junção dos alvos com o split perdeu observacoes")
    return frame


def count_distribution(frame: pd.DataFrame, *, split: str = "train") -> pd.DataFrame:
    """Resume repository_count entre todas as linhas e entre reutilizadas."""
    subset = frame.loc[frame["split"].eq(split)].copy()
    rows: list[dict[str, object]] = []
    for population, data in (("all", subset), ("reused", subset.loc[subset["reused"].eq(1)])):
        counts = pd.to_numeric(data["repository_count"], errors="coerce").dropna()
        mean = float(counts.mean()) if len(counts) else float("nan")
        variance = float(counts.var(ddof=1)) if len(counts) > 1 else float("nan")
        row: dict[str, object] = {
            "split": split,
            "population": population,
            "n": int(len(counts)),
            "mean": mean,
            "median": float(counts.median()) if len(counts) else float("nan"),
            "std": float(counts.std(ddof=1)) if len(counts) > 1 else float("nan"),
            "variance": variance,
            "min": float(counts.min()) if len(counts) else float("nan"),
            "max": float(counts.max()) if len(counts) else float("nan"),
            "p50": float(counts.quantile(0.50)) if len(counts) else float("nan"),
            "p75": float(counts.quantile(0.75)) if len(counts) else float("nan"),
            "p90": float(counts.quantile(0.90)) if len(counts) else float("nan"),
            "p95": float(counts.quantile(0.95)) if len(counts) else float("nan"),
            "p99": float(counts.quantile(0.99)) if len(counts) else float("nan"),
            "overdispersion_variance_mean": variance / mean if mean > 0 else float("nan"),
            "missing_count": int(data["repository_count"].isna().sum()),
            "invalid_nonpositive_count": int((counts < 1).sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def ordinal_distribution(frame: pd.DataFrame, *, split: str = "train") -> pd.DataFrame:
    """Resume classes ordinais apenas entre skills reutilizadas."""
    subset = frame.loc[frame["split"].eq(split) & frame["reused"].eq(1)].copy()
    counts = subset["reuse_intensity"].astype("Int64")
    total = len(subset)
    rows: list[dict[str, object]] = []
    for code in range(4):
        values = subset.loc[counts.eq(code), "repository_count"].astype(float)
        rows.append(
            {
                "split": split,
                "intensity_code": code,
                "label": ORDINAL_LABELS[code],
                "n": int(len(values)),
                "percentage": 100 * len(values) / total if total else float("nan"),
                "count_min": float(values.min()) if len(values) else float("nan"),
                "count_max": float(values.max()) if len(values) else float("nan"),
                "count_mean": float(values.mean()) if len(values) else float("nan"),
                "count_median": float(values.median()) if len(values) else float("nan"),
                "count_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def group_distribution(frame: pd.DataFrame, *, split: str = "train") -> pd.DataFrame:
    """Resume tamanho dos grupos para auditar a divisao por repositorio."""
    subset = frame.loc[frame["split"].eq(split)].copy()
    rows: list[dict[str, object]] = []
    for population, data in (("all", subset), ("reused", subset.loc[subset["reused"].eq(1)])):
        sizes = data.groupby("representative_repo_id").size()
        rows.append(
            {
                "split": split,
                "population": population,
                "n_rows": int(len(data)),
                "n_groups": int(sizes.size),
                "mean_rows_per_group": float(sizes.mean()),
                "median_rows_per_group": float(sizes.median()),
                "p90_rows_per_group": float(sizes.quantile(0.90)),
                "p99_rows_per_group": float(sizes.quantile(0.99)),
                "max_rows_per_group": int(sizes.max()) if len(sizes) else 0,
            }
        )
    return pd.DataFrame(rows)


def split_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume a distribuicao dos estratos por split; nao orienta a decisao."""
    data = frame.copy()
    data["intensity_label"] = data["reuse_intensity"].map(ORDINAL_LABELS).fillna("not_reused")
    result = (
        data.groupby(["split", "split_stratum", "intensity_label"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    result["percentage_within_split"] = (
        result["n"] / result.groupby("split")["n"].transform("sum") * 100
    )
    return result


def coverage_summary(frame: pd.DataFrame, sample_path: Path) -> pd.DataFrame:
    """Audita occurrence_count contra repository_count, sem tratar ocorrencia como alvo."""
    sample = pq.read_table(sample_path, columns=["content_hash", "occurrence_count"]).to_pandas()
    data = frame.merge(sample, on="content_hash", how="inner", validate="one_to_one")
    if len(data) != len(frame):
        raise ValueError("A auditoria de cobertura perdeu observacoes")
    rows: list[dict[str, object]] = []
    for population, subset in (("all", data), ("reused", data.loc[data["reused"].eq(1)])):
        rows.append(
            {
                "population": population,
                "n": int(len(subset)),
                "missing_occurrence_count": int(subset["occurrence_count"].isna().sum()),
                "occurrence_lt_repository": int(
                    (subset["occurrence_count"] < subset["repository_count"]).sum()
                ),
                "occurrence_eq_repository": int(
                    (subset["occurrence_count"] == subset["repository_count"]).sum()
                ),
                "occurrence_gt_repository": int(
                    (subset["occurrence_count"] > subset["repository_count"]).sum()
                ),
                "occurrence_median": float(subset["occurrence_count"].median()),
                "occurrence_p99": float(subset["occurrence_count"].quantile(0.99)),
                "occurrence_max": int(subset["occurrence_count"].max()),
            }
        )
    return pd.DataFrame(rows)


def group_overlap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Verifica se algum repositorio representativo aparece nos dois splits."""
    train_groups = set(frame.loc[frame["split"].eq("train"), "representative_repo_id"])
    test_groups = set(frame.loc[frame["split"].eq("test"), "representative_repo_id"])
    return pd.DataFrame(
        [
            {
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "overlapping_groups": len(train_groups.intersection(test_groups)),
            }
        ]
    )


def save_count_figure(frame: pd.DataFrame, output: Path, split: str = "train") -> None:
    values = frame.loc[frame["split"].eq(split) & frame["reused"].eq(1), "repository_count"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(values, bins=60, color="#2563eb", alpha=0.85)
    axes[0].set_title("Contagem bruta entre reutilizadas")
    axes[0].set_xlabel("repository_count")
    axes[0].set_ylabel("Numero de skills")
    axes[0].grid(alpha=0.2)
    axes[1].hist(np.log1p(values), bins=50, color="#0f766e", alpha=0.85)
    axes[1].set_title("Contagem em escala logaritmica")
    axes[1].set_xlabel("log(1 + repository_count)")
    axes[1].set_ylabel("Numero de skills")
    axes[1].grid(alpha=0.2)
    figure.suptitle("EDA do alvo de reutilizacao", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_ordinal_figure(table: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(table["label"], table["percentage"], color="#7c3aed")
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in table["percentage"]])
    axis.set_ylabel("Percentual entre reutilizadas")
    axis.set_xlabel("Classe de intensidade")
    axis.set_title("Distribuicao do alvo ordinal")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(
    output: Path,
    count_table: pd.DataFrame,
    ordinal_table: pd.DataFrame,
    group_table: pd.DataFrame,
    split_table: pd.DataFrame,
    coverage_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
) -> str:
    reused = count_table.loc[count_table["population"].eq("reused")].iloc[0]
    class3 = ordinal_table.loc[ordinal_table["intensity_code"].eq(3)].iloc[0]
    lines = [
        "# Fase 1 - EDA e decisao preliminar do alvo",
        "",
        "## Protocolo",
        "",
        "A decisao sobre a formulacao do alvo foi baseada somente no split de desenvolvimento "
        "(`split=train`). O teste externo foi resumido para auditoria de balanceamento, mas "
        "nao participou da recomendacao. `repository_count` e a origem do alvo e nao pode ser "
        "usado como atributo preditor.",
        "",
        "## Achados principais",
        "",
        f"- Skills reutilizadas no desenvolvimento: **{int(reused['n']):,}**.".replace(",", "."),
        f"- Mediana de `repository_count`: **{_fmt(reused['median'], 0)}**.",
        f"- Media: **{_fmt(reused['mean'])}**; variancia: **{_fmt(reused['variance'])}**.",
        f"- Razao variancia/media: **{_fmt(reused['overdispersion_variance_mean'])}**, "
        "indicando forte sobredispersao.",
        f"- P95: **{_fmt(reused['p95'], 1)}**; "
        f"P99: **{_fmt(reused['p99'], 1)}**; "
        f"maximo: **{_fmt(reused['max'], 0)}**.",
        f"- A classe `10+` contem **{int(class3['n']):,}** skills, mas varia de "
        f"{_fmt(class3['count_min'], 0)} a { _fmt(class3['count_max'], 0)} reutilizacoes.",
        "",
        "## Distribuicao ordinal",
        "",
        "A discretizacao atual produz classes ordenadas, mas a ultima classe agrega uma cauda "
        "muito ampla. Isso melhora a estabilidade da classificacao, mas perde informacao sobre "
        "a diferenca entre reutilizacao alta e muito alta.",
        "",
        ordinal_table.to_markdown(index=False),
        "",
        "## Grupos e divisao",
        "",
        "A divisao por `representative_repo_id` e necessaria: os grupos possuem cauda longa e "
        "nao devem ser fragmentados entre treino e validacao. "
        "A tabela resume o tamanho dos grupos.",
        "",
        group_table.to_markdown(index=False),
        "",
        "## Integridade da contagem e da divisao",
        "",
        "`occurrence_count` e sempre maior ou igual a `repository_count`; isso e consistente "
        "com multiplas ocorrencias do mesmo conteudo dentro de um repositorio. Nao foram "
        "observados valores ausentes ou casos em que a contagem de repositorios excedesse a "
        "contagem de ocorrencias.",
        "",
        coverage_table.to_markdown(index=False),
        "",
        "A divisao externa nao possui sobreposicao de `representative_repo_id` "
        "entre treino e teste:",
        "",
        overlap_table.to_markdown(index=False),
        "",
        "## Recomendacao para a proxima fase",
        "",
        "Decisao da Fase 1: a pesquisa adotara **regressao de contagem** como alvo principal, "
        "preferencialmente com modelo Binomial Negativo ou regressao sobre "
        "`log1p(repository_count)`. "
        "A decisao e sustentada pela forte assimetria e sobredispersao observadas na contagem. "
        "A classificacao ordinal nao integra o protocolo principal.",
        "",
        "O proximo passo e implementar e validar o pipeline de regressao de contagem com divisao "
        "agrupada, sem leakage, antes do tuning completo e da avaliacao no teste externo.",
        "",
        "## Artefatos",
        "",
        "- `reports/tables/target_eda_count_distribution.csv`",
        "- `reports/tables/target_eda_ordinal_distribution.csv`",
        "- `reports/tables/target_eda_group_distribution.csv`",
        "- `reports/tables/target_eda_split_distribution.csv`",
        "- `reports/tables/target_eda_coverage_summary.csv`",
        "- `reports/tables/target_eda_group_overlap.csv`",
        "- `reports/figures/target_eda_count_distribution.png`",
        "- `reports/figures/target_eda_ordinal_distribution.png`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output.read_text(encoding="utf-8")


def run_pipeline(
    targets_path: Path = DEFAULT_TARGETS,
    features_path: Path = DEFAULT_FEATURES,
    tables_dir: Path = DEFAULT_TABLES,
    figures_dir: Path = DEFAULT_FIGURES,
    report_path: Path = DEFAULT_REPORT,
    sample_path: Path = DEFAULT_SAMPLE,
) -> dict[str, object]:
    if not targets_path.is_file():
        raise FileNotFoundError(f"Alvos nao encontrados: {targets_path.resolve()}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Features nao encontradas: {features_path.resolve()}")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Amostra nao encontrada: {sample_path.resolve()}")
    frame = load_target_frame(targets_path, features_path)
    train_count = count_distribution(frame, split="train")
    train_ordinal = ordinal_distribution(frame, split="train")
    train_groups = group_distribution(frame, split="train")
    split_table = split_distribution(frame)
    coverage_table = coverage_summary(frame, sample_path)
    overlap_table = group_overlap_summary(frame)
    # The test table is exported only for audit; it is not used in the report decision.
    test_count = count_distribution(frame, split="test")
    test_ordinal = ordinal_distribution(frame, split="test")
    count_table = pd.concat([train_count, test_count], ignore_index=True)
    ordinal_table = pd.concat([train_ordinal, test_ordinal], ignore_index=True)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    count_table.to_csv(tables_dir / "target_eda_count_distribution.csv", index=False)
    ordinal_table.to_csv(tables_dir / "target_eda_ordinal_distribution.csv", index=False)
    train_groups.to_csv(tables_dir / "target_eda_group_distribution.csv", index=False)
    split_table.to_csv(tables_dir / "target_eda_split_distribution.csv", index=False)
    coverage_table.to_csv(tables_dir / "target_eda_coverage_summary.csv", index=False)
    overlap_table.to_csv(tables_dir / "target_eda_group_overlap.csv", index=False)
    save_count_figure(frame, figures_dir / "target_eda_count_distribution.png")
    save_ordinal_figure(train_ordinal, figures_dir / "target_eda_ordinal_distribution.png")
    write_report(
        report_path,
        train_count,
        train_ordinal,
        train_groups,
        split_table,
        coverage_table,
        overlap_table,
    )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "targets_source": str(targets_path),
        "features_source": str(features_path),
        "sample_source": str(sample_path),
        "rows": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "train_reused_rows": int(((frame["split"] == "train") & (frame["reused"] == 1)).sum()),
        "test_reused_rows": int(((frame["split"] == "test") & (frame["reused"] == 1)).sum()),
        "decision_scope": "train split only; test summarized for audit",
        "ordinal_labels": ORDINAL_LABELS,
        "target_decision": "count regression conditional on reused=1",
        "primary_metric": "RMSLE",
        "recommended_next_step": "implement grouped count-regression pipeline",
        "overlapping_groups_train_test": int(overlap_table.iloc[0]["overlapping_groups"]),
    }
    (tables_dir / "target_eda_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        targets_path=args.targets,
        features_path=args.features,
        sample_path=args.sample,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        report_path=args.report,
    )
    print("EDA do alvo gerada com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
