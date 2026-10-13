"""Constroi a primeira tabela analitica a partir da amostra oficial do GitSkills."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_DATABASE = Path("data/raw/gitskills-sample/agent_skills_sample.db")
DEFAULT_OUTPUT = Path("data/processed/skills_target.parquet")
DEFAULT_TABLES = Path("reports/tables")
DEFAULT_FIGURES = Path("reports/figures")


def connect_read_only(database: Path) -> sqlite3.Connection:
    """Abre o SQLite em modo somente leitura para preservar o dado bruto."""
    resolved = database.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Banco da amostra nao encontrado: {resolved}")
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def build_analytical_table(
    connection: sqlite3.Connection, reuse_threshold: int = 2
) -> pd.DataFrame:
    """Retorna uma linha por hash, com alvo baseado em repositorios distintos."""
    if reuse_threshold < 2:
        raise ValueError("reuse_threshold deve ser pelo menos 2")

    query = """
        WITH reuse_counts AS (
            SELECT
                file_sha AS content_hash,
                COUNT(*) AS occurrence_count,
                COUNT(DISTINCT repo_full_name) AS repository_count
            FROM artifacts
            GROUP BY file_sha
        )
        SELECT
            counts.content_hash,
            counts.occurrence_count,
            counts.repository_count,
            CASE WHEN counts.repository_count >= :reuse_threshold THEN 1 ELSE 0 END AS reused,
            representative.repo_full_name AS representative_repo_id,
            representative.path AS representative_path,
            representative.filename,
            representative.location_class,
            representative.content,
            representative.frontmatter_valid,
            representative.name,
            representative.description,
            representative.body_chars,
            representative.sibling_count,
            representative.sibling_bytes,
            representative.has_scripts,
            representative.has_references,
            representative.composition_fetched,
            representative.composition_truncated,
            representative.history_fetched,
            representative.first_commit_at,
            representative.last_commit_at,
            representative.commit_count,
            repos.owner AS repo_owner,
            repos.stars AS repo_stars,
            repos.forks AS repo_forks,
            repos.is_fork AS repo_is_fork,
            repos.language AS repo_language,
            repos.license AS repo_license,
            repos.created_at AS repo_created_at,
            repos.pushed_at AS repo_pushed_at,
            repos.metadata_fetched AS repo_metadata_fetched
        FROM reuse_counts AS counts
        JOIN artifacts AS representative
          ON representative.file_sha = counts.content_hash
         AND representative.dedup_primary = 1
        LEFT JOIN repos
          ON repos.full_name = representative.repo_full_name
        ORDER BY counts.content_hash
    """
    return pd.read_sql_query(
        query,
        connection,
        params={"reuse_threshold": reuse_threshold},
    )


def build_quality_report(
    connection: sqlite3.Connection, analytical_table: pd.DataFrame
) -> pd.DataFrame:
    """Calcula invariantes essenciais e cobertura da tabela analitica."""
    scalar_queries = {
        "artifact_rows": "SELECT COUNT(*) FROM artifacts",
        "distinct_hashes": "SELECT COUNT(DISTINCT file_sha) FROM artifacts",
        "representatives": "SELECT SUM(dedup_primary) FROM artifacts",
        "distinct_repositories": "SELECT COUNT(DISTINCT repo_full_name) FROM artifacts",
        "hashes_without_exactly_one_representative": """
            SELECT COUNT(*) FROM (
                SELECT file_sha
                FROM artifacts
                GROUP BY file_sha
                HAVING SUM(dedup_primary) != 1
            )
        """,
        "representatives_without_content": """
            SELECT COUNT(*) FROM artifacts
            WHERE dedup_primary = 1 AND content IS NULL
        """,
        "duplicate_hash_repository_groups": """
            SELECT COUNT(*) FROM (
                SELECT file_sha, repo_full_name
                FROM artifacts
                GROUP BY file_sha, repo_full_name
                HAVING COUNT(*) > 1
            )
        """,
    }
    metrics: list[dict[str, int | float | str]] = []
    for metric, query in scalar_queries.items():
        value = connection.execute(query).fetchone()[0]
        metrics.append({"metric": metric, "value": value})

    reused = int(analytical_table["reused"].sum())
    rows = len(analytical_table)
    derived = {
        "analytical_rows": rows,
        "reused_contents": reused,
        "singleton_contents": rows - reused,
        "reused_percentage": round(100 * reused / rows, 4),
        "maximum_repository_count": int(analytical_table["repository_count"].max()),
        "mean_repository_count": round(float(analytical_table["repository_count"].mean()), 4),
        "missing_representative_repository_metadata": int(
            analytical_table["repo_metadata_fetched"].isna().sum()
        ),
    }
    metrics.extend({"metric": metric, "value": value} for metric, value in derived.items())
    return pd.DataFrame(metrics)


def build_reuse_distribution(analytical_table: pd.DataFrame) -> pd.DataFrame:
    """Agrega o numero de conteudos por quantidade de repositorios."""
    return (
        analytical_table.groupby("repository_count", as_index=False)
        .size()
        .rename(columns={"size": "content_count"})
        .sort_values("repository_count")
    )


def sha256_file(path: Path) -> str:
    """Calcula um checksum reproduzivel do arquivo de origem."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_distribution_figure(distribution: pd.DataFrame, output: Path) -> None:
    """Salva a distribuicao de reutilizacao com eixo vertical logaritmico."""
    visible = distribution.copy()
    visible["repository_bin"] = visible["repository_count"].clip(upper=20).astype(str)
    visible.loc[visible["repository_count"] > 20, "repository_bin"] = "21+"
    visible = visible.groupby("repository_bin", as_index=False, sort=False)["content_count"].sum()

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(visible["repository_bin"], visible["content_count"], color="#2563eb")
    axis.set_yscale("log")
    axis.set_xlabel("Numero de repositorios distintos")
    axis.set_ylabel("Conteudos distintos (escala log)")
    axis.set_title("Distribuicao da reutilizacao na amostra GitSkills")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_pipeline(
    database: Path,
    output: Path,
    tables_dir: Path,
    figures_dir: Path,
    reuse_threshold: int,
) -> dict[str, int | float | str]:
    """Executa a construcao do alvo, os relatorios e o manifesto."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with connect_read_only(database) as connection:
        analytical = build_analytical_table(connection, reuse_threshold)
        quality = build_quality_report(connection, analytical)

    expected_hashes = int(quality.loc[quality["metric"] == "distinct_hashes", "value"].iloc[0])
    if len(analytical) != expected_hashes:
        raise RuntimeError(
            f"Tabela analitica tem {len(analytical)} linhas; esperado: {expected_hashes}"
        )
    if analytical["content_hash"].duplicated().any():
        raise RuntimeError("content_hash duplicado na tabela analitica")

    distribution = build_reuse_distribution(analytical)
    analytical.to_parquet(output, index=False)
    quality.to_csv(tables_dir / "data_quality.csv", index=False)
    distribution.to_csv(tables_dir / "reuse_distribution.csv", index=False)
    save_distribution_figure(distribution, figures_dir / "reuse_distribution.png")

    manifest: dict[str, int | float | str] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_database": str(database),
        "source_sha256": sha256_file(database),
        "reuse_threshold": reuse_threshold,
        "analytical_rows": len(analytical),
        "reused_contents": int(analytical["reused"].sum()),
        "reused_percentage": round(100 * float(analytical["reused"].mean()), 4),
    }
    (tables_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--reuse-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        database=args.database,
        output=args.output,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        reuse_threshold=args.reuse_threshold,
    )
    print("Tabela analitica e relatorios gerados com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
