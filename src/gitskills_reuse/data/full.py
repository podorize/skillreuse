"""Constroi estatisticas populacionais e a amostra final de modelagem."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_ARTIFACTS = Path("data/raw/gitskills-full/artifacts/*.parquet")
DEFAULT_REPOS = Path("data/raw/gitskills-full/repos/*.parquet")
DEFAULT_INDEX = Path("data/interim/full_population_index.parquet")
DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")
DEFAULT_TABLES = Path("reports/tables")
DEFAULT_FIGURES = Path("reports/figures")
DEFAULT_TEMP = Path("work/duckdb-tmp")
DATASET_REVISION = "ebab17454a7c236f8f26b183567f1a126f42e3f8"


def sql_path(path: Path) -> str:
    """Escapa um caminho para uso como literal SQL do DuckDB."""
    return str(path).replace("'", "''")


def connect(temp_dir: Path) -> duckdb.DuckDBPyConnection:
    """Cria uma conexao local com area temporaria explicita."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order = false")
    return connection


def build_population_index(
    connection: duckdb.DuckDBPyConnection,
    artifacts_glob: Path,
    output: Path,
    reuse_threshold: int,
) -> None:
    """Materializa uma linha por conteudo, sem carregar os textos em memoria."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT
                file_sha AS content_hash,
                COUNT(*) AS occurrence_count,
                COUNT(DISTINCT repo_full_name) AS repository_count,
                CASE
                    WHEN COUNT(DISTINCT repo_full_name) >= {int(reuse_threshold)} THEN 1
                    ELSE 0
                END AS reused
            FROM read_parquet('{sql_path(artifacts_glob)}')
            GROUP BY file_sha
            ORDER BY file_sha
        ) TO '{sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    temporary.replace(output)


def select_deterministic_hashes(
    connection: duckdb.DuckDBPyConnection,
    population_index: Path,
    sample_size: int,
    sampling_seed: int = 20261009,
) -> pd.DataFrame:
    """Seleciona hashes por uma segunda dispersao criptografica com semente fixa."""
    return connection.execute(
        f"""
        SELECT content_hash, reused
        FROM read_parquet('{sql_path(population_index)}')
        ORDER BY md5(content_hash || ':{int(sampling_seed)}'), content_hash
        LIMIT {int(sample_size)}
        """
    ).fetchdf()


def build_modeling_sample(
    connection: duckdb.DuckDBPyConnection,
    artifacts_glob: Path,
    repos_glob: Path,
    population_index: Path,
    output: Path,
    sample_size: int,
    sampling_seed: int,
) -> None:
    """Junta os conteudos selecionados a uma ocorrencia representativa e seu repositorio."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"""
        COPY (
            WITH selected AS (
                SELECT *
                FROM read_parquet('{sql_path(population_index)}')
                ORDER BY md5(content_hash || ':{int(sampling_seed)}'), content_hash
                LIMIT {int(sample_size)}
            )
            SELECT
                selected.content_hash,
                selected.occurrence_count,
                selected.repository_count,
                selected.reused,
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
            FROM selected
            JOIN read_parquet('{sql_path(artifacts_glob)}') AS representative
              ON representative.file_sha = selected.content_hash
             AND representative.dedup_primary = 1
            LEFT JOIN read_parquet('{sql_path(repos_glob)}') AS repos
              ON repos.full_name = representative.repo_full_name
            ORDER BY selected.content_hash
        ) TO '{sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    temporary.replace(output)


def build_quality_report(
    connection: duckdb.DuckDBPyConnection,
    artifacts_glob: Path,
    population_index: Path,
    modeling_sample: Path,
) -> pd.DataFrame:
    """Calcula invariantes e cobertura das duas camadas de dados."""
    artifacts = sql_path(artifacts_glob)
    index = sql_path(population_index)
    sample = sql_path(modeling_sample)
    queries = {
        "artifact_occurrences": f"SELECT COUNT(*) FROM read_parquet('{artifacts}')",
        "distinct_repositories": (
            f"SELECT COUNT(DISTINCT repo_full_name) FROM read_parquet('{artifacts}')"
        ),
        "population_contents": f"SELECT COUNT(*) FROM read_parquet('{index}')",
        "population_reused_contents": (
            f"SELECT COALESCE(SUM(reused), 0) FROM read_parquet('{index}')"
        ),
        "population_maximum_repository_count": (
            f"SELECT MAX(repository_count) FROM read_parquet('{index}')"
        ),
        "representatives": (
            f"SELECT COALESCE(SUM(dedup_primary), 0) FROM read_parquet('{artifacts}')"
        ),
        "hashes_without_exactly_one_representative": f"""
            SELECT COUNT(*) FROM (
                SELECT file_sha
                FROM read_parquet('{artifacts}')
                GROUP BY file_sha
                HAVING SUM(dedup_primary) != 1
            )
        """,
        "representatives_without_content": f"""
            SELECT COUNT(*) FROM read_parquet('{artifacts}')
            WHERE dedup_primary = 1 AND content IS NULL
        """,
        "duplicate_hash_repository_groups": f"""
            SELECT COUNT(*) FROM (
                SELECT file_sha, repo_full_name
                FROM read_parquet('{artifacts}')
                GROUP BY file_sha, repo_full_name
                HAVING COUNT(*) > 1
            )
        """,
        "modeling_sample_contents": f"SELECT COUNT(*) FROM read_parquet('{sample}')",
        "modeling_sample_reused_contents": (
            f"SELECT COALESCE(SUM(reused), 0) FROM read_parquet('{sample}')"
        ),
        "modeling_sample_distinct_hashes": (
            f"SELECT COUNT(DISTINCT content_hash) FROM read_parquet('{sample}')"
        ),
        "modeling_sample_missing_content": (
            f"SELECT COUNT(*) FROM read_parquet('{sample}') WHERE content IS NULL"
        ),
    }
    values = {name: connection.execute(query).fetchone()[0] for name, query in queries.items()}
    population_rows = int(values["population_contents"])
    population_positive = int(values["population_reused_contents"])
    sample_rows = int(values["modeling_sample_contents"])
    sample_positive = int(values["modeling_sample_reused_contents"])
    values["population_reused_percentage"] = round(
        100 * population_positive / population_rows, 6
    )
    values["modeling_sample_reused_percentage"] = round(
        100 * sample_positive / sample_rows, 6
    )
    values["sample_population_rate_difference_percentage_points"] = round(
        values["modeling_sample_reused_percentage"]
        - values["population_reused_percentage"],
        6,
    )
    return pd.DataFrame([{"metric": name, "value": value} for name, value in values.items()])


def build_reuse_distribution(
    connection: duckdb.DuckDBPyConnection, population_index: Path
) -> pd.DataFrame:
    """Agrega a distribuicao populacional do numero de repositorios por conteudo."""
    return connection.execute(
        f"""
        SELECT repository_count, COUNT(*) AS content_count
        FROM read_parquet('{sql_path(population_index)}')
        GROUP BY repository_count
        ORDER BY repository_count
        """
    ).fetchdf()


def save_distribution_figure(distribution: pd.DataFrame, output: Path) -> None:
    """Salva a distribuicao populacional com cauda agregada."""
    visible = distribution.copy()
    visible["repository_bin"] = visible["repository_count"].clip(upper=20).astype(str)
    visible.loc[visible["repository_count"] > 20, "repository_bin"] = "21+"
    visible = visible.groupby("repository_bin", as_index=False, sort=False)[
        "content_count"
    ].sum()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(visible["repository_bin"], visible["content_count"], color="#2563eb")
    axis.set_yscale("log")
    axis.set_xlabel("Numero de repositorios distintos")
    axis.set_ylabel("Conteudos distintos (escala log)")
    axis.set_title("Distribuicao populacional da reutilizacao no GitSkills")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(quality: pd.DataFrame, name: str) -> int | float:
    return quality.loc[quality["metric"] == name, "value"].iloc[0]


def run_pipeline(
    artifacts_glob: Path,
    repos_glob: Path,
    population_index: Path,
    modeling_sample: Path,
    tables_dir: Path,
    figures_dir: Path,
    temp_dir: Path,
    reuse_threshold: int,
    sample_size: int,
    sampling_seed: int,
) -> dict[str, int | float | str]:
    """Executa a camada populacional e produz a amostra final auditavel."""
    if reuse_threshold < 2:
        raise ValueError("reuse_threshold deve ser pelo menos 2")
    if sample_size < 1:
        raise ValueError("sample_size deve ser positivo")
    tables_dir.mkdir(parents=True, exist_ok=True)

    with connect(temp_dir) as connection:
        build_population_index(connection, artifacts_glob, population_index, reuse_threshold)
        population_rows = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_path(population_index)}')"
        ).fetchone()[0]
        if sample_size > population_rows:
            raise ValueError(
                f"sample_size={sample_size} excede a populacao de {population_rows} conteudos"
            )
        build_modeling_sample(
            connection,
            artifacts_glob,
            repos_glob,
            population_index,
            modeling_sample,
            sample_size,
            sampling_seed,
        )
        quality = build_quality_report(
            connection, artifacts_glob, population_index, modeling_sample
        )
        distribution = build_reuse_distribution(connection, population_index)

    if int(metric(quality, "population_contents")) != int(metric(quality, "representatives")):
        raise RuntimeError("Numero de representantes difere do numero de conteudos")
    if int(metric(quality, "hashes_without_exactly_one_representative")) != 0:
        raise RuntimeError("Ha hashes sem exatamente um representante")
    if int(metric(quality, "modeling_sample_contents")) != sample_size:
        raise RuntimeError("A amostra final nao possui o tamanho solicitado")
    if int(metric(quality, "modeling_sample_distinct_hashes")) != sample_size:
        raise RuntimeError("A amostra final contem hashes duplicados")

    quality.to_csv(tables_dir / "full_data_quality.csv", index=False)
    distribution.to_csv(tables_dir / "full_reuse_distribution.csv", index=False)
    save_distribution_figure(
        distribution, figures_dir / "full_reuse_distribution.png"
    )

    manifest: dict[str, int | float | str] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_revision": DATASET_REVISION,
        "artifacts_source": str(artifacts_glob),
        "repos_source": str(repos_glob),
        "reuse_threshold": reuse_threshold,
        "sampling_method": "ascending md5(content_hash || ':' || sampling_seed)",
        "sampling_seed": sampling_seed,
        "requested_sample_size": sample_size,
        "population_contents": int(metric(quality, "population_contents")),
        "population_reused_contents": int(metric(quality, "population_reused_contents")),
        "population_reused_percentage": float(
            metric(quality, "population_reused_percentage")
        ),
        "modeling_sample_contents": int(metric(quality, "modeling_sample_contents")),
        "modeling_sample_reused_contents": int(
            metric(quality, "modeling_sample_reused_contents")
        ),
        "modeling_sample_reused_percentage": float(
            metric(quality, "modeling_sample_reused_percentage")
        ),
        "sample_population_rate_difference_percentage_points": float(
            metric(quality, "sample_population_rate_difference_percentage_points")
        ),
        "population_index_sha256": sha256_file(population_index),
        "modeling_sample_sha256": sha256_file(modeling_sample),
    }
    (tables_dir / "full_population_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--population-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--modeling-sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--reuse-threshold", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--sampling-seed", type=int, default=20261009)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        artifacts_glob=args.artifacts,
        repos_glob=args.repos,
        population_index=args.population_index,
        modeling_sample=args.modeling_sample,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        temp_dir=args.temp_dir,
        reuse_threshold=args.reuse_threshold,
        sample_size=args.sample_size,
        sampling_seed=args.sampling_seed,
    )
    print("Populacao e amostra final geradas com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
