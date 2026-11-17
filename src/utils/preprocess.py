
import json
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import duckdb
from datetime import UTC, datetime

from utils.download import DATASET_REVISION, sha256_file
from utils.misc import save_manifest

def sql_path(path: Path) -> str:
    """Escapa um caminho para uso como literal SQL do DuckDB."""
    return str(path).replace("'", "''")

def connect(temp_dir: Path) -> duckdb.DuckDBPyConnection:
    """Cria uma conexao local com area temporaria explicita."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute("SET threads = 2")
    connection.execute("SET preserve_insertion_order = false")
    return connection

def copy_query(connection: duckdb.DuckDBPyConnection, query: str, output: Path) -> None:
    """Executa uma consulta e grava o resultado em Parquet de forma atomica."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    connection.execute(
        f"""
        COPY ({query})
        TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    temporary.replace(output)

def build_dataset(connection: duckdb.DuckDBPyConnection,
    artifacts_glob: Path,
    repos_glob: Path,
    output: Path,
) -> None:
    """Gera uma linha por hash, com agregacoes e uma ocorrencia representativa."""
    artifacts = sql_path(artifacts_glob)
    repos = sql_path(repos_glob)

    output.parent.mkdir(parents=True, exist_ok=True)

    population_temp = output.with_name(
        f"{output.stem}_population_temp.parquet"
    )

    try:
        # 1. Agrega todos os conteúdos.
        population_query = f"""
            SELECT
                file_sha AS content_hash,
                COUNT(*) AS occurrence_count,
                COUNT(DISTINCT repo_full_name) AS repository_count
            FROM read_parquet('{artifacts}')
            GROUP BY file_sha
        """

        copy_query(
            connection,
            population_query,
            population_temp,
        )

        # 2. Associa cada conteúdo à ocorrência representativa
        # e aos metadados do repositório.
        dataset_query = f"""
            SELECT
                population.content_hash,
                population.occurrence_count,
                population.repository_count,

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

            FROM read_parquet('{sql_path(population_temp)}') AS population

            LEFT JOIN read_parquet('{artifacts}') AS representative
                ON representative.file_sha = population.content_hash
                AND representative.dedup_primary = 1

            LEFT JOIN read_parquet('{repos}') AS repos
                ON repos.full_name = representative.repo_full_name
        """

        copy_query(
            connection,
            dataset_query,
            output,
        )

    finally:
        population_temp.unlink(missing_ok=True)

def build_modeling_sample(
    connection: duckdb.DuckDBPyConnection,
    full_dataset: Path,
    output: Path,
    sample_size: int,
    sampling_seed: int = 20261009,
) -> None:
    """Seleciona uma amostra deterministica a partir do dataset completo."""
    if sample_size < 1:
        raise ValueError("sample_size deve ser positivo")

    full_data = sql_path(full_dataset)
    query = f"""
        SELECT *
        FROM read_parquet('{full_data}')
        ORDER BY md5(content_hash || ':{int(sampling_seed)}'), content_hash
        LIMIT {int(sample_size)}
    """

    population_rows = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet('{full_data}')"
    ).fetchone()[0]

    if sample_size > population_rows:
        raise ValueError(
            f"sample_size={sample_size} excede a populacao de "
            f"{population_rows} conteudos"
        )

    copy_query(connection, query, output)


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
    sample_size: int,
    sampling_seed: int,
) -> dict[str, int | float | str]:
    """Executa a camada populacional e produz a amostra final auditavel."""
    if sample_size < 1:
        raise ValueError("sample_size deve ser positivo")
    # tables_dir.mkdir(parents=True, exist_ok=True)

    with connect(temp_dir) as connection:
        # Primeiro materializa todos os conteudos unicos.
        build_dataset(connection,
                      artifacts_glob,
                      repos_glob,
                      population_index)
        full_dataset_manifest = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "file_path": str(population_index)
        }
        print(f"Dataset salvo em {population_index.parent}")
        save_manifest(full_dataset_manifest, population_index.parent)

        build_modeling_sample(
            connection,
            population_index,
            modeling_sample,
            sample_size,
            sampling_seed,
        )

    #     quality = build_quality_report(
    #         connection, artifacts_glob, population_index, modeling_sample
    #     )
    #     distribution = build_reuse_distribution(connection, population_index)

    # if int(metric(quality, "population_contents")) != int(metric(quality, "representatives")):
    #     raise RuntimeError("Numero de representantes difere do numero de conteudos")
    # if int(metric(quality, "hashes_without_exactly_one_representative")) != 0:
    #     raise RuntimeError("Ha hashes sem exatamente um representante")
    # if int(metric(quality, "modeling_sample_contents")) != sample_size:
    #     raise RuntimeError("A amostra final nao possui o tamanho solicitado")
    # if int(metric(quality, "modeling_sample_distinct_hashes")) != sample_size:
    #     raise RuntimeError("A amostra final contem hashes duplicados")

    # quality.to_csv(tables_dir / "full_data_quality.csv", index=False)
    # distribution.to_csv(tables_dir / "full_reuse_distribution.csv", index=False)
    # save_distribution_figure(
    #     distribution, figures_dir / "full_reuse_distribution.png"
    # )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_revision": DATASET_REVISION,
        "artifacts_source": str(artifacts_glob),
        "repos_source": str(repos_glob),
        "file_path": str(modeling_sample),
        "sampling_method": "ascending md5(content_hash || ':' || sampling_seed)",
        "sampling_seed": sampling_seed,
        "requested_sample_size": sample_size,
        # "population_contents": int(metric(quality, "population_contents")),
        # "population_reused_contents": int(metric(quality, "population_reused_contents")),
        # "population_reused_percentage": float(
        #     metric(quality, "population_reused_percentage")
        # ),
        # "modeling_sample_contents": int(metric(quality, "modeling_sample_contents")),
        # "modeling_sample_reused_contents": int(
        #     metric(quality, "modeling_sample_reused_contents")
        # ),
        # "modeling_sample_reused_percentage": float(
        #     metric(quality, "modeling_sample_reused_percentage")
        # ),
        # "sample_population_rate_difference_percentage_points": float(
        #     metric(quality, "sample_population_rate_difference_percentage_points")
        # ),
        "population_index_sha256": sha256_file(population_index),
        "modeling_sample_sha256": sha256_file(modeling_sample),
    }

    save_manifest(manifest, modeling_sample.parent)

    return manifest
