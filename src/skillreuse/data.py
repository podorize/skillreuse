"""Download, agregacao do alvo e amostra deterministica do GitSkills."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# ==========================================================================
# Baixa e valida as tabelas essenciais do espelho Parquet do GitSkills.
# ==========================================================================

DATASET_ID = "mvaccargiu/gitskills"

REVISION = "ebab17454a7c236f8f26b183567f1a126f42e3f8"

DEFAULT_OUTPUT = Path("data/raw/gitskills-full")

DEFAULT_MANIFEST = Path("results/tables/full_download_manifest.json")

TABLES = ("artifacts", "repos")

RESERVE_BYTES = 8 * 1024**3

@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str

def api_url(table: str) -> str:
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{encoded_revision}/data/{table}?recursive=true&expand=true"
    )

def resolve_url(remote_path: str) -> str:
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )

def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "SkillReuse/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def parse_remote_files(payload: Any, table: str) -> list[RemoteFile]:
    """Converte a resposta da API em uma lista ordenada de Parquets verificaveis."""
    if not isinstance(payload, list):
        raise ValueError("Resposta inesperada da API do Hugging Face")
    prefix = f"data/{table}/"
    files = []
    for entry in payload:
        path = entry.get("path", "")
        lfs = entry.get("lfs") or {}
        is_expected_file = (
            entry.get("type") == "file"
            and path.startswith(prefix)
            and path.endswith(".parquet")
        )
        if not is_expected_file:
            continue
        size = int(entry.get("size", 0))
        digest = str(lfs.get("oid", ""))
        if size <= 0 or len(digest) != 64:
            raise ValueError(f"Metadados incompletos para {path}")
        files.append(RemoteFile(path=path, size=size, sha256=digest))
    if not files:
        raise ValueError(f"Nenhum Parquet encontrado para a tabela {table}")
    return sorted(files, key=lambda item: item.path)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_existing(path: Path, remote: RemoteFile) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != remote.size:
        raise RuntimeError(
            f"Arquivo final com tamanho incorreto: {path}. "
            "Remova-o manualmente antes de tentar novamente."
        )
    digest = sha256_file(path)
    if digest != remote.sha256:
        raise RuntimeError(
            f"Checksum incorreto no arquivo final: {path}. "
            "Remova-o manualmente antes de tentar novamente."
        )
    return True

def download_file(remote: RemoteFile, destination: Path) -> None:
    """Baixa para `.part`, retomando quando o servidor aceita HTTP Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(destination, remote):
        print(f"OK existente: {destination.name}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size > remote.size:
        raise RuntimeError(f"Download parcial maior que o esperado: {partial}")

    headers = {"User-Agent": "SkillReuse/0.1"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(resolve_url(remote.path), headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Falha HTTP ao baixar {remote.path}: {error.code}") from error

    status = getattr(response, "status", response.getcode())
    if current_size and status != 206:
        response.close()
        raise RuntimeError(
            f"Servidor nao aceitou retomada de {partial}. "
            "Remova o arquivo `.part` para reiniciar este item."
        )

    mode = "ab" if current_size else "wb"
    with response, partial.open(mode) as stream, tqdm(
        total=remote.size,
        initial=current_size,
        unit="B",
        unit_scale=True,
        desc=destination.name,
        file=sys.stdout,
    ) as progress:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            progress.update(len(chunk))

    if partial.stat().st_size != remote.size:
        raise RuntimeError(
            f"Download incompleto de {remote.path}: "
            f"{partial.stat().st_size} de {remote.size} bytes"
        )
    digest = sha256_file(partial)
    if digest != remote.sha256:
        raise RuntimeError(f"Checksum incorreto apos baixar {remote.path}")
    partial.replace(destination)

def collect_remote_files() -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for table in TABLES:
        files.extend(parse_remote_files(fetch_json(api_url(table)), table))
    return files

def ensure_disk_space(output: Path, files: list[RemoteFile]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for remote in files:
        local = output / Path(remote.path).relative_to("data")
        downloaded = local.stat().st_size if local.exists() else 0
        partial = local.with_suffix(local.suffix + ".part")
        downloaded += partial.stat().st_size if partial.exists() else 0
        missing_bytes += max(0, remote.size - downloaded)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes - missing_bytes < RESERVE_BYTES:
        raise RuntimeError(
            "Espaco insuficiente: o download preservaria menos de 8 GiB livres. "
            f"Livres={free_bytes / 1024**3:.2f} GiB, "
            f"faltam baixar={missing_bytes / 1024**3:.2f} GiB."
        )

def run_download(output: Path, manifest_path: Path) -> dict[str, Any]:
    files = collect_remote_files()
    ensure_disk_space(output, files)
    for remote in files:
        local = output / Path(remote.path).relative_to("data")
        download_file(remote, local)

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "tables": list(TABLES),
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
        "files": [asdict(item) for item in files],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest

def download_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()

def download_main() -> None:
    args = download_parse_args()
    manifest = run_download(args.output, args.manifest)
    print(
        f"Download concluido: {manifest['file_count']} arquivos, "
        f"{manifest['total_bytes'] / 1024**3:.2f} GiB."
    )


# ==========================================================================
# Constroi estatisticas populacionais e a amostra final de modelagem.
# ==========================================================================

matplotlib.use("Agg")

DEFAULT_ARTIFACTS = Path("data/raw/gitskills-full/artifacts/*.parquet")

DEFAULT_REPOS = Path("data/raw/gitskills-full/repos/*.parquet")

DEFAULT_INDEX = Path("data/interim/full_population_index.parquet")

DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")

DEFAULT_TABLES = Path("results/tables")

DEFAULT_FIGURES = Path("results/figures")

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

def build_parse_args() -> argparse.Namespace:
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

def build_main() -> None:
    args = build_parse_args()
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


# ==========================================================================
# Define os dois alvos e seleciona a divisao externa da amostra final.
# ==========================================================================

SOURCE_COLUMNS = ["content_hash", "representative_repo_id", "repository_count", "reused"]

INTENSITY_LABELS = {
    0: "2 repositorios",
    1: "3-4 repositorios",
    2: "5-9 repositorios",
    3: "10+ repositorios",
}

STRATA = ("not_reused", "reuse_2", "reuse_3_4", "reuse_5_9", "reuse_10_plus")

def build_targets(source: pd.DataFrame) -> pd.DataFrame:
    """Cria intensidade condicional, sem usar contagem de ocorrencias como substituto."""
    missing = set(SOURCE_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para os alvos: {sorted(missing)}")
    if source["content_hash"].isna().any() or source["content_hash"].duplicated().any():
        raise ValueError("content_hash deve ser unico e nao nulo")
    if source["representative_repo_id"].isna().any():
        raise ValueError("representative_repo_id nao pode ser nulo")

    counts = pd.to_numeric(source["repository_count"], errors="coerce")
    if counts.isna().any() or (counts < 1).any() or (counts != counts.round()).any():
        raise ValueError("repository_count deve ser inteiro positivo")
    reused = pd.to_numeric(source["reused"], errors="coerce")
    if reused.isna().any() or not reused.isin([0, 1]).all():
        raise ValueError("reused deve ser binario e nao nulo")
    if not reused.eq(counts.ge(2).astype(int)).all():
        raise ValueError("reused nao corresponde a repository_count >= 2")

    targets = source[SOURCE_COLUMNS].copy()
    targets["repository_count"] = counts.astype("int32")
    targets["reused"] = reused.astype("int8")
    intensity = pd.Series(pd.NA, index=targets.index, dtype="Int8")
    intensity.loc[counts.eq(2)] = 0
    intensity.loc[counts.between(3, 4)] = 1
    intensity.loc[counts.between(5, 9)] = 2
    intensity.loc[counts.ge(10)] = 3
    targets["reuse_intensity"] = intensity

    stratum = pd.Series("not_reused", index=targets.index, dtype="string")
    for code, name in enumerate(STRATA[1:]):
        stratum.loc[intensity.eq(code).fillna(False)] = name
    targets["split_stratum"] = stratum
    return targets

def _candidate_score(
    targets: pd.DataFrame,
    test_indices: np.ndarray,
    test_size: float,
    overall_positive_rate: float,
    overall_conditional_rates: np.ndarray,
) -> tuple[float, float, float, float]:
    test = targets.iloc[test_indices]
    size_gap = abs(len(test) / len(targets) - test_size)
    binary_gap = abs(float(test["reused"].mean()) - overall_positive_rate)
    positive_intensity = test.loc[test["reused"].eq(1), "reuse_intensity"]
    if positive_intensity.empty:
        raise ValueError("Candidato sem exemplos reutilizados")
    conditional_rates = (
        positive_intensity.value_counts(normalize=True)
        .reindex(range(4), fill_value=0)
        .to_numpy(dtype=float)
    )
    conditional_gap = float(np.max(np.abs(conditional_rates - overall_conditional_rates)))
    return (size_gap + binary_gap + conditional_gap, size_gap, binary_gap, conditional_gap)

def make_two_target_split(
    targets: pd.DataFrame,
    test_size: float = 0.20,
    random_seed: int = 20261009,
    restarts: int = 12,
) -> tuple[pd.Series, dict[str, object]]:
    """Seleciona candidato agrupado por regra fixa, sem consultar modelos ou resultados."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size deve estar entre 0 e 0.5")
    if restarts < 1:
        raise ValueError("restarts deve ser positivo")
    required = set(SOURCE_COLUMNS + ["reuse_intensity", "split_stratum"])
    if required - set(targets.columns):
        raise ValueError(f"Alvos incompletos: {sorted(required - set(targets.columns))}")
    if targets["content_hash"].duplicated().any():
        raise ValueError("content_hash duplicado nos alvos")
    if targets["representative_repo_id"].isna().any():
        raise ValueError("Grupos ausentes nos alvos")
    if set(targets["split_stratum"].dropna().unique()) != set(STRATA):
        raise ValueError("Todos os cinco estratos devem estar presentes")

    ordered = targets.sort_values("content_hash").reset_index(drop=True)
    positive_rate = float(ordered["reused"].mean())
    conditional_rates = (
        ordered.loc[ordered["reused"].eq(1), "reuse_intensity"]
        .value_counts(normalize=True)
        .reindex(range(4), fill_value=0)
        .to_numpy(dtype=float)
    )
    n_splits = max(2, round(1 / test_size))
    best: tuple[tuple[float, float, float, float, int, int], np.ndarray] | None = None
    for restart in range(restarts):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed + restart
        )
        for fold, (_, test_indices) in enumerate(
            splitter.split(
                ordered,
                y=ordered["split_stratum"],
                groups=ordered["representative_repo_id"],
            )
        ):
            score = _candidate_score(
                ordered, test_indices, test_size, positive_rate, conditional_rates
            )
            key = (*score, restart, fold)
            if best is None or key < best[0]:
                best = (key, test_indices)
    if best is None:
        raise RuntimeError("Nenhuma divisao candidata foi gerada")

    key, test_indices = best
    ordered_assignments = pd.Series("train", index=ordered.index, dtype="string")
    ordered_assignments.iloc[test_indices] = "test"
    split_by_hash = pd.Series(ordered_assignments.to_numpy(), index=ordered["content_hash"])
    assignments = targets["content_hash"].map(split_by_hash).astype("string")
    assignments.index = targets.index

    train_groups = set(targets.loc[assignments.eq("train"), "representative_repo_id"])
    test_groups = set(targets.loc[assignments.eq("test"), "representative_repo_id"])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Ha {len(overlap)} grupos compartilhados")
    summary = intensity_split_summary(targets.assign(split=assignments))
    if (summary["count"] <= 0).any():
        raise RuntimeError("Uma classe de intensidade ficou sem exemplos em treino ou teste")
    counts = {
        split: {row["split_stratum"]: int(row["count"]) for _, row in part.iterrows()}
        for split, part in summary.groupby("split")
    }
    signature = hashlib.sha256(
        targets.sort_values("content_hash")["content_hash"]
        .to_csv(index=False, header=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()
    manifest: dict[str, object] = {
        "strategy": "StratifiedGroupKFold over five strata; fixed-score candidate selection",
        "random_seed": random_seed,
        "restarts": restarts,
        "candidate_count": restarts * n_splits,
        "selected_restart": key[4],
        "selected_fold": key[5],
        "selection_score": round(key[0], 8),
        "score_components": {
            "absolute_test_fraction_gap": round(key[1], 8),
            "absolute_binary_prevalence_gap": round(key[2], 8),
            "maximum_conditional_intensity_gap": round(key[3], 8),
        },
        "requested_test_fraction": test_size,
        "actual_test_fraction": round(float(assignments.eq("test").mean()), 8),
        "train_rows": int(assignments.eq("train").sum()),
        "test_rows": int(assignments.eq("test").sum()),
        "overall_positive_rate": round(positive_rate, 8),
        "train_positive_rate": round(
            float(targets.loc[assignments.eq("train"), "reused"].mean()), 8
        ),
        "test_positive_rate": round(float(targets.loc[assignments.eq("test"), "reused"].mean()), 8),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": 0,
        "stratum_counts": counts,
        "sample_hash_signature_sha256": signature,
    }
    return assignments, manifest

def intensity_split_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Frequencias dos cinco estratos e taxas condicionais dos quatro positivos."""
    rows = []
    for split in ("train", "test"):
        part = assignments.loc[assignments["split"].eq(split)]
        positive_count = int(part["reused"].sum())
        counts = part["split_stratum"].value_counts()
        for stratum in STRATA:
            count = int(counts.get(stratum, 0))
            rows.append(
                {
                    "split": split,
                    "split_stratum": stratum,
                    "count": count,
                    "percentage_of_split": count / len(part) if len(part) else 0.0,
                    "percentage_among_reused": (
                        count / positive_count
                        if stratum != "not_reused" and positive_count
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)
