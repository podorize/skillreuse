"""Extrai atributos em lotes e congela a divisao final de treino e teste."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from gitskills_reuse.data.targets import (
    SOURCE_COLUMNS,
    build_targets,
    intensity_split_summary,
    make_two_target_split,
)
from gitskills_reuse.features.build import (
    add_engineered_features,
    build_feature_summary,
    save_class_figure,
    split_fingerprint,
)

DEFAULT_INPUT = Path("data/processed/final_modeling_sample.parquet")
DEFAULT_TARGETS = Path("data/processed/final_targets.parquet")
DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")
DEFAULT_SPLITS = Path("data/processed/final_split_assignments.parquet")
DEFAULT_TABLES = Path("reports/tables")
DEFAULT_FIGURES = Path("reports/figures")

IDENTIFIER_COLUMNS = ["content_hash", "representative_repo_id", "reused"]
EXTRACTION_COLUMNS = ["content", "name", "description"]
RAW_FEATURE_COLUMNS = [
    "filename",
    "location_class",
    "frontmatter_valid",
    "body_chars",
    "sibling_count",
    "sibling_bytes",
    "has_scripts",
    "has_references",
    "composition_fetched",
    "composition_truncated",
    "history_fetched",
    "first_commit_at",
    "last_commit_at",
    "commit_count",
    "repo_stars",
    "repo_forks",
    "repo_is_fork",
    "repo_language",
    "repo_license",
    "repo_created_at",
    "repo_pushed_at",
    "repo_metadata_fetched",
]
ENGINEERED_PREFIXES = ("text_", "markdown_", "frontmatter_")


def engineered_column_names(columns: list[str]) -> list[str]:
    """Seleciona somente atributos derivados, sem repetir campos brutos."""
    raw = set(IDENTIFIER_COLUMNS + EXTRACTION_COLUMNS + RAW_FEATURE_COLUMNS)
    return [
        column
        for column in columns
        if column.startswith(ENGINEERED_PREFIXES) and column not in raw
    ]


def read_split_frame(input_path: Path) -> pd.DataFrame:
    """Le somente os identificadores e a contagem usada para construir os alvos."""
    return pq.read_table(input_path, columns=SOURCE_COLUMNS).to_pandas()


def write_structured_features_batched(
    input_path: Path,
    output_path: Path,
    assignments: pd.DataFrame,
    batch_size: int = 2_000,
) -> list[str]:
    """Extrai atributos em lotes e nao duplica o texto integral na saida."""
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    split_lookup = assignments.set_index("content_hash")["split"]
    parquet = pq.ParquetFile(input_path)
    input_columns = IDENTIFIER_COLUMNS + EXTRACTION_COLUMNS + RAW_FEATURE_COLUMNS
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    engineered_columns: list[str] = []

    try:
        batches = parquet.iter_batches(batch_size=batch_size, columns=input_columns)
        progress = tqdm(total=parquet.metadata.num_rows, unit="conteudo")
        for batch in batches:
            frame = batch.to_pandas()
            featured = add_engineered_features(frame)
            if not engineered_columns:
                engineered_columns = engineered_column_names(featured.columns.tolist())
            output = featured[
                IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS + engineered_columns
            ].copy()
            output.insert(
                3,
                "split",
                output["content_hash"].map(split_lookup).astype("string"),
            )
            if output["split"].isna().any():
                raise RuntimeError("Ha conteudos sem atribuicao de treino ou teste")

            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                output_schema = table.schema.remove_metadata()
                writer = pq.ParquetWriter(temporary, output_schema, compression="zstd")
            if output_schema is None:
                raise RuntimeError("Esquema de saida nao inicializado")
            table = table.cast(output_schema)
            writer.write_table(table)
            progress.update(len(output))
        progress.close()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("Nenhum conteudo foi processado")
    temporary.replace(output_path)
    return engineered_columns


def validate_outputs(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    features_path: Path,
) -> dict[str, int]:
    """Confere cardinalidade, unicidade, atribuicoes e isolamento dos grupos."""
    metadata = pq.ParquetFile(features_path).metadata
    feature_keys = pq.read_table(
        features_path,
        columns=["content_hash", "representative_repo_id", "reused", "split"],
    ).to_pandas()
    train_groups = set(
        feature_keys.loc[feature_keys["split"] == "train", "representative_repo_id"]
    )
    test_groups = set(
        feature_keys.loc[feature_keys["split"] == "test", "representative_repo_id"]
    )
    checks = {
        "source_rows": len(source),
        "assignment_rows": len(assignments),
        "feature_rows": metadata.num_rows,
        "feature_columns": metadata.num_columns,
        "duplicate_source_hashes": int(source["content_hash"].duplicated().sum()),
        "duplicate_feature_hashes": int(feature_keys["content_hash"].duplicated().sum()),
        "missing_split_assignments": int(feature_keys["split"].isna().sum()),
        "group_overlap": len(train_groups & test_groups),
    }
    if len({checks["source_rows"], checks["assignment_rows"], checks["feature_rows"]}) != 1:
        raise RuntimeError("As tabelas finais possuem quantidades de linhas diferentes")
    if any(
        checks[name] != 0
        for name in (
            "duplicate_source_hashes",
            "duplicate_feature_hashes",
            "missing_split_assignments",
            "group_overlap",
        )
    ):
        raise RuntimeError(f"Falha nas invariantes finais: {checks}")
    return checks


def run_pipeline(
    input_path: Path,
    targets_path: Path,
    features_path: Path,
    splits_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    test_size: float,
    random_seed: int,
    batch_size: int,
    split_restarts: int,
) -> dict[str, object]:
    """Gera os dois alvos, a divisao externa e os atributos auditaveis."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Amostra final nao encontrada: {input_path.resolve()}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.parent.mkdir(parents=True, exist_ok=True)

    source = read_split_frame(input_path)
    targets = build_targets(source)
    split_values, split_manifest = make_two_target_split(
        targets, test_size, random_seed, restarts=split_restarts
    )
    target_temporary = targets_path.with_suffix(targets_path.suffix + ".tmp")
    targets.to_parquet(target_temporary, index=False)
    target_temporary.replace(targets_path)

    assignments = targets.drop(columns=["repository_count"]).copy()
    assignments["split"] = split_values
    split_temporary = splits_path.with_suffix(splits_path.suffix + ".tmp")
    assignments.to_parquet(split_temporary, index=False)
    split_temporary.replace(splits_path)

    engineered_columns = write_structured_features_batched(
        input_path, features_path, assignments, batch_size
    )
    checks = validate_outputs(source, assignments, features_path)

    numeric_columns = engineered_columns + [
        "frontmatter_valid",
        "body_chars",
        "sibling_count",
        "sibling_bytes",
        "has_scripts",
        "has_references",
        "composition_fetched",
        "composition_truncated",
        "history_fetched",
        "commit_count",
        "repo_stars",
        "repo_forks",
        "repo_is_fork",
        "repo_metadata_fetched",
    ]
    numeric = pq.read_table(features_path, columns=numeric_columns).to_pandas()
    build_feature_summary(numeric, numeric_columns).to_csv(
        tables_dir / "final_feature_summary.csv", index=False
    )

    split_summary = (
        assignments.groupby("split", as_index=False)
        .agg(
            rows=("content_hash", "size"),
            positives=("reused", "sum"),
            groups=("representative_repo_id", "nunique"),
        )
        .sort_values("split")
    )
    split_summary["negatives"] = split_summary["rows"] - split_summary["positives"]
    split_summary["positive_rate"] = split_summary["positives"] / split_summary["rows"]
    split_summary.to_csv(tables_dir / "final_split_summary.csv", index=False)
    intensity_split_summary(assignments).to_csv(
        tables_dir / "final_intensity_split_summary.csv", index=False
    )
    pd.DataFrame([checks]).to_csv(tables_dir / "final_feature_quality.csv", index=False)
    save_class_figure(
        assignments,
        figures_dir / "final_class_distribution.png",
        title="Distribuicao do alvo na amostra final GitSkills",
    )

    manifest: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_table": str(input_path),
        "target_table": str(targets_path),
        "feature_table": str(features_path),
        "split_table": str(splits_path),
        "intensity_split_summary_table": str(tables_dir / "final_intensity_split_summary.csv"),
        "batch_size": batch_size,
        "raw_feature_count": len(RAW_FEATURE_COLUMNS),
        "engineered_feature_count": len(engineered_columns),
        **split_manifest,
        **checks,
        "split_fingerprint_sha256": split_fingerprint(assignments),
    }
    (tables_dir / "final_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=20261009)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--split-restarts", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        input_path=args.input,
        targets_path=args.targets,
        features_path=args.features,
        splits_path=args.splits,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        test_size=args.test_size,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        split_restarts=args.split_restarts,
    )
    print("Dois alvos, divisao final e atributos gerados com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
