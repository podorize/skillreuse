"""Atributos estruturais e divisao externa treino/teste."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from skillreuse.data import (
    SOURCE_COLUMNS,
    build_targets,
    intensity_split_summary,
    make_two_target_split,
)

# ==========================================================================
# Extrai atributos iniciais e congela uma divisao agrupada de treino e teste.
# ==========================================================================

matplotlib.use("Agg")

WORD_PATTERN = re.compile(r"\b[^\W_]+(?:[-'][^\W_]+)*\b", flags=re.UNICODE)

SENTENCE_PATTERN = re.compile(r"[.!?]+(?:\s|$)")

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+\S", flags=re.MULTILINE)

UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-+*]\s+\S", flags=re.MULTILINE)

ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s+\S", flags=re.MULTILINE)

FENCE_PATTERN = re.compile(r"^\s*(```|~~~)", flags=re.MULTILINE)

INLINE_CODE_PATTERN = re.compile(r"(?<!`)`[^`\n]+`(?!`)")

MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\([^)]+\)")

URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?", flags=re.MULTILINE)

CHECKBOX_PATTERN = re.compile(r"^\s*[-+*]\s+\[[ xX]\]\s+", flags=re.MULTILINE)

TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$", flags=re.MULTILINE)

SHELL_PROMPT_PATTERN = re.compile(r"^\s*\$\s+\S", flags=re.MULTILINE)

def extract_markdown_features(value: object) -> dict[str, int | float]:
    """Extrai medidas simples e auditaveis de texto e estrutura Markdown."""
    text = value if isinstance(value, str) else ""
    words = WORD_PATTERN.findall(text.lower())
    unique_words = set(words)
    lines = text.splitlines()
    headings = HEADING_PATTERN.findall(text)
    sentence_count = len(SENTENCE_PATTERN.findall(text))
    if words and sentence_count == 0:
        sentence_count = 1

    return {
        "text_char_count": len(text),
        "text_line_count": len(lines),
        "text_nonempty_line_count": sum(bool(line.strip()) for line in lines),
        "text_word_count": len(words),
        "text_unique_word_count": len(unique_words),
        "text_lexical_diversity": len(unique_words) / len(words) if words else 0.0,
        "text_average_word_length": (
            sum(len(word) for word in words) / len(words) if words else 0.0
        ),
        "text_sentence_count": sentence_count,
        "text_average_sentence_words": len(words) / sentence_count if sentence_count else 0.0,
        "markdown_heading_count": len(headings),
        "markdown_max_heading_depth": max((len(mark) for mark in headings), default=0),
        "markdown_unordered_list_items": len(UNORDERED_LIST_PATTERN.findall(text)),
        "markdown_ordered_list_items": len(ORDERED_LIST_PATTERN.findall(text)),
        "markdown_fenced_code_blocks": len(FENCE_PATTERN.findall(text)) // 2,
        "markdown_inline_code_count": len(INLINE_CODE_PATTERN.findall(text)),
        "markdown_link_count": len(MARKDOWN_LINK_PATTERN.findall(text)),
        "markdown_url_count": len(URL_PATTERN.findall(text)),
        "markdown_blockquote_count": len(BLOCKQUOTE_PATTERN.findall(text)),
        "markdown_checkbox_count": len(CHECKBOX_PATTERN.findall(text)),
        "markdown_table_row_count": len(TABLE_ROW_PATTERN.findall(text)),
        "markdown_shell_prompt_count": len(SHELL_PROMPT_PATTERN.findall(text)),
    }

def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona atributos derivados sem remover texto ou identificadores."""
    extracted = pd.DataFrame(
        [extract_markdown_features(content) for content in frame["content"]],
        index=frame.index,
    )
    result = pd.concat([frame.copy(), extracted], axis=1)
    result["frontmatter_name_char_count"] = result["name"].fillna("").str.len()
    result["frontmatter_description_char_count"] = result["description"].fillna("").str.len()
    result["frontmatter_name_missing"] = result["name"].isna().astype("int8")
    result["frontmatter_description_missing"] = result["description"].isna().astype("int8")
    return result

def make_grouped_split(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_seed: int = 20261009,
) -> tuple[pd.Series, dict[str, int | float | str]]:
    """Cria uma divisao deterministica, estratificada e sem grupos sobrepostos."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size deve estar entre 0 e 0.5")
    required = {"content_hash", "reused", "representative_repo_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para a divisao: {sorted(missing)}")
    if frame["representative_repo_id"].isna().any():
        raise ValueError("representative_repo_id possui valores ausentes")

    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    target_rate = float(frame["reused"].mean())
    candidates: list[tuple[float, np.ndarray]] = []
    for _, test_indices in splitter.split(
        frame,
        y=frame["reused"],
        groups=frame["representative_repo_id"],
    ):
        candidate = frame.iloc[test_indices]
        size_difference = abs(len(candidate) / len(frame) - test_size)
        rate_difference = abs(float(candidate["reused"].mean()) - target_rate)
        candidates.append((size_difference + rate_difference, test_indices))

    _, selected_test_indices = min(candidates, key=lambda item: item[0])
    assignments = pd.Series("train", index=frame.index, dtype="string")
    assignments.iloc[selected_test_indices] = "test"

    train_groups = set(frame.loc[assignments == "train", "representative_repo_id"])
    test_groups = set(frame.loc[assignments == "test", "representative_repo_id"])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Foram encontrados {len(overlap)} grupos em treino e teste")

    train_mask = assignments == "train"
    test_mask = assignments == "test"
    manifest: dict[str, int | float | str] = {
        "strategy": "StratifiedGroupKFold",
        "random_seed": random_seed,
        "requested_test_fraction": test_size,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "actual_test_fraction": round(float(test_mask.mean()), 6),
        "overall_positive_rate": round(target_rate, 6),
        "train_positive_rate": round(float(frame.loc[train_mask, "reused"].mean()), 6),
        "test_positive_rate": round(float(frame.loc[test_mask, "reused"].mean()), 6),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": 0,
    }
    return assignments, manifest

def split_fingerprint(assignments: pd.DataFrame) -> str:
    """Produz uma assinatura estavel para detectar mudancas na divisao."""
    canonical = assignments.sort_values("content_hash").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def build_feature_summary(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Resume cobertura e distribuicao dos atributos numericos extraidos."""
    rows = []
    for column in feature_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(100 * float(series.isna().mean()), 4),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "minimum": float(series.min()) if series.notna().any() else np.nan,
                "maximum": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)

def save_class_figure(
    frame: pd.DataFrame,
    output: Path,
    title: str = "Distribuicao do alvo na amostra GitSkills",
) -> None:
    counts = frame["reused"].value_counts().sort_index()
    labels = ["Unica em um repositorio", "Reutilizada"]
    figure, axis = plt.subplots(figsize=(7.5, 5))
    bars = axis.bar(labels, counts.reindex([0, 1], fill_value=0), color=["#64748b", "#2563eb"])
    axis.set_ylabel("Conteudos distintos")
    axis.set_title(title)
    axis.bar_label(bars, fmt="%d", padding=4)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


# ==========================================================================
# Extrai atributos em lotes e congela a divisao final de treino e teste.
# ==========================================================================

DEFAULT_INPUT = Path("data/processed/final_modeling_sample.parquet")

DEFAULT_TARGETS = Path("data/processed/final_targets.parquet")

DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")

DEFAULT_SPLITS = Path("data/processed/final_split_assignments.parquet")

DEFAULT_TABLES = Path("results/tables")

DEFAULT_FIGURES = Path("results/figures")

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

def features_parse_args() -> argparse.Namespace:
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

def features_main() -> None:
    args = features_parse_args()
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
