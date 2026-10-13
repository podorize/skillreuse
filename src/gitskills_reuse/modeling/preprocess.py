"""Construcao de transformadores ajustados somente no treino de cada dobra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEAKAGE_COLUMNS = {
    "content_hash",
    "representative_repo_id",
    "repository_count",
    "occurrence_count",
    "reused",
    "reuse_intensity",
    "split_stratum",
    "split",
    "text",
}

STRUCTURAL_NUMERIC_FEATURES = [
    "frontmatter_valid", "body_chars", "sibling_count", "sibling_bytes",
    "has_scripts", "has_references", "composition_fetched", "composition_truncated",
    "text_char_count", "text_line_count", "text_nonempty_line_count", "text_word_count",
    "text_unique_word_count", "text_lexical_diversity", "text_average_word_length",
    "text_sentence_count", "text_average_sentence_words", "markdown_heading_count",
    "markdown_max_heading_depth", "markdown_unordered_list_items",
    "markdown_ordered_list_items", "markdown_fenced_code_blocks",
    "markdown_inline_code_count", "markdown_link_count", "markdown_url_count",
    "markdown_blockquote_count", "markdown_checkbox_count", "markdown_table_row_count",
    "markdown_shell_prompt_count", "frontmatter_name_char_count",
    "frontmatter_description_char_count", "frontmatter_name_missing",
    "frontmatter_description_missing",
]

STRUCTURAL_CATEGORICAL_FEATURES = ["filename", "location_class"]

REPOSITORY_NUMERIC_FEATURES = [
    "history_fetched", "commit_count", "repo_stars", "repo_forks", "repo_is_fork",
    "repo_metadata_fetched",
]

REPOSITORY_CATEGORICAL_FEATURES = ["repo_language", "repo_license"]

NUMERIC_FEATURES = STRUCTURAL_NUMERIC_FEATURES + REPOSITORY_NUMERIC_FEATURES
CATEGORICAL_FEATURES = STRUCTURAL_CATEGORICAL_FEATURES + REPOSITORY_CATEGORICAL_FEATURES


def build_preprocessor(
    frame: pd.DataFrame,
    *,
    use_text: bool = True,
    use_structural: bool = True,
    use_repository: bool = True,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
    max_features: int = 50_000,
) -> ColumnTransformer:
    """Monta blocos multimodais; o chamador ajusta o transformer no fold de treino."""
    transformers: list[tuple[str, object, Sequence[str] | str]] = []
    if use_text:
        if "content" not in frame:
            raise ValueError("Coluna 'content' necessaria para o bloco TF-IDF")
        transformers.append(
            (
                "text_tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
                "content",
            )
        )

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    if use_structural:
        numeric_cols.extend([col for col in STRUCTURAL_NUMERIC_FEATURES if col in frame])
        categorical_cols.extend([col for col in STRUCTURAL_CATEGORICAL_FEATURES if col in frame])

    if use_repository:
        numeric_cols.extend([col for col in REPOSITORY_NUMERIC_FEATURES if col in frame])
        categorical_cols.extend([col for col in REPOSITORY_CATEGORICAL_FEATURES if col in frame])

    if numeric_cols:
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("structured_numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]
        )
        transformers.append(("repository_categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("Ao menos um grupo de atributos deve estar habilitado")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def validate_predictor_columns(frame: pd.DataFrame) -> None:
    """Falha se uma coluna alvo ou identificador estiver prestes a virar predictor."""
    leaked = sorted(LEAKAGE_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"Colunas de alvo, identificacao ou particao devem ser removidas: {leaked}")


def predictor_view(
    frame: pd.DataFrame,
    *,
    use_text: bool = True,
    use_structural: bool = True,
    use_repository: bool = True,
) -> pd.DataFrame:
    """Retorna somente as colunas de entrada explicitas, com guarda contra vazamento."""
    allowed: list[str] = []
    if use_text and "content" in frame:
        allowed.append("content")
    if use_structural:
        allowed.extend(STRUCTURAL_NUMERIC_FEATURES + STRUCTURAL_CATEGORICAL_FEATURES)
    if use_repository:
        allowed.extend(REPOSITORY_NUMERIC_FEATURES + REPOSITORY_CATEGORICAL_FEATURES)

    if not allowed:
        allowed = ["content"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES

    predictors = frame[[column for column in allowed if column in frame]].copy()
    validate_predictor_columns(predictors)
    return predictors
