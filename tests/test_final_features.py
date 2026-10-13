import pandas as pd

from gitskills_reuse.features.final import (
    EXTRACTION_COLUMNS,
    IDENTIFIER_COLUMNS,
    RAW_FEATURE_COLUMNS,
    engineered_column_names,
)


def test_final_feature_schema_excludes_text_and_target_leakage() -> None:
    output_columns = set(IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS)

    assert "content" not in output_columns
    assert "name" not in output_columns
    assert "description" not in output_columns
    assert "occurrence_count" not in output_columns
    assert "repository_count" not in output_columns
    assert "reuse_intensity" not in output_columns
    assert set(EXTRACTION_COLUMNS).isdisjoint(output_columns)


def test_final_feature_identifiers_are_minimal() -> None:
    assert IDENTIFIER_COLUMNS == ["content_hash", "representative_repo_id", "reused"]
    assert pd.Series(IDENTIFIER_COLUMNS).is_unique


def test_engineered_columns_do_not_repeat_raw_frontmatter_field() -> None:
    columns = ["frontmatter_valid", "frontmatter_name_missing", "text_word_count"]

    assert engineered_column_names(columns) == [
        "frontmatter_name_missing",
        "text_word_count",
    ]
