import pandas as pd
import pytest

from gitskills_reuse.modeling.preprocess import (
    REPOSITORY_CATEGORICAL_FEATURES,
    REPOSITORY_NUMERIC_FEATURES,
    STRUCTURAL_CATEGORICAL_FEATURES,
    STRUCTURAL_NUMERIC_FEATURES,
    build_preprocessor,
    predictor_view,
    validate_predictor_columns,
)


def _make_dummy_frame() -> pd.DataFrame:
    data = {
        "content_hash": ["h1", "h2"],
        "representative_repo_id": ["r1", "r2"],
        "reused": [0, 1],
        "content": ["skill one content", "skill two content"],
    }
    for col in STRUCTURAL_NUMERIC_FEATURES + REPOSITORY_NUMERIC_FEATURES:
        data[col] = [1, 2]
    for col in STRUCTURAL_CATEGORICAL_FEATURES + REPOSITORY_CATEGORICAL_FEATURES:
        data[col] = ["val_a", "val_b"]
    return pd.DataFrame(data)


def test_predictor_view_filters_groups() -> None:
    frame = _make_dummy_frame()

    view_all = predictor_view(frame, use_text=True, use_structural=True, use_repository=True)
    assert "content" in view_all.columns
    assert "text_word_count" in view_all.columns
    assert "repo_stars" in view_all.columns
    assert "reused" not in view_all.columns

    view_structure = predictor_view(frame, use_text=False, use_structural=True, use_repository=False)
    assert "content" not in view_structure.columns
    assert "text_word_count" in view_structure.columns
    assert "repo_stars" not in view_structure.columns

    view_repo = predictor_view(frame, use_text=False, use_structural=False, use_repository=True)
    assert "content" not in view_repo.columns
    assert "text_word_count" not in view_repo.columns
    assert "repo_stars" in view_repo.columns

    view_text = predictor_view(frame, use_text=True, use_structural=False, use_repository=False)
    assert "content" in view_text.columns
    assert "text_word_count" not in view_text.columns
    assert "repo_stars" not in view_text.columns


def test_build_preprocessor_ablation_transformers() -> None:
    frame = _make_dummy_frame()

    pre_all = build_preprocessor(frame, use_text=True, use_structural=True, use_repository=True)
    names_all = [name for name, _, _ in pre_all.transformers]
    assert "text_tfidf" in names_all
    assert "structured_numeric" in names_all
    assert "repository_categorical" in names_all

    pre_struct = build_preprocessor(frame, use_text=False, use_structural=True, use_repository=False)
    names_struct = [name for name, _, _ in pre_struct.transformers]
    assert "text_tfidf" not in names_struct
    assert "structured_numeric" in names_struct

    pre_repo = build_preprocessor(frame, use_text=False, use_structural=False, use_repository=True)
    names_repo = [name for name, _, _ in pre_repo.transformers]
    assert "text_tfidf" not in names_repo
    assert "structured_numeric" in names_repo
    assert "repository_categorical" in names_repo

    with pytest.raises(ValueError, match="Ao menos um grupo"):
        build_preprocessor(frame, use_text=False, use_structural=False, use_repository=False)
