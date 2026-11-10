import numpy as np
import pandas as pd

from skillreuse.owner import PATH_PATTERNS
from skillreuse.owner_models import (
    CLEAN_FULL,
    SHORTCUTS,
    VIEWS,
    _clf_metrics,
    _preprocessor,
    _subsets,
)


def test_clean_views_exclude_selection_contaminated_features() -> None:
    for view, columns in VIEWS.items():
        if view in ("clean_full+shortcuts", "shortcuts_only"):
            continue
        assert not set(columns) & set(SHORTCUTS), view
    assert set(SHORTCUTS) <= set(VIEWS["clean_full+shortcuts"])
    assert not {"owner_count", "cross_owner", "repository_count", "canonical_share"} & set(
        CLEAN_FULL
    )


def test_path_patterns_match_expected_directories() -> None:
    paths = pd.Series([
        ".claude/skills/a/SKILL.md", "templates/x/SKILL.md", "plugins/p/skills/b/SKILL.md",
        ".codex/skills/c/SKILL.md", "SKILL.md",
    ])
    flags = {k: paths.str.contains(v, case=False, regex=True) for k, v in PATH_PATTERNS.items()}
    assert flags["p_claude_dir"].tolist() == [True, False, False, False, False]
    assert flags["p_template"].tolist() == [False, True, False, False, False]
    assert flags["p_plugin"].tolist() == [False, False, True, False, False]
    assert flags["p_tool_dir"].tolist() == [False, False, False, True, False]


def test_preprocessor_handles_numeric_and_categorical_columns() -> None:
    frame = pd.DataFrame({
        "log_stars": [1.0, np.nan, 3.0, 2.0] * 5,
        "repo_language": ["Python", None, "Go", "Python"] * 5,
    })
    out = _preprocessor(["log_stars", "repo_language"]).fit_transform(frame)
    assert out.shape[0] == 20


def test_name_unseen_subset_ignores_empty_names() -> None:
    train = pd.DataFrame({"name_norm": ["a", "b", ""]})
    valid = pd.DataFrame({"name_norm": ["a", "c", ""]})
    mask = _subsets(train, valid)["name_unseen"]
    assert mask.tolist() == [False, True, True]


def test_classification_metrics_perfect_and_constant() -> None:
    y = np.array([0, 0, 1, 1])
    assert _clf_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert _clf_metrics(y, np.full(4, 0.5))["roc_auc"] == 0.5


def test_skewed_columns_selects_nonnegative_continuous_only() -> None:
    from skillreuse.feature_eda import skewed_columns

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "body_chars": np.exp(rng.normal(0, 2, 500)),       # assimetrica e continua
        "text_char_count": rng.normal(10, 1, 500).clip(0),  # simetrica
        "has_scripts": rng.integers(0, 2, 500).astype(float),  # binaria
        "markdown_link_count": -np.exp(rng.normal(0, 2, 500)),  # negativa
    })
    cols = skewed_columns(frame, ["body_chars", "text_char_count", "has_scripts",
                                  "markdown_link_count"])
    assert cols == ["body_chars"]


def test_log_preprocessor_reduces_skew_and_keeps_shape() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "body_chars": np.exp(rng.normal(0, 2, 300)),
        "log_stars": rng.normal(2, 1, 300),
    })
    plain = _preprocessor(["body_chars", "log_stars"]).fit_transform(frame)
    logged = _preprocessor(["body_chars", "log_stars"], ["body_chars"]).fit_transform(frame)
    assert plain.shape[0] == logged.shape[0] == 300
    assert pd.Series(np.asarray(logged)[:, 0]).skew() < pd.Series(np.asarray(plain)[:, 0]).skew()
