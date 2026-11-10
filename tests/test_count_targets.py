import pandas as pd

from skillreuse.benchmark import run_pilot
from skillreuse.explain import explain_ridge
from skillreuse.models import (
    normalize_text,
    target_summary,
    validate_target_exclusion,
    validate_target_views,
)


def sample_count_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "content": ["a", "b"],
            "representative_repo_id": ["r1", "r2"],
            "repo_stars": [1, 4],
            "repository_count": [2, 10],
            "reused": [1, 1],
            "split_stratum": ["reuse_2", "reuse_10_plus"],
            "split": ["train", "train"],
            "count_target": [2, 10],
            "log_count_target": [1.0986, 2.3979],
        }
    )


def test_count_summary_uses_raw_count_and_overdispersion() -> None:
    summary = target_summary(sample_count_frame()).iloc[0]
    assert summary["rows"] == 2
    assert summary["min"] == 2
    assert summary["max"] == 10
    assert summary["variance_mean_ratio"] > 0


def test_count_target_is_not_in_predictor_view() -> None:
    columns = validate_target_exclusion(sample_count_frame())
    assert "repository_count" not in columns
    assert "count_target" not in columns
    assert "log_count_target" not in columns


def test_all_feature_views_exclude_count_target() -> None:
    counts = validate_target_views(sample_count_frame())
    assert set(counts) == {"skill_only", "repository_only", "hybrid"}
    assert counts["repository_only"] >= 0


def test_grouped_count_pilot_keeps_fold_groups_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "content": [f"skill token{i % 3}" for i in range(12)],
            "representative_repo_id": [f"repo{i}" for i in range(12)],
            "repo_stars": list(range(12)),
            "repository_count": [2 + i % 3 for i in range(12)],
            "count_target": [2 + i % 3 for i in range(12)],
            "log_count_target": [1.0 + i % 3 for i in range(12)],
            "split_stratum": ["reuse_2", "reuse_3_4", "reuse_5_9"] * 4,
        }
    )
    folds, artifacts = run_pilot(
        frame,
        candidate_features=(5,),
        top_k=(2,),
        views=("repository_only",),
        models=("dummy_log",),
        folds=3,
    )
    assert len(folds) == 3
    assert folds["group_overlap"].eq(0).all()
    assert artifacts["manifest"]["group_column"] == "representative_repo_id"


def test_count_explain_returns_global_and_local_tables() -> None:
    frame = sample_count_frame().assign(content_hash=["a", "b"])
    artifacts = explain_ridge(
        frame,
        view_name="repository_only",
        candidate_features=5,
        top_k_text=2,
        local_rows=1,
    )
    assert not artifacts["importance"].empty
    assert not artifacts["local"].empty
    assert artifacts["manifest"]["test_external_used"] is False


def test_text_artifact_variant_removes_dates_urls_and_numbers() -> None:
    text = "See https://example.test on 2026-02-27, issue 42."
    cleaned = normalize_text(text, "drop_artifacts")
    assert "https" not in cleaned
    assert "2026" not in cleaned
    assert "42" not in cleaned
