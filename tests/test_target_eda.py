import pandas as pd

from skillreuse.report import (
    count_distribution,
    group_distribution,
    ordinal_distribution,
    split_distribution,
)


def sample_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "content_hash": list("abcdef"),
            "representative_repo_id": ["r1", "r1", "r2", "r3", "r3", "r4"],
            "repository_count": [1, 2, 3, 5, 10, 2],
            "reused": [0, 1, 1, 1, 1, 1],
            "reuse_intensity": [pd.NA, 0, 1, 2, 3, 0],
            "split_stratum": [
                "not_reused",
                "reuse_2",
                "reuse_3_4",
                "reuse_5_9",
                "reuse_10_plus",
                "reuse_2",
            ],
            "split": ["train", "train", "train", "train", "test", "test"],
        }
    )


def test_target_distribution_reports_overdispersion_and_reused_rows() -> None:
    table = count_distribution(sample_targets(), split="train")
    reused = table.loc[table["population"].eq("reused")].iloc[0]
    assert reused["n"] == 3
    assert reused["median"] == 3
    assert reused["overdispersion_variance_mean"] > 0


def test_ordinal_distribution_has_four_fixed_classes() -> None:
    table = ordinal_distribution(sample_targets(), split="train")
    assert table["intensity_code"].tolist() == [0, 1, 2, 3]
    assert table["n"].sum() == 3


def test_group_distribution_preserves_repository_groups() -> None:
    table = group_distribution(sample_targets(), split="train")
    all_rows = table.loc[table["population"].eq("all")].iloc[0]
    assert all_rows["n_groups"] == 3
    assert all_rows["max_rows_per_group"] == 2


def test_split_distribution_contains_train_and_test() -> None:
    table = split_distribution(sample_targets())
    assert set(table["split"]) == {"train", "test"}
