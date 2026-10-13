import pandas as pd
import pytest

from gitskills_reuse.data.targets import (
    STRATA,
    build_targets,
    intensity_split_summary,
    make_two_target_split,
)


def test_targets_follow_repository_count_and_keep_singletons_without_intensity() -> None:
    counts = [1, 2, 3, 4, 5, 9, 10, 951]
    source = pd.DataFrame(
        {
            "content_hash": [f"hash-{index}" for index in range(len(counts))],
            "representative_repo_id": [f"repo-{index}" for index in range(len(counts))],
            "repository_count": counts,
            "reused": [int(count >= 2) for count in counts],
        }
    )

    targets = build_targets(source)

    assert targets["reuse_intensity"].tolist()[1:] == [0, 1, 1, 2, 2, 3, 3]
    assert pd.isna(targets.loc[0, "reuse_intensity"])
    assert targets["split_stratum"].tolist() == [
        "not_reused",
        "reuse_2",
        "reuse_3_4",
        "reuse_3_4",
        "reuse_5_9",
        "reuse_5_9",
        "reuse_10_plus",
        "reuse_10_plus",
    ]


def test_target_builder_rejects_inconsistent_or_duplicate_source() -> None:
    source = pd.DataFrame(
        {
            "content_hash": ["a", "b"],
            "representative_repo_id": ["r1", "r2"],
            "repository_count": [1, 2],
            "reused": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="nao corresponde"):
        build_targets(source)

    source.loc[1, "reused"] = 1
    source.loc[1, "content_hash"] = "a"
    with pytest.raises(ValueError, match="unico"):
        build_targets(source)


def test_two_target_split_is_deterministic_grouped_and_order_independent() -> None:
    rows = []
    for group in range(50):
        for count in (1, 2, 3, 5, 10):
            rows.append(
                {
                    "content_hash": f"hash-{group:02d}-{count:02d}",
                    "representative_repo_id": f"repo-{group:02d}",
                    "repository_count": count,
                    "reused": int(count >= 2),
                }
            )
    targets = build_targets(pd.DataFrame(rows))
    first, manifest = make_two_target_split(targets, random_seed=7, restarts=3)
    second, second_manifest = make_two_target_split(
        targets.sample(frac=1, random_state=17), random_seed=7, restarts=3
    )
    first_by_hash = pd.Series(first.to_numpy(), index=targets["content_hash"])
    shuffled = targets.sample(frac=1, random_state=17)
    second_by_hash = pd.Series(second.to_numpy(), index=shuffled["content_hash"])

    assert first_by_hash.sort_index().equals(second_by_hash.sort_index())
    assert manifest == second_manifest
    assert manifest["candidate_count"] == 15
    assert manifest["group_overlap"] == 0
    assert set(targets.loc[first.eq("train"), "representative_repo_id"]).isdisjoint(
        set(targets.loc[first.eq("test"), "representative_repo_id"])
    )
    summary = intensity_split_summary(targets.assign(split=first))
    assert set(summary["split_stratum"]) == set(STRATA)
    assert (summary["count"] > 0).all()
