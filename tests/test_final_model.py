import numpy as np
import pandas as pd

from skillreuse.final_model import (
    CLF_SPACES,
    FAMILIES,
    REG_NAME,
    _cluster_bootstrap,
    make_estimator,
    spaces,
)


def test_every_family_has_estimators_and_search_spaces() -> None:
    assert set(CLF_SPACES) == set(FAMILIES) == set(REG_NAME)
    for task in ("classification", "regression"):
        for family in FAMILIES:
            assert hasattr(make_estimator(task, family), "fit")
            assert spaces(task, family)


def test_cluster_bootstrap_resamples_whole_owners() -> None:
    owners = np.array(["a"] * 3 + ["b"] * 2 + ["c"])
    seen: list[np.ndarray] = []

    def capture(weights: np.ndarray) -> float:
        seen.append(weights.copy())
        return float(weights.sum())

    out = _cluster_bootstrap(owners, capture, 50, seed=1)
    assert len(out) == 50
    for w in seen:
        assert len(set(w[:3])) == 1 and len(set(w[3:5])) == 1  # linhas do mesmo dono
        assert w[3] == w[4]
    assert all(np.isfinite(out))


def test_holdout_is_owner_and_hash_disjoint_from_sample() -> None:
    import json
    from pathlib import Path

    manifest = Path("results/owner/holdout/holdout_manifest.json")
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text())
    assert data["owner_overlap_with_sample_reused"] == 0
    assert data["hash_overlap_with_200k_sample"] == 0
    assert data["max_rows_per_owner"] <= data["cap_per_owner"]
    assert isinstance(pd.__version__, str)
