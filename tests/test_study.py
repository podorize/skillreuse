"""Testes das funcoes do estudo: T0 e alvo, divisao, dobras, regras, janela e limiar."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skillreuse import cohort as C
from skillreuse import study as S
from skillreuse.stats import boot_p, nb_test, owner_bootstrap


def _dates() -> pd.DataFrame:
    t = pd.Timestamp
    rows = [  # skill a: origem em ana, copia por bruno 10 dias depois e por um agregador antes
        ("a", "ana/x", "ana", 5, t("2026-01-10", tz="UTC")),
        ("a", "bruno/y", "bruno", 5, t("2026-01-20", tz="UTC")),
        ("a", "mirror/all", "mirror", 5000, t("2026-01-05", tz="UTC")),
        # skill b: so autocopia (mesmo dono) e uma copia nao datada
        ("b", "carla/p", "carla", 3, t("2026-02-01", tz="UTC")),
        ("b", "carla/q", "carla", 3, t("2026-02-05", tz="UTC")),
        ("b", "davi/z", "davi", 3, pd.NaT),
    ]
    d = pd.DataFrame(rows, columns=["content_hash", "repo_full_name", "owner", "repo_skills",
                                    "arrival"])
    d["path"] = "SKILL.md"
    d["repo_created_at"] = d["arrival"]
    return d


def test_timing_excludes_aggregators_and_self_copies() -> None:
    t = C.timing(_dates(), 100).set_index("content_hash")
    assert t.loc["a", "author"] == "ana"                       # o agregador nao vira origem
    assert t.loc["a", "first_copy_lag_days"] == pytest.approx(10)
    assert np.isnan(t.loc["b", "first_copy_lag_days"])         # autocopia nao conta
    assert not t.loc["b", "all_copies_dated"]
    none = C.timing(_dates(), None).set_index("content_hash")  # sem exclusao: agregador e origem
    assert none.loc["a", "author"] == "mirror"


def test_split_authors_holds_out_30_percent_per_band() -> None:
    t = pd.DataFrame({"author": [f"u{i}" for i in range(1000)]})
    hold = C.split_authors(t)
    assert abs(hold.mean() - 0.30) < 0.01


def _toy(n_owners: int = 60, per_owner: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"content_hash": f"{o:03d}{k}", "owner": f"o{o}",
                          "replicated": int(rng.random() < 0.2)}
                         for o in range(n_owners) for k in range(per_owner)])


def test_folds_are_author_disjoint_and_cover_all_rows() -> None:
    frame, seen = _toy(), []
    for _r, _f, tr, va in S.folds(frame):
        assert not set(frame.iloc[tr]["owner"]) & set(frame.iloc[va]["owner"])
        seen.append(len(va))
    assert sum(seen) == S.REPEATS * len(frame)


def test_window_rule_picks_smallest_window_covering_80_percent() -> None:
    lags = [5] * 70 + [40] * 15 + [80] * 15
    dev = pd.DataFrame({"t0": pd.Timestamp("2026-01-01", tz="UTC"),
                        "first_copy_lag_days": lags})
    assert S.choose_window(dev)["W"] == 60


def test_eda_rules_flag_constant_skewed_and_collinear() -> None:
    rng = np.random.default_rng(1)
    n = 400
    base = rng.lognormal(3, 1.5, n)
    dev = pd.DataFrame({c: rng.normal(5, 1, n) for c in S.TEXT + S.ORIGIN})
    for c in S.CATEGORICAL:
        dev[c] = "x"
    dev["repo_is_fork"] = 0.0
    dev["body_chars"] = base
    dev["text_char_count"] = base * 1.01
    rules = S.eda_rules(dev)
    assert "repo_is_fork" in rules["constant"]
    assert "body_chars" in rules["skewed_log1p_nontree"]
    assert len({"text_char_count", "body_chars"} & set(rules["drop_collinear_nontree"])) == 1


def test_views_keep_text_and_origin_apart() -> None:
    assert len(S.TEXT) == 27 and not set(S.TEXT) & set(S.ORIGIN)
    assert set(S.VIEWS["size"]) <= set(S.TEXT)
    rules = {"constant": [], "drop_collinear_nontree": ["text_char_count"]}
    assert "text_char_count" in S.view_columns("text", "hist_gb", rules)
    assert "text_char_count" not in S.view_columns("text", "logistic", rules)


def test_owner_bootstrap_resamples_whole_authors() -> None:
    owners = np.array(["a"] * 3 + ["b"] * 2 + ["c"])
    out = owner_bootstrap(owners, lambda w: float(w[0] == w[1] == w[2]), n_boot=50, seed=3)
    assert out.min() == 1.0


def test_boot_p_and_nb_test() -> None:
    assert boot_p(np.full(1999, 0.7), 0.5) == 2 / 2000
    t = nb_test(np.array([0.05, 0.06, 0.04, 0.05, 0.07, 0.05, 0.06, 0.05, 0.04]), ratio=0.5)
    assert t["ci95_low"] > 0 and t["p_nb"] < 0.05


def test_threshold_report_counts() -> None:
    r = S.threshold_report(np.array([1, 1, 0, 0, 0]), np.array([0.9, 0.3, 0.8, 0.1, 0.2]), 0.5)
    assert (r["true_positive"], r["false_positive"], r["false_negative"]) == (1, 1, 1)
    assert abs(r["balanced_accuracy"] - (0.5 + 2 / 3) / 2) < 1e-12


def test_frozen_models_match_recorded_hashes() -> None:
    import joblib

    freeze = Path("results/phase5/freeze.json")
    if not freeze.is_file():
        pytest.skip("modelos congelados ausentes")
    for view, entry in json.loads(freeze.read_text())["models"].items():
        assert S.sha256(Path(entry["file"])) == entry["sha256"], view
        assert hasattr(joblib.load(entry["file"])["pipeline"], "predict_proba"), view
