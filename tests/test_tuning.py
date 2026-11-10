import numpy as np
import pandas as pd

from skillreuse.benchmark import REGRESSION_MODELS, _fit_predict, paired_test_table
from skillreuse.tuning import SEARCH_SPACES, paired_tables, summarize


def _folds() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for repeat in (1, 2, 3):
        for fold in (1, 2, 3):
            base = rng.normal(0.6, 0.01)
            for model, variant, shift in [
                ("random_forest_log", "tuned", 0.0),
                ("random_forest_log", "default", 0.001),
                ("ridge_log", "tuned", 0.02),
                ("ridge_log", "default", 0.02),
                ("dummy_median_log", "baseline", 0.15),
                ("dummy_mean_log", "baseline", 0.1),
            ]:
                rows.append(
                    {"repeat": repeat, "fold": fold, "model": model, "variant": variant,
                     "view": "repository_only", "train_rows": 200, "validation_rows": 100,
                     "rmsle": base + shift, "mae": 3 + shift, "mae_log1p": 0.4, "spearman": 0.3}
                )
    return pd.DataFrame(rows)


def test_every_regressor_has_a_search_space() -> None:
    assert set(SEARCH_SPACES) == set(REGRESSION_MODELS)


def test_paired_test_detects_consistent_difference() -> None:
    df = _folds()
    df["config"] = df["model"] + "|" + df["variant"]
    table = paired_test_table(
        df, [("random_forest_log|tuned", "dummy_median_log|baseline")], label="config"
    )
    row = table[table["metric"] == "rmsle"].iloc[0]
    assert row["mean_diff_a_minus_b"] < 0
    assert row["a_better_share"] == 1.0
    assert row["p_nadeau_bengio"] < 0.05
    assert row["ci95_high"] < 0


def test_paired_tables_families_and_holm() -> None:
    df = _folds()
    table = paired_tables(df, summarize(df))
    assert {"best_vs_tuned_models", "best_vs_baseline", "tuned_vs_default"} <= set(table["family"])
    assert (table["p_nb_holm"] >= table["p_nadeau_bengio"] - 1e-12).all()


def test_dummy_median_prediction_is_constant() -> None:
    y = np.array([1.0, 2.0, 2.0, 10.0])
    pred = _fit_predict("dummy_median_log", np.zeros((4, 1)), y, np.zeros((3, 1)))
    assert np.allclose(pred, np.expm1(np.median(np.log1p(y))))
