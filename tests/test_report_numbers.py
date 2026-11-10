"""Confere os numeros citados no relatorio (paper/draft) contra os artefatos em results/owner."""

import json
from pathlib import Path

import pandas as pd
import pytest

R = Path("results/owner")
DRAFT = Path("paper/draft")
pytestmark = pytest.mark.skipif(
    not (R / "holdout" / "holdout_results.json").exists() or not DRAFT.exists(),
    reason="artefatos ou rascunho ausentes",
)


def text() -> str:
    return "\n".join(p.read_text() for p in sorted(DRAFT.glob("0[0-8]_*.md")))


def neg(x: float, nd: int = 3) -> str:
    return ("−" if x < 0 else "") + f"{abs(x):.{nd}f}"


def thousands(n: int) -> str:
    return f"{n:,}"


def check(doc: str, expected: list[str]) -> None:
    missing = [s for s in expected if s not in doc]
    assert not missing, f"nao encontrado no relatorio: {missing}"


def test_holdout_numbers() -> None:
    r = json.loads((R / "holdout" / "holdout_results.json").read_text())
    c, g = r["classification"], r["regression"]
    lo, hi = c["roc_auc_ci95"]
    check(text(), [
        f"{c['roc_auc']:.3f}", f"[{lo:.3f}; {hi:.3f}]", f"{c['pr_auc']:.3f}",
        f"{c['brier']:.3f}", f"{c['log_loss']:.3f}", f"{g['rmsle']:.3f}",
        f"{g['rmsle_baseline_mean_log1p']:.3f}", f"{g['spearman']:.3f}",
        f"[{neg(g['rmsle_diff_ci95'][0])}; {neg(g['rmsle_diff_ci95'][1])}]",
        thousands(r["holdout_rows"]), thousands(r["holdout_owners"]),
    ])


def test_threshold_numbers() -> None:
    p = R / "interpretation" / "threshold_results.json"
    if not p.exists():
        pytest.skip("interpretacao ausente")
    t = json.loads(p.read_text())
    h, c = t["holdout"], t["holdout"]["confusion"]
    true_min = c["true_single_owner_pred_single_owner"] + c["true_single_owner_pred_cross_owner"]
    check(text(), [
        f"{t['threshold']:.2f}", f"{h['balanced_accuracy']:.3f}",
        f"[{h['balanced_accuracy_ci95'][0]:.3f}; {h['balanced_accuracy_ci95'][1]:.3f}]",
        f"{h['minority_pr_auc']:.3f}",
        f"[{h['minority_pr_auc_ci95'][0]:.3f}; {h['minority_pr_auc_ci95'][1]:.3f}]",
        f"{h['minority_recall']:.2f}", f"{h['minority_precision']:.2f}",
        f"{t['dev_balanced_accuracy_at_threshold']:.3f}",
        f"{t['dev_balanced_accuracy_at_0.5']:.3f}",
        thousands(c["true_single_owner_pred_single_owner"]), thousands(true_min),
        thousands(c["true_cross_owner_pred_single_owner"]),
    ])


def test_eda_and_audit_numbers() -> None:
    e = json.loads((R / "eda_summary.json").read_text())
    reused, cross = e["population_reused_by_repo"], e["population_cross_owner"]
    check(text(), [
        f"{e['share_same_owner_only'] * 100:.1f}%", f"{e['top1_owner_share'] * 100:.1f}%",
        f"{e['top5_owner_share'] * 100:.1f}%", thousands(reused), thousands(cross),
        thousands(reused - cross), f"{(reused - cross) / reused * 100:.1f}%",
        f"{e['representative_rule_match'] * 100:.1f}%", thousands(e["dev_rows"]),
        thousands(e["dev_owners"]), f"{e['name_empty_share'] * 100:.1f}%",
        thousands(e["population_contents"]),
    ])
    audit = pd.read_csv(R / "audit_by_repository_count.csv")
    exp = []
    for _, row in audit.iterrows():
        exp += [thousands(int(row["n"])), f"{row['p_representative_canonical'] * 100:.1f}%",
                f"{row['mean_share_of_copies_canonical'] * 100:.1f}%"]
    check(text(), exp)


def test_primary_hypotheses_numbers() -> None:
    h = pd.read_csv(R / "primary" / "primary_hypotheses.csv")
    exp = []
    for _, row in h.iterrows():
        exp += [f"{row['score_a']:.3f}", f"{row['score_b']:.3f}"]
        assert row["p_holm_family"] < 0.05, row["hypothesis"]
    check(text(), exp)
    assert "All hypotheses are rejected after Holm correction" in text()


def test_sensitivity_numbers() -> None:
    s = pd.read_csv(R / "figures" / "sensitivity_owner_limit.csv")
    exp = []
    for _, row in s.iterrows():
        exp += [f"{row['auc_clean_rf']:.3f}", f"{row['rmsle_gain_clean_rf']:.3f}"]
    check(text(), exp)


def test_tuning_numbers() -> None:
    c = pd.read_csv(R / "final" / "classification_nested_summary.csv").set_index("config")
    g = pd.read_csv(R / "final" / "regression_nested_summary.csv").set_index("config")
    check(text(), [
        f"{c.loc['random_forest|default', 'mean']:.3f}",
        f"{c.loc['random_forest|tuned', 'mean']:.3f}",
        f"{g.loc['hist_gradient_boosting|default', 'mean']:.3f}",
        f"{g.loc['baseline', 'mean']:.3f}",
        f"{g.loc['random_forest|default', 'mean']:.3f}",
    ])


def test_feature_eda_numbers() -> None:
    p = R / "feature_eda" / "feature_eda_summary.json"
    if not p.exists():
        pytest.skip("feature EDA ausente")
    f = json.loads(p.read_text())
    check(text(), [
        str(f["matrix_columns_after_preprocessing"]), f"{f['cross_owner_prevalence'] * 100:.1f}%",
        f"{f['max_missing_share'] * 100:.1f}%",
        str(f["numeric_columns"] + f["categorical_columns"]),
    ])
    assert len(f["skewed_columns_gt_2"]) == 25 and "Twenty-five" in text()
    assert f["collinear_pairs_gt_0.95"] == 8 and "Eight pairs" in text()


def test_holdout_manifest_matches_text() -> None:
    m = json.loads((R / "holdout" / "holdout_manifest.json").read_text())
    check(text(), [thousands(m["rows"]), thousands(m["owners"]), str(m["max_rows_per_owner"]),
                   f"{m['sanity_cross_owner_prevalence'] * 100:.1f}%"])
