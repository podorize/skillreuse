"""Confere os numeros citados no artigo (paper/draft) contra os arquivos de resultado."""

import json
from pathlib import Path

import pandas as pd
import pytest

R = Path("results")
DRAFT = Path("paper/draft")
pytestmark = pytest.mark.skipif(
    not (R / "phase6" / "holdout_results.json").exists() or not DRAFT.exists(),
    reason="resultados ou rascunho ausentes",
)


def text() -> str:
    return "\n".join(p.read_text() for p in sorted(DRAFT.glob("0[0-8]_*.md")))


def load(path: str) -> dict:
    return json.loads((R / path).read_text())


def check(expected: list[str]) -> None:
    doc = text()
    missing = [s for s in expected if s not in doc]
    assert not missing, f"nao encontrado no artigo: {missing}"


def ci(pair: list[float], nd: int = 3) -> str:
    return f"[{pair[0]:.{nd}f}; {pair[1]:.{nd}f}]"


def test_data_and_split_numbers() -> None:
    m = load("phase2/manifest.json")
    s = m["collection_status"]
    check([f"{s['datada']:,}", f"{m['contents_dated']:,}", f"{m['contents_after_cap']:,}",
           f"{m['dev_rows']:,}", f"{m['holdout_rows']:,}", f"{m['dev_authors']:,}",
           f"{m['holdout_authors']:,}", f"{s['data_posterior_ao_retrato']:,}"])
    c = load("phase1/manifest.json")["counts"]
    check([f"{c['population_contents']:,}", f"{c['occurrences']:,}",
           f"{c['occurrences_outside_aggregators']:,}", f"{c['occurrences_in_aggregators']:,}"])


def test_window_and_eligibility() -> None:
    w = load("phase3/window.json")
    e = load("phase3/eda_summary.json")
    check([f"W = {w['W']} days", f"{w['coverage']['30']:.1%}", f"{w['coverage']['60']:.1%}",
           f"{w['lag_quantiles']['0.5']:.1f} days", f"{e['dev_eligible_rows']:,}",
           f"{e['replicated_rate']:.1%}"])


def test_model_selection_numbers() -> None:
    spot = pd.read_csv(R / "phase4" / "spot_summary.csv").set_index("config")
    check([f"{spot.loc[a, 'roc_auc_mean']:.3f} ± {spot.loc[a, 'roc_auc_std']:.3f}"
           for a in spot.index])
    nested = pd.read_csv(R / "phase4" / "nested_summary.csv").set_index("config")["mean"]
    check([f"{v:.3f}" for v in nested.to_numpy()])
    assert load("phase4/winner.json")["winner"] == "logistic|default"


def test_holdout_numbers() -> None:
    h = load("phase6/holdout_results.json")
    c = load("phase6/control_popularity.json")
    assert h["H1"]["success"] and c["C1_popularity"]["success"]
    check([f"{h['H1']['auc']:.3f}", ci(h["H1"]["ci95"]), f"{h['holdout_rows']:,}",
           f"{h['holdout_authors']:,}", f"{h['point']['size']['roc_auc']:.3f}",
           ci(h["view_ci95"]["size"]), f"{h['text_minus_size']['diff']:.3f}",
           ci(h["text_minus_size"]["ci95"]),
           f"{c['point']['origin_pop']['roc_auc']:.3f}",
           f"{c['point']['origin_pop+text']['roc_auc']:.3f}", ci(c["view_ci95"]["origin_pop"]),
           ci(c["view_ci95"]["origin_pop+text"]), f"{c['C1_popularity']['diff']:.3f}",
           ci(c["C1_popularity"]["ci95"]), f"{h['threshold']['precision']:.1%}",
           f"{h['threshold']['balanced_accuracy']:.3f}", f"{h['point']['text']['pr_auc']:.3f}"])


def test_robustness_numbers() -> None:
    s = load("phase7/sensitivity.json")
    for k in ("window_30", "window_90", "temporal"):
        check([f"{s[k]['auc']:.3f}", ci(s[k]["ci95"]), f"{s[k]['rows']:,}"])
    for k in ("aggregators_agg500", "aggregators_aggnone"):
        if isinstance(s[k], dict):
            check([f"{s[k]['auc']:.3f}", ci(s[k]["ci95"]), f"{s[k]['rows']:,}"])


def test_interpretation_numbers() -> None:
    p = pd.read_csv(R / "phase7" / "permutation_holdout.csv").set_index("feature")
    style = p.loc["family:Text style"]
    check([f"{style['auc_drop']:.3f}", ci([style["ci95_low"], style["ci95_high"]])])
    for fam in ("Size", "Front matter", "Links", "Code", "Markdown structure"):
        check([f"{p.loc[f'family:{fam}', 'auc_drop']:.3f}"])
    check([f"{p.loc['text_sentence_count', 'auc_drop']:.3f}"])
    assert load("phase7/interpretation_status.json")["rq2"] == "respondida"
