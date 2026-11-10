"""Direcao dos efeitos (dependencia parcial) e analise de limiar com a classe minoritaria.

* Dependencia parcial dos modelos congelados, calculada **fora da amostra** nas dobras de
  desenvolvimento agrupadas por dono (o modelo e treinado no treino da dobra e a curva usa as
  linhas de validacao); a media e o desvio entre dobras mostram a estabilidade.
* Limiar: escolhido no desenvolvimento (previsoes fora da amostra de 3 repeticoes x 3 dobras)
  pela maximizacao da acuracia balanceada; depois aplicado, sem ajuste, as previsoes ja
  salvas do holdout (leitura descritiva do mesmo modelo, nao uma nova avaliacao).
* Classe minoritaria: ``cross_owner = 0`` (conteudo que existe so dentro de um dono).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from skillreuse.final_model import _cluster_bootstrap, _plain, _predict, _target, pipeline
from skillreuse.owner_models import CLEAN_FULL, _folds, _prepare

CONFIG_DIR = Path("results/owner/final")
HOLDOUT_PRED = Path("results/owner/holdout/classification_predictions.csv")
OUTPUT_DIR = Path("results/owner/interpretation")
PDP_FEATURES = {
    "classification": ["log_stars", "log_forks", "push_recency_days"],
    "regression": ["log_skills_in_repo", "path_depth"],
}
LICENSE = "repo_license"


def _config(task: str) -> dict[str, object]:
    return json.loads((CONFIG_DIR / f"final_config_{task}.json").read_text())


def _frame(seed: int = 2027) -> pd.DataFrame:
    ns = argparse.Namespace(tier=None, exclude_owners_above=None, cap_per_owner=50,
                            max_rows=None, seed=seed)
    return _prepare(ns)


def pdp_grid(series: pd.Series, n: int = 12) -> np.ndarray:
    """Grade de quantis (5%-95%) sem repeticoes."""
    q = np.quantile(series.dropna(), np.linspace(0.05, 0.95, n))
    return np.unique(q)


def pdp_oof(task: str, frame: pd.DataFrame, seed: int = 2027) -> pd.DataFrame:
    cfg = _config(task)
    grids = {f: pdp_grid(frame[f]) for f in PDP_FEATURES[task]}
    top_licenses = frame[LICENSE].fillna("(missing)").value_counts().head(6).index.tolist()
    rows: list[dict[str, object]] = []
    for _, fold, train, valid in _folds(frame, "owner", 1, 3, seed):
        pipe = pipeline(task, cfg["family"], cfg["params"]).fit(train[CLEAN_FULL],
                                                               _target(train, task))
        base = valid[CLEAN_FULL]
        for feature, grid in grids.items():
            for value in grid:
                x = base.copy()
                x[feature] = value
                mean = float(_predict(task, pipe, x).mean())
                rows.append({"task": task, "feature": feature, "value": float(value),
                             "fold": fold, "mean_prediction": mean})
        for lic in top_licenses:
            x = base.copy()
            x[LICENSE] = np.nan if lic == "(missing)" else lic
            rows.append({"task": task, "feature": LICENSE, "value": lic, "fold": fold,
                         "mean_prediction": float(_predict(task, pipe, x).mean())})
        rows.append({"task": task, "feature": "(observed)", "value": "", "fold": fold,
                     "mean_prediction": float(_predict(task, pipe, base).mean())})
        print(f"[pdp/{task}] dobra {fold} concluida", flush=True)
    return pd.DataFrame(rows)


# --------------------------- limiar e classe minoritaria ---------------------------


def balanced_accuracy(y: np.ndarray, prob: np.ndarray, threshold: float,
                      w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else w
    pred = prob >= threshold
    pos, neg = y == 1, y == 0
    tpr = np.sum(w[pos & pred]) / max(np.sum(w[pos]), 1e-12)
    tnr = np.sum(w[neg & ~pred]) / max(np.sum(w[neg]), 1e-12)
    return float((tpr + tnr) / 2)


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.40, 0.96, 0.01), 2)
    rows = []
    for t in grid:
        pred_pos = prob >= t
        tp_min = np.sum((y == 0) & ~pred_pos)  # minoritaria = cross_owner 0, prevista se prob < t
        pred_min = np.sum(~pred_pos)
        rows.append({
            "threshold": float(t), "balanced_accuracy": balanced_accuracy(y, prob, t),
            "minority_precision": float(tp_min / max(pred_min, 1)),
            "minority_recall": float(tp_min / max(np.sum(y == 0), 1)),
        })
    table = pd.DataFrame(rows)
    best = float(table.loc[table["balanced_accuracy"].idxmax(), "threshold"])
    return best, table


def confusion(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, int]:
    pred_pos = prob >= threshold
    return {
        "true_cross_owner_pred_cross_owner": int(np.sum((y == 1) & pred_pos)),
        "true_cross_owner_pred_single_owner": int(np.sum((y == 1) & ~pred_pos)),
        "true_single_owner_pred_cross_owner": int(np.sum((y == 0) & pred_pos)),
        "true_single_owner_pred_single_owner": int(np.sum((y == 0) & ~pred_pos)),
    }


def oof_classification(frame: pd.DataFrame, seed: int = 2027) -> pd.DataFrame:
    cfg = _config("classification")
    parts = []
    for repeat, fold, train, valid in _folds(frame, "owner", 3, 3, seed):
        pipe = pipeline("classification", cfg["family"], cfg["params"]).fit(
            train[CLEAN_FULL], _target(train, "classification"))
        parts.append(pd.DataFrame({
            "repeat": repeat, "fold": fold, "content_hash": valid["content_hash"].to_numpy(),
            "y": valid["cross_owner"].to_numpy(dtype=int),
            "prob": _predict("classification", pipe, valid[CLEAN_FULL])}))
        print(f"[oof] repeticao {repeat} dobra {fold} concluida", flush=True)
    return pd.concat(parts, ignore_index=True)


def threshold_analysis(frame: pd.DataFrame, out: Path) -> dict[str, object]:
    oof = oof_classification(frame)
    oof.to_csv(out / "dev_oof_predictions.csv", index=False)
    y, prob = oof["y"].to_numpy(), oof["prob"].to_numpy()
    best, table = choose_threshold(y, prob)
    table.to_csv(out / "threshold_dev_oof.csv", index=False)
    per_repeat = [average_precision_score(1 - g["y"], 1 - g["prob"])
                  for _, g in oof.groupby("repeat")]
    result: dict[str, object] = {
        "rule": "maximize balanced accuracy on development out-of-fold predictions",
        "threshold": best,
        "dev_balanced_accuracy_at_threshold": float(table.loc[table.threshold == best,
                                                              "balanced_accuracy"].iloc[0]),
        "dev_balanced_accuracy_at_0.5": balanced_accuracy(y, prob, 0.5),
        "dev_minority_prevalence": float(1 - y.mean()),
        "dev_minority_pr_auc_mean_over_repeats": float(np.mean(per_repeat)),
        "dev_confusion_at_threshold": confusion(y, prob, best),
    }
    if HOLDOUT_PRED.exists():
        hold = pd.read_csv(HOLDOUT_PRED)
        yh, ph = hold["y"].to_numpy(), hold["prob"].to_numpy()
        owners = hold["repo_owner"].astype(str).to_numpy()
        ba = balanced_accuracy(yh, ph, best)
        boot_ba = _cluster_bootstrap(owners, lambda w: balanced_accuracy(yh, ph, best, w), 2000,
                                     2027)
        boot_ap = _cluster_bootstrap(
            owners, lambda w: average_precision_score(1 - yh, 1 - ph, sample_weight=w), 2000,
            2027)
        conf = confusion(yh, ph, best)
        minority_pred = conf["true_single_owner_pred_single_owner"] + conf[
            "true_cross_owner_pred_single_owner"]
        result["holdout"] = {
            "balanced_accuracy": ba,
            "balanced_accuracy_ci95": [float(np.percentile(boot_ba, 2.5)),
                                       float(np.percentile(boot_ba, 97.5))],
            "minority_prevalence": float(1 - yh.mean()),
            "minority_pr_auc": float(average_precision_score(1 - yh, 1 - ph)),
            "minority_pr_auc_ci95": [float(np.percentile(boot_ap, 2.5)),
                                     float(np.percentile(boot_ap, 97.5))],
            "minority_precision": conf["true_single_owner_pred_single_owner"]
            / max(minority_pred, 1),
            "minority_recall": conf["true_single_owner_pred_single_owner"]
            / max(conf["true_single_owner_pred_single_owner"]
                  + conf["true_single_owner_pred_cross_owner"], 1),
            "confusion": conf,
            "note": "threshold fixed on development; descriptive reading of the saved "
            "holdout predictions, not a new evaluation",
        }
        precision, recall, _ = precision_recall_curve(1 - yh, 1 - ph)
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
            out / "holdout_minority_pr_curve.csv", index=False)
    return result


def interpretation_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _frame()
    pdp = pd.concat([pdp_oof(t, frame) for t in ("classification", "regression")],
                    ignore_index=True)
    pdp.to_csv(args.output_dir / "pdp_oof.csv", index=False)
    result = threshold_analysis(frame, args.output_dir)
    (args.output_dir / "threshold_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_plain) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_plain))
