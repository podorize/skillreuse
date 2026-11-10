"""Tuning aninhado no desenvolvimento, congelamento do modelo final e avaliacao unica no holdout.

Etapa 1 (``skillreuse final-model``): para cada tarefa (``cross_owner`` e ``log1p(owner_count)``),
seis familias x {configuracao padrao, configuracao buscada} sao comparadas por validacao
aninhada (externa: repeticoes x dobras agrupadas por dono; interna: ``RandomizedSearchCV`` com
dobras agrupadas por dono). Escolhe-se a combinacao com melhor media externa (AUC ou RMSLE),
refaz-se a busca em todo o desenvolvimento quando a escolhida e "tuned" e grava-se
``final_config.json``. O holdout nao e lido.

Etapa 2 (``skillreuse holdout-eval``): treina no desenvolvimento inteiro com a configuracao
congelada, avalia **uma unica vez** no holdout e estima ICs por bootstrap de donos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from skillreuse.benchmark import _holm, _metrics, default_regressor, paired_test_table
from skillreuse.owner import load_holdout_frame
from skillreuse.owner_models import (
    CLEAN_FULL,
    _clf_metrics,
    _dense,
    _folds,
    _prepare,
    _preprocessor,
    _strata,
)
from skillreuse.tuning import SEARCH_SPACES as REG_SPACES

OUTPUT_DIR = Path("results/owner/final")
HOLDOUT_DIR = Path("results/owner/holdout")
FAMILIES = ("linear", "knn", "decision_tree", "random_forest", "gradient_boosting",
            "hist_gradient_boosting")
REG_NAME = {"linear": "ridge_log", "knn": "knn_log", "decision_tree": "decision_tree_log",
            "random_forest": "random_forest_log", "gradient_boosting": "gradient_boosting_log",
            "hist_gradient_boosting": "hist_gradient_boosting_log"}

CLF_SPACES: dict[str, dict[str, object]] = {
    "linear": {"C": loguniform(1e-3, 1e2)},
    "knn": {"n_neighbors": randint(5, 101), "weights": ["uniform", "distance"], "p": [1, 2]},
    "decision_tree": {"max_depth": randint(3, 25), "min_samples_leaf": randint(1, 61),
                      "min_samples_split": randint(2, 40)},
    "random_forest": {"n_estimators": [100, 200, 300], "max_depth": randint(5, 26),
                      "min_samples_leaf": randint(1, 31), "max_features": ["sqrt", 0.5, 1.0]},
    "gradient_boosting": {"n_estimators": randint(100, 301),
                          "learning_rate": loguniform(0.02, 0.2), "max_depth": randint(2, 5),
                          "subsample": uniform(0.6, 0.4), "min_samples_leaf": randint(1, 41)},
    "hist_gradient_boosting": {"learning_rate": loguniform(0.02, 0.2),
                               "max_iter": randint(100, 501), "max_leaf_nodes": randint(8, 65),
                               "min_samples_leaf": randint(5, 101),
                               "l2_regularization": loguniform(1e-3, 10.0)},
}


def make_estimator(task: str, family: str, threads: int = -1) -> object:
    """Configuracao padrao de cada familia (as mesmas do protocolo principal)."""
    if task == "regression":
        est = default_regressor(REG_NAME[family])
        if threads == 1 and "n_jobs" in est.get_params():
            est.set_params(n_jobs=1)
        return est
    models = {
        "linear": lambda: LogisticRegression(C=1.0, max_iter=2000),
        "knn": lambda: KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=threads),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=2027),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, max_features="sqrt",
            n_jobs=threads, random_state=2027),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=2027),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=1.0,
            random_state=2027),
    }
    return models[family]()


def spaces(task: str, family: str) -> dict[str, object]:
    return REG_SPACES[REG_NAME[family]] if task == "regression" else CLF_SPACES[family]


def pipeline(task: str, family: str, params: dict[str, object] | None = None,
             threads: int = -1) -> Pipeline:
    dense = "passthrough" if family == "linear" else FunctionTransformer(
        _dense, accept_sparse=True)
    pipe = Pipeline([("pre", _preprocessor(CLEAN_FULL)), ("dense", dense),
                     ("model", make_estimator(task, family, threads))])
    if params:
        pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
    return pipe


def _target(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "regression":
        return frame["log_owner_count"].to_numpy(dtype=float)
    return frame["cross_owner"].to_numpy(dtype=int)


def _predict(task: str, pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return pipe.predict(x) if task == "regression" else pipe.predict_proba(x)[:, 1]


def _score(task: str, frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    if task == "regression":
        return _metrics(frame["owner_count"].to_numpy(dtype=float),
                        np.maximum(np.expm1(pred), 0.0))
    return _clf_metrics(frame["cross_owner"].to_numpy(dtype=int), pred)


def _plain(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


N_JOBS = 4  # a maquina de desenvolvimento tem pouca memoria; ajustavel por --n-jobs


def _search(task: str, family: str, train: pd.DataFrame, *, n_iter: int, inner_folds: int,
            seed: int) -> RandomizedSearchCV:
    splitter = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    cv = list(splitter.split(train, y=_strata(train),
                             groups=train["repo_owner"].astype(str).to_numpy()))
    search = RandomizedSearchCV(
        pipeline(task, family, threads=1),
        {f"model__{k}": v for k, v in spaces(task, family).items()},
        n_iter=n_iter, cv=cv, n_jobs=N_JOBS, random_state=2027, refit=True, error_score="raise",
        scoring="roc_auc" if task == "classification" else "neg_root_mean_squared_error",
    )
    return search.fit(train[CLEAN_FULL], _target(train, task))


def nested(task: str, frame: pd.DataFrame, *, repeats: int, folds: int, inner_folds: int,
           n_iter: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat, fold, train, valid in _folds(frame, "owner", repeats, folds, seed):
        base = {"repeat": repeat, "fold": fold, "train_rows": len(train),
                "validation_rows": len(valid), "task": task}
        y_tr = _target(train, task)
        dummy = (DummyRegressor(strategy="mean") if task == "regression"
                 else DummyClassifier(strategy="prior")).fit(train[["filename"]], y_tr)
        pred = (dummy.predict(valid[["filename"]]) if task == "regression"
                else dummy.predict_proba(valid[["filename"]])[:, 1])
        rows.append({**base, "config": "baseline", **_score(task, valid, pred)})
        for family in FAMILIES:
            default = pipeline(task, family).fit(train[CLEAN_FULL], y_tr)
            rows.append({**base, "config": f"{family}|default",
                         **_score(task, valid, _predict(task, default, valid[CLEAN_FULL]))})
            search = _search(task, family, train, n_iter=n_iter, inner_folds=inner_folds,
                             seed=seed + repeat - 1)
            tuned = _predict(task, search.best_estimator_, valid[CLEAN_FULL])
            rows.append({
                **base, "config": f"{family}|tuned", **_score(task, valid, tuned),
                "inner_score": float(abs(search.best_score_)),
                "best_params": json.dumps({k.removeprefix("model__"): _plain(v) for k, v in
                                           search.best_params_.items()}, sort_keys=True)})
            key = "roc_auc" if task == "classification" else "rmsle"
            print(f"[{task}] rep {repeat} dobra {fold} {family}: "
                  f"default {rows[-2][key]:.4f} -> tuned {rows[-1][key]:.4f}", flush=True)
    return pd.DataFrame(rows)


def final_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cap-per-owner", type=int, default=50)
    parser.add_argument("--tasks", nargs="+", default=["classification", "regression"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def final_main() -> None:
    global N_JOBS
    args = final_parse_args()
    N_JOBS = args.n_jobs
    ns = argparse.Namespace(tier=None, exclude_owners_above=None,
                            cap_per_owner=args.cap_per_owner, max_rows=args.max_rows,
                            seed=args.seed)
    frame = _prepare(ns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_hash = hashlib.sha256("".join(sorted(frame["content_hash"])).encode()).hexdigest()
    for task in args.tasks:
        started = time.monotonic()
        folds_df = nested(task, frame, repeats=args.repeats, folds=args.folds,
                          inner_folds=args.inner_folds, n_iter=args.n_iter, seed=args.seed)
        metric = "roc_auc" if task == "classification" else "rmsle"
        folds_df.to_csv(args.output_dir / f"{task}_nested_folds.csv", index=False)
        summary = (folds_df.groupby("config")[metric].agg(["mean", "std"]).reset_index()
                   .sort_values("mean", ascending=(task == "regression")))
        summary.to_csv(args.output_dir / f"{task}_nested_summary.csv", index=False)
        candidates = summary[summary["config"] != "baseline"]
        best = str(candidates.iloc[0]["config"])
        family, variant = best.split("|")
        paired_df = folds_df.copy()
        if task == "classification":
            paired_df["neg_roc_auc"] = -paired_df["roc_auc"]
        pm = ("neg_roc_auc",) if task == "classification" else ("rmsle",)
        pairs = ([(best, c) for c in candidates["config"] if c != best]
                 + [(best, "baseline")])
        tests = paired_test_table(paired_df, pairs, label="config", metrics=pm)
        tests["p_nb_holm"] = _holm(tests["p_nadeau_bengio"].tolist())
        tests.to_csv(args.output_dir / f"{task}_selection_tests.csv", index=False)
        params: dict[str, object] = {}
        cv_score = None
        if variant == "tuned":
            search = _search(task, family, frame, n_iter=args.n_iter,
                             inner_folds=args.inner_folds, seed=args.seed)
            params = {k.removeprefix("model__"): _plain(v) for k, v in
                      search.best_params_.items()}
            cv_score = float(abs(search.best_score_))
        config = {
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "task": task, "family": family, "variant": variant, "params": params,
            "features": CLEAN_FULL, "selection_metric": metric,
            "nested_mean_of_selected": float(candidates.iloc[0]["mean"]),
            "nested_mean_baseline": float(summary[summary["config"] == "baseline"]["mean"].iloc[0]),
            "full_dev_inner_cv_score": cv_score,
            "dev_rows": int(len(frame)), "dev_frame_sha256": frame_hash,
            "cap_per_owner": args.cap_per_owner, "seed": args.seed,
            "repeats": args.repeats, "folds": args.folds, "n_iter": args.n_iter,
            "holdout_read": False,
            "runtime_seconds": round(time.monotonic() - started, 1),
        }
        (args.output_dir / f"final_config_{task}.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(summary.round(4).to_string(index=False))
        print("congelado:", config["task"], family, variant, params)


# ---------------------------------------------------------------------------
# Avaliacao unica no holdout
# ---------------------------------------------------------------------------


def _cluster_bootstrap(owners: np.ndarray, fn, n_boot: int, seed: int) -> np.ndarray:
    codes, inverse = np.unique(owners, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))
        out[b] = fn(counts[inverse].astype(float))
    return out


def eval_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliacao unica no holdout")
    parser.add_argument("--config-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--allow-reevaluation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HOLDOUT_DIR)
    return parser.parse_args()


def holdout_eval_main() -> None:
    args = eval_parse_args()
    lock = args.output_dir / "evaluation_lock.json"
    if lock.exists() and not args.allow_reevaluation:
        raise SystemExit(f"Holdout ja avaliado ({lock}). Reavaliar exige --allow-reevaluation e "
                         "deve ser declarado no relatorio.")
    configs = {t: json.loads((args.config_dir / f"final_config_{t}.json").read_text())
               for t in ("classification", "regression")}
    ns = argparse.Namespace(tier=None, exclude_owners_above=None, cap_per_owner=50,
                            max_rows=None, seed=configs["classification"]["seed"])
    dev = _prepare(ns)
    hold = load_holdout_frame()
    owners = hold["repo_owner"].astype(str).to_numpy()
    result: dict[str, object] = {"evaluated_at_utc": datetime.now(UTC).isoformat(),
                                 "holdout_rows": int(len(hold)),
                                 "holdout_owners": int(len(set(owners)))}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = configs["classification"]
    pipe = pipeline("classification", cfg["family"], cfg["params"]).fit(
        dev[CLEAN_FULL], _target(dev, "classification"))
    prob = _predict("classification", pipe, hold[CLEAN_FULL])
    y = hold["cross_owner"].to_numpy(dtype=int)
    point = _clf_metrics(y, prob)
    boot_auc = _cluster_bootstrap(owners, lambda w: roc_auc_score(y, prob, sample_weight=w),
                                  args.n_boot, 2027)
    boot_pr = _cluster_bootstrap(
        owners, lambda w: average_precision_score(y, prob, sample_weight=w), args.n_boot, 2027)
    prior = float(_target(dev, "classification").mean())
    result["classification"] = {
        "model": f"{cfg['family']}|{cfg['variant']}", **point,
        "roc_auc_ci95": [float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5))],
        "pr_auc_ci95": [float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5))],
        "prior_from_dev": prior,
        "brier_baseline": float(brier_score_loss(y, np.full(len(y), prior))),
        "log_loss_baseline": float(log_loss(y, np.full(len(y), prior), labels=[0, 1])),
        "criterion_auc_lower_ci_gt_0.5": bool(np.percentile(boot_auc, 2.5) > 0.5),
    }
    pd.DataFrame({"content_hash": hold["content_hash"], "repo_owner": owners, "y": y,
                  "prob": prob}).to_csv(args.output_dir / "classification_predictions.csv",
                                        index=False)

    cfg = configs["regression"]
    pipe = pipeline("regression", cfg["family"], cfg["params"]).fit(
        dev[CLEAN_FULL], _target(dev, "regression"))
    pred = np.maximum(_predict("regression", pipe, hold[CLEAN_FULL]), 0.0)
    base = float(_target(dev, "regression").mean())
    y_log = hold["log_owner_count"].to_numpy(dtype=float)
    sq_model, sq_base = (pred - y_log) ** 2, (base - y_log) ** 2

    def rmsle_diff(w: np.ndarray) -> float:
        return float(np.sqrt(np.average(sq_model, weights=w)) - np.sqrt(np.average(sq_base,
                                                                                    weights=w)))

    boot = _cluster_bootstrap(owners, rmsle_diff, args.n_boot, 2027)
    point = _score("regression", hold, pred)
    result["regression"] = {
        "model": f"{cfg['family']}|{cfg['variant']}", **point,
        "rmsle_baseline_mean_log1p": float(np.sqrt(sq_base.mean())),
        "rmsle_diff_model_minus_baseline": rmsle_diff(np.ones(len(y_log))),
        "rmsle_diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "criterion_diff_upper_ci_lt_0": bool(np.percentile(boot, 97.5) < 0),
    }
    pd.DataFrame({"content_hash": hold["content_hash"], "repo_owner": owners,
                  "y_log": y_log, "pred_log": pred}).to_csv(
        args.output_dir / "regression_predictions.csv", index=False)
    result["success"] = bool(
        result["classification"]["criterion_auc_lower_ci_gt_0.5"]  # type: ignore[index]
        and result["regression"]["criterion_diff_upper_ci_lt_0"])  # type: ignore[index]
    result["n_boot"] = args.n_boot
    (args.output_dir / "holdout_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lock.write_text(json.dumps({"evaluated_at_utc": result["evaluated_at_utc"],
                                "configs": {t: c["family"] + "|" + c["variant"]
                                            for t, c in configs.items()}}, indent=2) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
