"""Modelos para ``owner_count`` e ``cross_owner`` com validacao agrupada por dono.

Protocolo (somente desenvolvimento; teste externo nao lido):

* repeticoes ``repeats`` x ``folds`` de ``StratifiedGroupKFold`` agrupado por dono do
  repositorio representativo (ou por repositorio, para medir o vazamento do agrupamento antigo);
* visoes de atributos *limpas* (sem ``location_class``, ``history_fetched``, ``commit_count`` e
  ``p_claude_dir``, que dependem da regra de escolha do representante) e a visao com esses
  atalhos, apenas como achado;
* checagem de ``name``: por dobra, fracao de linhas de validacao cujo ``name`` aparece no
  treino e metricas no subconjunto sem sobreposicao de ``name``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skillreuse.benchmark import _metrics, default_regressor, paired_test_table
from skillreuse.models import STRUCTURAL_NUMERIC_FEATURES
from skillreuse.owner import load_owner_frame

DEFAULT_OUTPUT_DIR = Path("results/owner/models")
CATEGORICAL = {"filename", "location_class", "repo_language", "repo_license"}
SKILL = [*STRUCTURAL_NUMERIC_FEATURES, "filename"]
REPO = ["log_stars", "log_forks", "repo_is_fork", "repo_age_days", "push_recency_days",
        "repo_active_days", "repo_language", "repo_license"]
CATALOG = ["log_skills_in_repo", "catalog_ge_10"]
PATH = ["path_depth", "path_is_root", "p_template", "p_plugin", "p_vendor", "p_docs", "p_pkg",
        "p_tool_dir"]
SHORTCUTS = ["location_class", "history_fetched", "commit_count", "p_claude_dir"]
CLEAN_FULL = SKILL + REPO + CATALOG + PATH
VIEWS: dict[str, list[str]] = {
    "skill": SKILL,
    "repo": REPO,
    "repo+catalog": REPO + CATALOG,
    "path+catalog": PATH + CATALOG,
    "clean_full": CLEAN_FULL,
    "clean_full+shortcuts": CLEAN_FULL + SHORTCUTS,
    "shortcuts_only": SHORTCUTS,
}
REG_MODELS = ("ridge_log", "random_forest_log")
CLF_MODELS = ("logistic", "random_forest")


def _preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [c for c in columns if c not in CATEGORICAL]
    categorical = [c for c in columns if c in CATEGORICAL]
    blocks: list[tuple[str, object, list[str]]] = []
    if numeric:
        blocks.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), numeric))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), categorical))
    return ColumnTransformer(blocks, remainder="drop")


def _dense(x: object) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _classifier(name: str) -> object:
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000)
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3, max_features="sqrt", n_jobs=-1,
        random_state=2027,
    )


def _clf_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    ok = len(np.unique(y)) > 1
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if ok else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if ok else float("nan"),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "prevalence": float(y.mean()),
    }


def _strata(frame: pd.DataFrame) -> np.ndarray:
    bins = pd.cut(frame["owner_count"], [0, 1, 2, 4, 9, np.inf], labels=False)
    return bins.astype(str).to_numpy()


def _folds(frame: pd.DataFrame, group_by: str, repeats: int, folds: int, seed: int):
    group_col = "repo_owner" if group_by == "owner" else "representative_repo_id"
    groups = frame[group_col].astype(str).to_numpy()
    strata = _strata(frame)
    for repeat in range(1, repeats + 1):
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed + repeat - 1
        )
        for fold, (tr, va) in enumerate(splitter.split(frame, y=strata, groups=groups), 1):
            train = frame.iloc[tr].reset_index(drop=True)
            valid = frame.iloc[va].reset_index(drop=True)
            if set(train[group_col]) & set(valid[group_col]):
                raise RuntimeError(f"Repeticao {repeat}, dobra {fold}: grupos compartilhados")
            yield repeat, fold, train, valid


def _subsets(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, np.ndarray]:
    seen = set(train["name_norm"][train["name_norm"] != ""])
    unseen = ~valid["name_norm"].isin(seen).to_numpy()
    return {"all": np.ones(len(valid), dtype=bool), "name_unseen": unseen}


def run(
    frame: pd.DataFrame, *, task: str, views: list[str], group_by: str, repeats: int,
    folds: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    overlap: list[dict[str, object]] = []
    for repeat, fold, train, valid in _folds(frame, group_by, repeats, folds, seed):
        subsets = _subsets(train, valid)
        overlap.append({
            "repeat": repeat, "fold": fold, "validation_rows": len(valid),
            "name_empty_share": float((valid["name_norm"] == "").mean()),
            "name_seen_in_train_share": float(1 - subsets["name_unseen"].mean()),
        })
        if task == "regression":
            y_log = train["log_owner_count"].to_numpy(dtype=float)
            y_valid = valid["owner_count"].to_numpy(dtype=float)
        else:
            y_log = train["cross_owner"].to_numpy(dtype=int)
            y_valid = valid["cross_owner"].to_numpy(dtype=int)
        preds: dict[str, np.ndarray] = {}
        if task == "regression":
            for name, strategy in (("dummy_mean_log", "mean"), ("dummy_median_log", "median")):
                dummy = DummyRegressor(strategy=strategy).fit(train[["filename"]], y_log)
                preds[name] = dummy.predict(valid[["filename"]])
        else:
            dummy = DummyClassifier(strategy="prior").fit(train[["filename"]], y_log)
            preds["dummy_prior"] = dummy.predict_proba(valid[["filename"]])[:, 1]
        for view in views:
            columns = VIEWS[view]
            pre = _preprocessor(columns)
            x_train = pre.fit_transform(train[columns])
            x_valid = pre.transform(valid[columns])
            names = REG_MODELS if task == "regression" else CLF_MODELS
            for model in names:
                dense = model not in ("ridge_log", "logistic")
                xt, xv = (_dense(x_train), _dense(x_valid)) if dense else (x_train, x_valid)
                if task == "regression":
                    est = default_regressor(model).fit(xt, y_log)
                    preds[f"{view}|{model}"] = est.predict(xv)
                else:
                    est = _classifier(model).fit(xt, y_log)
                    preds[f"{view}|{model}"] = est.predict_proba(xv)[:, 1]
        for config, pred in preds.items():
            for subset, mask in subsets.items():
                if mask.sum() < 30:
                    continue
                base = {"repeat": repeat, "fold": fold, "subset": subset, "config": config,
                        "train_rows": len(train), "validation_rows": int(mask.sum())}
                if task == "regression":
                    values = _metrics(y_valid[mask], np.maximum(np.expm1(pred[mask]), 0.0))
                else:
                    values = _clf_metrics(y_valid[mask], pred[mask])
                records.append({**base, **values})
        print(f"[{task}/{group_by}] repeticao {repeat} dobra {fold} concluida", flush=True)
    return pd.DataFrame(records), pd.DataFrame(overlap)


def _pairs(task: str, views: list[str]) -> list[tuple[str, str]]:
    names = REG_MODELS if task == "regression" else CLF_MODELS
    dummy = "dummy_mean_log" if task == "regression" else "dummy_prior"
    pairs: list[tuple[str, str]] = []
    for m in names:
        for v in views:
            pairs.append((f"{v}|{m}", dummy))
        if "clean_full" in views:
            pairs += [(f"clean_full|{m}", f"{v}|{m}") for v in views
                      if v not in ("clean_full", "clean_full+shortcuts", "shortcuts_only")]
        if {"clean_full", "clean_full+shortcuts"} <= set(views):
            pairs.append((f"clean_full+shortcuts|{m}", f"clean_full|{m}"))
    return pairs


def summarize(folds_df: pd.DataFrame, task: str) -> pd.DataFrame:
    cols = (["rmsle", "mae", "mae_log1p", "spearman"] if task == "regression"
            else ["roc_auc", "pr_auc", "brier", "log_loss"])
    out = folds_df.groupby(["subset", "config"], as_index=False)[cols].agg(["mean", "std"])
    out.columns = ["_".join(p for p in c if p).rstrip("_") for c in out.columns]
    key = "rmsle_mean" if task == "regression" else "roc_auc_mean"
    return out.sort_values(["subset", key], ascending=[True, task == "regression"])


def paired(folds_df: pd.DataFrame, task: str, views: list[str]) -> pd.DataFrame:
    tables = []
    df = folds_df.copy()
    metrics: tuple[str, ...] = ("rmsle", "mae")
    if task == "classification":
        df["neg_roc_auc"] = -df["roc_auc"]
        df["neg_pr_auc"] = -df["pr_auc"]
        metrics = ("neg_roc_auc", "neg_pr_auc")
    for subset, part in df.groupby("subset"):
        configs = set(part["config"])
        pairs = [(a, b) for a, b in _pairs(task, views) if a in configs and b in configs]
        table = paired_test_table(part, pairs, label="config", metrics=metrics)
        table.insert(0, "subset", subset)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def owner_models_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("regression", "classification", "both"),
                        default="both")
    parser.add_argument("--group-by", choices=("owner", "repo"), default="owner")
    parser.add_argument("--views", nargs="+", default=list(VIEWS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--exclude-owners-above", type=int, default=None,
                        help="remove donos com mais linhas que esse limite (sensibilidade)")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def owner_models_main() -> None:
    args = owner_models_parse_args()
    frame = load_owner_frame("train")
    if args.exclude_owners_above is not None:
        sizes = frame["repo_owner"].map(frame["repo_owner"].value_counts())
        frame = frame[sizes <= args.exclude_owners_above].reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=args.seed)
        frame = frame.reset_index(drop=True)
    tasks = ("regression", "classification") if args.task == "both" else (args.task,)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        folds_df, overlap = run(
            frame, task=task, views=args.views, group_by=args.group_by,
            repeats=args.repeats, folds=args.folds, seed=args.seed,
        )
        summary = summarize(folds_df, task)
        folds_df.to_csv(args.output_dir / f"{task}_folds.csv", index=False)
        overlap.to_csv(args.output_dir / f"{task}_name_overlap.csv", index=False)
        summary.to_csv(args.output_dir / f"{task}_summary.csv", index=False)
        paired(folds_df, task, args.views).to_csv(args.output_dir / f"{task}_paired.csv",
                                                  index=False)
        print(summary[summary["subset"] == "all"].round(4).to_string(index=False))
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "repeated StratifiedGroupKFold; external test not read",
        "group_by": args.group_by,
        "rows": int(len(frame)),
        "groups": int(frame["repo_owner" if args.group_by == "owner"
                            else "representative_repo_id"].nunique()),
        "repeats": args.repeats,
        "folds": args.folds,
        "seeds": [args.seed + r for r in range(args.repeats)],
        "exclude_owners_above": args.exclude_owners_above,
        "views": {v: VIEWS[v] for v in args.views},
        "targets": {"regression": "log1p(owner_count)", "classification": "owner_count >= 2"},
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
