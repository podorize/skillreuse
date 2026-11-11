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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from skillreuse.benchmark import _holm, _metrics, default_regressor, paired_test_table
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
    # ablacao: o modelo completo sem exatamente um bloco (so na analise de RQ3)
    "minus_skill": REPO + CATALOG + PATH,
    "minus_repo": SKILL + CATALOG + PATH,
    "minus_catalog": SKILL + REPO + PATH,
    "minus_path": SKILL + REPO + CATALOG,
}
ABLATION_VIEWS = ("minus_skill", "minus_repo", "minus_catalog", "minus_path")
DEFAULT_VIEWS = [v for v in VIEWS if v not in ABLATION_VIEWS]
REG_MODELS = ("ridge_log", "random_forest_log")
CLF_MODELS = ("logistic", "random_forest")


def _log1p_clip(x: object) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def _preprocessor(columns: list[str], log_cols: list[str] | None = None) -> ColumnTransformer:
    """``log_cols``: numericas assimetricas que recebem ``log1p`` (so para modelos lineares)."""
    log_cols = [c for c in (log_cols or []) if c in columns]
    numeric = [c for c in columns if c not in CATEGORICAL and c not in log_cols]
    categorical = [c for c in columns if c in CATEGORICAL]
    blocks: list[tuple[str, object, list[str]]] = []
    if log_cols:
        blocks.append(("num_log", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(_log1p_clip)),
            ("scale", StandardScaler(with_mean=False)),
        ]), log_cols))
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
    folds: int, seed: int, reg_target: str = "owner", log_skewed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    overlap: list[dict[str, object]] = []
    log_cols: list[str] | None = None
    if log_skewed:
        from skillreuse.feature_eda import skewed_columns

        log_cols = skewed_columns(frame)
    for repeat, fold, train, valid in _folds(frame, group_by, repeats, folds, seed):
        subsets = _subsets(train, valid)
        overlap.append({
            "repeat": repeat, "fold": fold, "validation_rows": len(valid),
            "name_empty_share": float((valid["name_norm"] == "").mean()),
            "name_seen_in_train_share": float(1 - subsets["name_unseen"].mean()),
        })
        if task == "regression":
            log_col, raw_col = (
                ("log_owner_count", "owner_count") if reg_target == "owner"
                else ("log_count_target", "count_target")
            )
            y_log = train[log_col].to_numpy(dtype=float)
            y_valid = valid[raw_col].to_numpy(dtype=float)
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
            pre = _preprocessor(columns, log_cols)
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
    parser.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--exclude-owners-above", type=int, default=None,
                        help="remove donos com mais linhas que esse limite (sensibilidade)")
    parser.add_argument("--cap-per-owner", type=int, default=None,
                        help="mantem no maximo N linhas por dono (ordem deterministica)")
    parser.add_argument("--tier", choices=("canonical", "skills-dir", "other"), default=None,
                        help="restringe a uma classe de localizacao do representante")
    parser.add_argument("--regression-target", choices=("owner", "repo"), default="owner")
    parser.add_argument("--log-skewed", action="store_true",
                        help="log1p nas numericas assimetricas (afeta so os modelos lineares)")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _prepare(args: argparse.Namespace) -> pd.DataFrame:
    frame = load_owner_frame("train")
    if args.tier is not None:
        frame = frame[frame["location_class"] == args.tier].reset_index(drop=True)
    if args.exclude_owners_above is not None:
        sizes = frame["repo_owner"].map(frame["repo_owner"].value_counts())
        frame = frame[sizes <= args.exclude_owners_above].reset_index(drop=True)
    if args.cap_per_owner is not None:
        frame = (
            frame.sort_values("content_hash").groupby("repo_owner", sort=False)
            .head(args.cap_per_owner).reset_index(drop=True)
        )
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=args.seed)
        frame = frame.reset_index(drop=True)
    return frame


def owner_models_main() -> None:
    args = owner_models_parse_args()
    frame = _prepare(args)
    tasks = ("regression", "classification") if args.task == "both" else (args.task,)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        folds_df, overlap = run(
            frame, task=task, views=args.views, group_by=args.group_by,
            repeats=args.repeats, folds=args.folds, seed=args.seed,
            reg_target=args.regression_target, log_skewed=args.log_skewed,
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
        "cap_per_owner": args.cap_per_owner,
        "tier": args.tier,
        "regression_target": args.regression_target,
        "log_skewed": args.log_skewed,
        "views": {v: VIEWS[v] for v in args.views},
        "targets": {"regression": "log1p(owner_count)", "classification": "owner_count >= 2"},
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Importancia por permutacao (grupos de atributos e colunas) no conjunto limpo
# ---------------------------------------------------------------------------

GROUPS = {"grupo:skill": SKILL, "grupo:repo": REPO, "grupo:catalog": CATALOG, "grupo:path": PATH}


def _pipeline_for(task: str, columns: list[str]) -> Pipeline:
    from sklearn.preprocessing import FunctionTransformer

    model = default_regressor("random_forest_log") if task == "regression" else _classifier(
        "random_forest")
    return Pipeline([
        ("pre", _preprocessor(columns)),
        ("dense", FunctionTransformer(_dense, accept_sparse=True)),
        ("model", model),
    ])


def run_importance(
    frame: pd.DataFrame, *, task: str, view: str, repeats: int, folds: int, seed: int,
    n_perm: int,
) -> pd.DataFrame:
    columns = VIEWS[view]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for repeat, fold, train, valid in _folds(frame, "owner", repeats, folds, seed):
        if task == "regression":
            y_tr = train["log_owner_count"].to_numpy(dtype=float)
            y_va = valid["log_owner_count"].to_numpy(dtype=float)
        else:
            y_tr = train["cross_owner"].to_numpy(dtype=int)
            y_va = valid["cross_owner"].to_numpy(dtype=int)
        pipe = _pipeline_for(task, columns).fit(train[columns], y_tr)

        def score(x: pd.DataFrame, pipe: Pipeline = pipe, y_va: np.ndarray = y_va) -> float:
            if task == "regression":
                pred = pipe.predict(x)
                return float(np.sqrt(np.mean((pred - y_va) ** 2)))  # RMSE em log1p == RMSLE
            return float(roc_auc_score(y_va, pipe.predict_proba(x)[:, 1]))

        base = score(valid[columns])
        for name, cols in {**GROUPS, **{c: [c] for c in columns}}.items():
            drops = []
            for _ in range(n_perm):
                shuffled = valid[columns].copy()
                shuffled[cols] = shuffled[cols].iloc[rng.permutation(len(valid))].to_numpy()
                s = score(shuffled)
                drops.append(s - base if task == "regression" else base - s)
            rows.append({"repeat": repeat, "fold": fold, "feature": name,
                         "metric_degradation": float(np.mean(drops)), "base_score": base})
        print(f"[importance/{task}] repeticao {repeat} dobra {fold} concluida", flush=True)
    return pd.DataFrame(rows)


def importance_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importancia por permutacao (conjunto limpo)")
    parser.add_argument("--view", choices=list(VIEWS), default="clean_full")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-perm", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--exclude-owners-above", type=int, default=None)
    parser.add_argument("--cap-per-owner", type=int, default=None)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/owner/importance"))
    return parser.parse_args()


def importance_main() -> None:
    args = importance_parse_args()
    frame = _prepare(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in ("regression", "classification"):
        raw = run_importance(frame, task=task, view=args.view, repeats=args.repeats,
                             folds=args.folds, seed=args.seed, n_perm=args.n_perm)
        raw.to_csv(args.output_dir / f"{task}_importance_folds.csv", index=False)
        summary = raw.groupby("feature")["metric_degradation"].agg(["mean", "std"]).reset_index()
        summary = summary.sort_values("mean", ascending=False)
        summary.to_csv(args.output_dir / f"{task}_importance.csv", index=False)
        print(summary.head(15).round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# Protocolo principal: hipoteses pre-especificadas e correcao de Holm
# ---------------------------------------------------------------------------

PRIMARY_CAP = 50
HYPOTHESES = [
    ("H1", "classification", "clean_full|random_forest", "dummy_prior", "neg_roc_auc",
     "atributos limpos preveem reuso entre donos (AUC > 0,5)"),
    ("H2", "regression", "clean_full|random_forest_log", "dummy_mean_log", "rmsle",
     "atributos limpos reduzem o RMSLE de owner_count frente a media de log1p"),
    ("H3a", "classification", "clean_full|random_forest", "repo+catalog|random_forest",
     "neg_roc_auc", "a skill acrescenta ao repositorio e catalogo (AUC)"),
    ("H3b", "classification", "clean_full|random_forest", "skill|random_forest",
     "neg_roc_auc", "repositorio e catalogo acrescentam a skill (AUC)"),
    ("H4", "regression", "clean_full+shortcuts|random_forest_log", "clean_full|random_forest_log",
     "rmsle", "os atalhos de coleta acrescentam aos atributos limpos (RMSLE)"),
]


def evaluate_hypotheses(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for hid, task, a, b, metric, text in HYPOTHESES:
        df = results[task].copy()
        df = df[df["subset"] == "all"]
        if task == "classification":
            df["neg_roc_auc"] = -df["roc_auc"]
        table = paired_test_table(df, [(a, b)], label="config", metrics=(metric,))
        col = "roc_auc" if task == "classification" else "rmsle"
        mean_a = float(df[df["config"] == a][col].mean())
        mean_b = float(df[df["config"] == b][col].mean())
        table.insert(0, "hypothesis", hid)
        table["description"] = text
        table["score_a"] = mean_a
        table["score_b"] = mean_b
        rows.append(table)
    out = pd.concat(rows, ignore_index=True)
    out["p_holm_family"] = _holm(out["p_nadeau_bengio"].tolist())
    out["reject_h0_at_0.05"] = out["p_holm_family"] < 0.05
    return out.drop(columns=["p_nb_holm"])


def primary_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocolo principal (paper/protocol.md)")
    parser.add_argument("--cap-per-owner", type=int, default=PRIMARY_CAP)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", type=Path, default=Path("results/owner/primary"))
    return parser.parse_args()


def primary_main() -> None:
    args = primary_parse_args()
    ns = argparse.Namespace(
        tier=None, exclude_owners_above=None, cap_per_owner=args.cap_per_owner,
        max_rows=None, seed=args.seed,
    )
    frame = _prepare(ns)
    views = list(DEFAULT_VIEWS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for task in ("regression", "classification"):
        folds_df, overlap = run(
            frame, task=task, views=views, group_by="owner", repeats=args.repeats,
            folds=args.folds, seed=args.seed,
        )
        results[task] = folds_df
        folds_df.to_csv(args.output_dir / f"{task}_folds.csv", index=False)
        overlap.to_csv(args.output_dir / f"{task}_name_overlap.csv", index=False)
        summarize(folds_df, task).to_csv(args.output_dir / f"{task}_summary.csv", index=False)
        paired(folds_df, task, views).to_csv(args.output_dir / f"{task}_paired_exploratory.csv",
                                             index=False)
    hypotheses = evaluate_hypotheses(results)
    hypotheses.to_csv(args.output_dir / "primary_hypotheses.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "paper/protocol.md",
        "cap_per_owner": args.cap_per_owner,
        "rows": int(len(frame)),
        "owners": int(frame["repo_owner"].nunique()),
        "repeats": args.repeats, "folds": args.folds,
        "seeds": [args.seed + r for r in range(args.repeats)],
        "hypotheses": [h[0] for h in HYPOTHESES],
        "family_correction": "Holm over the hypotheses, Nadeau-Bengio p-values",
        "external_test_read": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(hypotheses.drop(columns=["description"]).round(4).to_string(index=False))


def owner_overlap_main() -> None:
    """Fracao das linhas de validacao com algum dono de ocorrencia tambem presente no treino.

    A validacao agrupa pelo dono do repositorio representativo; o alvo, porem, conta todos os
    donos do conteudo. Mede-se quanto isso deixa de garantir disjuncao total de donos.
    """
    from skillreuse.owner import _connect

    parser = argparse.ArgumentParser(description=owner_overlap_main.__doc__)
    parser.add_argument("--cap-per-owner", type=int, default=PRIMARY_CAP)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", type=Path, default=Path("results/owner/primary"))
    args = parser.parse_args()
    ns = argparse.Namespace(tier=None, exclude_owners_above=None,
                            cap_per_owner=args.cap_per_owner, max_rows=None, seed=args.seed)
    frame = _prepare(ns)
    con = _connect()
    hashes = frame[["content_hash"]].drop_duplicates()
    con.register("h", hashes)
    occ = con.sql(
        "select distinct o.file_sha as content_hash, o.owner from o "
        "join h on h.content_hash = o.file_sha"
    ).df()
    owners = occ.groupby("content_hash")["owner"].agg(frozenset).to_dict()
    rows = []
    for repeat, fold, train, valid in _folds(frame, "owner", args.repeats, args.folds, args.seed):
        train_owners = set().union(*(owners[h] for h in train["content_hash"]))
        shared = [bool(owners[h] & train_owners) for h in valid["content_hash"]]
        only_rep = [h for h in valid["content_hash"] if len(owners[h]) > 1]
        rows.append({
            "repeat": repeat, "fold": fold, "validation_rows": len(valid),
            "share_with_shared_occurrence_owner": sum(shared) / len(valid),
            "multi_owner_rows": len(only_rep),
        })
    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "occurrence_owner_overlap.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("mean share:", round(float(out["share_with_shared_occurrence_owner"].mean()), 4))

    # holdout: donos de todas as ocorrencias contra os donos do desenvolvimento
    import pyarrow.parquet as pq

    from skillreuse.models import DEFAULT_SAMPLE

    holdout = pd.read_parquet("data/processed/holdout_sample.parquet",
                              columns=["content_hash"])
    dev = pq.read_table(DEFAULT_SAMPLE, columns=["content_hash", "repository_count",
                                                 "repo_owner"]).to_pandas()
    dev = dev[dev["repository_count"] >= 2]
    con.register("ho", holdout)
    con.register("dev", dev[["content_hash"]])
    ho_owners = con.sql(
        "select distinct o.file_sha as h, o.owner from o join ho on ho.content_hash = o.file_sha"
    ).df().groupby("h")["owner"].agg(set)
    dev_all = set(con.sql(
        "select distinct o.owner from o join dev on dev.content_hash = o.file_sha"
    ).df()["owner"])
    dev_rep = set(dev["repo_owner"].dropna())
    summary = {
        "holdout_rows": int(len(ho_owners)),
        "share_any_occurrence_owner_in_dev_representative_owners":
            float(ho_owners.map(lambda x: bool(x & dev_rep)).mean()),
        "share_any_occurrence_owner_in_dev_occurrence_owners":
            float(ho_owners.map(lambda x: bool(x & dev_all)).mean()),
        "cv_mean_share_shared_occurrence_owner":
            float(out["share_with_shared_occurrence_owner"].mean()),
    }
    target = args.output_dir / "occurrence_owner_overlap.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)
