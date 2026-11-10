"""Busca de hiperparametros aninhada, validacao agrupada repetida e testes pareados.

Protocolo (somente o conjunto de desenvolvimento; o teste externo nao e lido):

* validacao externa: ``repeats`` x ``folds`` com ``StratifiedGroupKFold`` por
  ``representative_repo_id`` (sementes ``seed + r``), mesmas particoes para todos os modelos;
* busca interna: ``RandomizedSearchCV`` com ``inner_folds`` dobras agrupadas dentro do treino
  de cada dobra externa; o pre-processamento e reajustado em cada dobra interna;
* a nota externa de cada modelo "tuned" vem de dobras que a busca nunca viu;
* para cada modelo tambem se avalia a configuracao fixa original ("default") e duas
  referencias constantes (mediana e media de ``log1p(repository_count)``);
* ao final, a busca e repetida em todo o desenvolvimento para fixar os hiperparametros.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from skillreuse.benchmark import (
    REGRESSION_MODELS,
    _holm,
    _metrics,
    default_regressor,
    paired_test_table,
    subset_frame,
)
from skillreuse.models import build_preprocessor, load_count_frame, predictor_view

DEFAULT_OUTPUT_DIR = Path("results/discipline_benchmark/tuned_nested")
VIEWS = {
    "repository_only": {"use_text": False, "use_structural": False, "use_repository": True},
    "structural_only": {"use_text": False, "use_structural": True, "use_repository": False},
    "structural_repository": {"use_text": False, "use_structural": True, "use_repository": True},
}
DUMMIES = {"dummy_median_log": "median", "dummy_mean_log": "mean"}

SEARCH_SPACES: dict[str, dict[str, object]] = {
    "ridge_log": {"alpha": loguniform(1e-2, 1e3)},
    "knn_log": {
        "n_neighbors": randint(5, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "decision_tree_log": {
        "max_depth": randint(3, 25),
        "min_samples_leaf": randint(1, 61),
        "min_samples_split": randint(2, 40),
    },
    "random_forest_log": {
        "n_estimators": [100, 200, 300],
        "max_depth": randint(5, 26),
        "min_samples_leaf": randint(1, 31),
        "max_features": ["sqrt", 0.5, 1.0],
    },
    "gradient_boosting_log": {
        "n_estimators": randint(100, 401),
        "learning_rate": loguniform(0.02, 0.2),
        "max_depth": randint(2, 6),
        "subsample": uniform(0.6, 0.4),
        "min_samples_leaf": randint(1, 41),
        "loss": ["squared_error", "huber"],
    },
    "hist_gradient_boosting_log": {
        "learning_rate": loguniform(0.02, 0.2),
        "max_iter": randint(100, 501),
        "max_leaf_nodes": randint(8, 65),
        "min_samples_leaf": randint(5, 101),
        "l2_regularization": loguniform(1e-3, 10.0),
    },
}


def _to_dense(matrix: object) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _pipeline(model_name: str, predictors: pd.DataFrame, config: dict[str, bool]) -> Pipeline:
    """Pre-processamento + regressor; o ajuste ocorre sempre sobre o treino de cada dobra."""
    dense = (
        FunctionTransformer(_to_dense, accept_sparse=True)
        if model_name != "ridge_log"
        else "passthrough"
    )
    return Pipeline(
        [
            ("pre", build_preprocessor(predictors, **config)),
            ("dense", dense),
            ("model", default_regressor(model_name)),
        ]
    )


def _single_thread(pipe: Pipeline) -> Pipeline:
    """A busca paraleliza as candidatas; os estimadores internos usam um unico nucleo."""
    if "n_jobs" in pipe.named_steps["model"].get_params():
        pipe.set_params(model__n_jobs=1)
    return pipe


def _inner_splits(train: pd.DataFrame, folds: int, seed: int) -> list[tuple[np.ndarray, ...]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(
        splitter.split(
            train,
            y=train["split_stratum"].astype(str).to_numpy(),
            groups=train["representative_repo_id"].astype(str).to_numpy(),
        )
    )


def _search(
    model_name: str,
    predictors: pd.DataFrame,
    y_log: np.ndarray,
    train_frame: pd.DataFrame,
    *,
    config: dict[str, bool],
    n_iter: int,
    inner_folds: int,
    seed: int,
) -> RandomizedSearchCV:
    pipe = _single_thread(_pipeline(model_name, predictors, config))
    search = RandomizedSearchCV(
        pipe,
        {f"model__{k}": v for k, v in SEARCH_SPACES[model_name].items()},
        n_iter=n_iter,
        cv=_inner_splits(train_frame, inner_folds, seed),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=2027,
        refit=True,
        error_score="raise",
    )
    return search.fit(predictors, y_log)


def _score(y_counts: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    return _metrics(y_counts, np.maximum(np.expm1(log_pred), 0.0))


def run_nested(
    frame: pd.DataFrame,
    *,
    view: str,
    models: tuple[str, ...],
    repeats: int,
    folds: int,
    inner_folds: int,
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    config = VIEWS[view]
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].astype(str).to_numpy()
    records: list[dict[str, object]] = []
    for repeat in range(1, repeats + 1):
        outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat - 1)
        for fold, (tr_idx, va_idx) in enumerate(outer.split(frame, y=strata, groups=groups), 1):
            train = frame.iloc[tr_idx].reset_index(drop=True)
            valid = frame.iloc[va_idx].reset_index(drop=True)
            if set(train["representative_repo_id"]) & set(valid["representative_repo_id"]):
                raise RuntimeError(f"Repeticao {repeat}, dobra {fold}: grupos compartilhados")
            x_train = predictor_view(train, **config)
            x_valid = predictor_view(valid, **config)
            y_log = train["log_count_target"].to_numpy(dtype=float)
            y_valid = valid["count_target"].to_numpy(dtype=float)
            base = {
                "repeat": repeat,
                "fold": fold,
                "view": view,
                "train_rows": len(train),
                "validation_rows": len(valid),
            }
            for name, strategy in DUMMIES.items():
                dummy = DummyRegressor(strategy=strategy).fit(x_train, y_log)
                records.append(
                    {**base, "model": name, "variant": "baseline",
                     **_score(y_valid, dummy.predict(x_valid))}
                )
            for model_name in models:
                start = time.monotonic()
                default = _pipeline(model_name, x_train, config).fit(x_train, y_log)
                records.append(
                    {**base, "model": model_name, "variant": "default",
                     "fit_seconds": time.monotonic() - start,
                     **_score(y_valid, default.predict(x_valid))}
                )
                start = time.monotonic()
                search = _search(
                    model_name, x_train, y_log, train,
                    config=config, n_iter=n_iter, inner_folds=inner_folds,
                    seed=seed + repeat - 1,
                )
                records.append(
                    {**base, "model": model_name, "variant": "tuned",
                     "fit_seconds": time.monotonic() - start,
                     "inner_rmsle": float(-search.best_score_),
                     "best_params": json.dumps(
                         {k.removeprefix("model__"): _plain(v) for k, v in
                          search.best_params_.items()}, sort_keys=True),
                     **_score(y_valid, search.predict(x_valid))}
                )
                print(
                    f"[{view}] rep {repeat} dobra {fold} {model_name}: "
                    f"default {records[-2]['rmsle']:.4f} -> tuned {records[-1]['rmsle']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(records)


def _plain(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def final_search(
    frame: pd.DataFrame, *, view: str, models: tuple[str, ...], inner_folds: int,
    n_iter: int, seed: int,
) -> dict[str, object]:
    """Fixa hiperparametros buscando em todo o desenvolvimento (sem o teste externo)."""
    config = VIEWS[view]
    predictors = predictor_view(frame, **config)
    y_log = frame["log_count_target"].to_numpy(dtype=float)
    best: dict[str, object] = {}
    for model_name in models:
        search = _search(
            model_name, predictors, y_log, frame,
            config=config, n_iter=n_iter, inner_folds=inner_folds, seed=seed,
        )
        best[model_name] = {
            "cv_rmsle": float(-search.best_score_),
            "params": {k.removeprefix("model__"): _plain(v) for k, v in
                       search.best_params_.items()},
        }
        print(f"[{view}] final {model_name}: cv_rmsle={-search.best_score_:.4f}", flush=True)
    return best


def summarize(folds_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rmsle", "mae", "mae_log1p", "spearman"]
    summary = folds_df.groupby(["view", "model", "variant"], as_index=False)[cols].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(p for p in c if p).rstrip("_") for c in summary.columns]
    return summary.sort_values("rmsle_mean").reset_index(drop=True)


def paired_tables(folds_df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    df = folds_df.copy()
    df["config"] = df["model"] + "|" + df["variant"]
    tuned = summary[summary["variant"] == "tuned"].sort_values("rmsle_mean")
    ranking = [f"{m}|tuned" for m in tuned["model"]]
    pairs = [(ranking[0], other) for other in ranking[1:]]
    pairs += [(ranking[0], "dummy_median_log|baseline"), (ranking[0], "dummy_mean_log|baseline")]
    pairs += [(f"{m}|tuned", f"{m}|default") for m in tuned["model"]]
    table = paired_test_table(df, pairs, label="config")
    families = []
    for _, row in table.iterrows():
        if row["b"].endswith("|default") and row["a"].split("|")[0] == row["b"].split("|")[0]:
            families.append("tuned_vs_default")
        elif row["b"].endswith("|baseline"):
            families.append("best_vs_baseline")
        else:
            families.append("best_vs_tuned_models")
    table.insert(0, "family", families)
    for _, idx in table.groupby(["family", "metric"]).groups.items():
        table.loc[idx, "p_nb_holm"] = _holm(table.loc[idx, "p_nadeau_bengio"].tolist())
    return table


def tune_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=tuple(VIEWS), default="repository_only")
    parser.add_argument("--models", nargs="+", default=list(REGRESSION_MODELS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def tune_main() -> None:
    args = tune_parse_args()
    unknown = sorted(set(args.models) - set(REGRESSION_MODELS))
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {unknown}")
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / args.view
    frame = subset_frame(load_count_frame(need_text=False, split_name="train"), args.max_rows)
    models = tuple(args.models)
    started = time.monotonic()
    folds_df = run_nested(
        frame, view=args.view, models=models, repeats=args.repeats, folds=args.folds,
        inner_folds=args.inner_folds, n_iter=args.n_iter, seed=args.seed,
    )
    summary = summarize(folds_df)
    paired = paired_tables(folds_df, summary)
    best = (
        {}
        if args.skip_final
        else final_search(
            frame, view=args.view, models=models, inner_folds=args.inner_folds,
            n_iter=args.n_iter, seed=args.seed,
        )
    )
    tuned = summary[summary["variant"] == "tuned"]
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "nested CV: repeated StratifiedGroupKFold (outer) + RandomizedSearchCV "
        "with grouped inner folds; external test not read",
        "view": args.view,
        "models": list(models),
        "repeats": args.repeats,
        "outer_folds": args.folds,
        "inner_folds": args.inner_folds,
        "n_iter": args.n_iter,
        "outer_seeds": [args.seed + r for r in range(args.repeats)],
        "search_random_state": 2027,
        "scoring": "neg RMSE on log1p(count) (equals RMSLE)",
        "search_spaces": {m: {k: str(v) for k, v in s.items()} for m, s in SEARCH_SPACES.items()},
        "rows": int(len(frame)),
        "groups": int(frame["representative_repo_id"].nunique()),
        "best_model_by_nested_rmsle": str(tuned.iloc[0]["model"]) if len(tuned) else None,
        "diagnostic_subset": args.max_rows is not None,
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(output_dir / "nested_folds.csv", index=False)
    summary.to_csv(output_dir / "nested_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_tests.csv", index=False)
    (output_dir / "best_params_full_dev.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
