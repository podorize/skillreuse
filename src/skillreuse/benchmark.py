"""Validacao agrupada, benchmark de regressores e avaliacao externa."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import PoissonRegressor, Ridge, SGDRegressor, TweedieRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from skillreuse.models import (
    DEFAULT_OUTPUT,
    VIEW_CONFIGS,
    build_preprocessor,
    load_count_frame,
    normalize_text,
    predictor_view,
)

# ==========================================================================
# Piloto agrupado de regressao de contagem para a Fase 3.
# ==========================================================================

DEFAULT_CANDIDATE_FEATURES = (5_000, 10_000, 20_000)

DEFAULT_TOP_K = (50, 100, 200, 500)

def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clipped = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(clipped) - np.log1p(y_true)) ** 2)))

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    clipped = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    if np.unique(y_true).size < 2 or np.unique(clipped).size < 2:
        rank = 0.0
    else:
        rank = spearmanr(y_true, clipped).statistic
    return {
        "rmsle": _rmsle(y_true, clipped),
        "mae": float(np.mean(np.abs(y_true - clipped))),
        "mae_log1p": float(np.mean(np.abs(np.log1p(y_true) - np.log1p(clipped)))),
        "spearman": float(rank) if np.isfinite(rank) else 0.0,
    }

def _fit_predict(
    model_name: str,
    x_train: object,
    y_train: np.ndarray,
    x_valid: object,
) -> np.ndarray:
    if model_name == "dummy_log":
        model = DummyRegressor(strategy="mean")
        model.fit(x_train, np.log1p(y_train))
        return np.expm1(model.predict(x_valid))
    if model_name == "dummy_median_log":
        model = DummyRegressor(strategy="median")
        model.fit(x_train, np.log1p(y_train))
        return np.expm1(model.predict(x_valid))
    if model_name == "ridge_log":
        model = Ridge(alpha=10.0)
        model.fit(x_train, np.log1p(y_train))
        return np.expm1(model.predict(x_valid))
    if model_name == "poisson":
        model = PoissonRegressor(alpha=1.0, max_iter=300)
        model.fit(x_train, y_train)
        return model.predict(x_valid)
    if model_name == "tweedie_1p5":
        model = TweedieRegressor(power=1.5, alpha=1.0, max_iter=300)
        model.fit(x_train, y_train)
        return model.predict(x_valid)
    if model_name == "sgd_log":
        model = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            max_iter=2_000,
            tol=1e-3,
            random_state=2027,
        )
        model.fit(x_train, np.log1p(y_train))
        return np.expm1(model.predict(x_valid))
    raise ValueError(f"Modelo desconhecido: {model_name}")

def _prepare_text_components(
    fold_train: pd.DataFrame,
    fold_valid: pd.DataFrame,
    *,
    candidate: int,
    text_variant: str = "raw",
) -> dict[str, object]:
    """Ajusta o vocabulário textual uma única vez por dobra e limite."""
    start = time.monotonic()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=candidate,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_content = fold_train["content"].map(lambda value: normalize_text(value, text_variant))
    valid_content = fold_valid["content"].map(lambda value: normalize_text(value, text_variant))
    text_train = vectorizer.fit_transform(train_content)
    text_valid = vectorizer.transform(valid_content)
    return {
        "text_train": text_train,
        "text_valid": text_valid,
        "prep_seconds": time.monotonic() - start,
    }

def _prepare_structured_components(
    fold_train: pd.DataFrame,
    fold_valid: pd.DataFrame,
    *,
    config: dict[str, bool],
) -> dict[str, object]:
    """Ajusta atributos estruturais/contextuais uma única vez por dobra e visão."""
    start = time.monotonic()
    non_text_config = {**config, "use_text": False}
    train_predictors = predictor_view(fold_train, **non_text_config)
    valid_predictors = predictor_view(fold_valid, **non_text_config)
    transformer = build_preprocessor(train_predictors, **non_text_config)
    return {
        "structured_train": transformer.fit_transform(train_predictors),
        "structured_valid": transformer.transform(valid_predictors),
        "prep_seconds": time.monotonic() - start,
    }

def _assemble_matrices(
    components: dict[str, object],
    *,
    config: dict[str, bool],
    top_k: int,
    y_train_log: np.ndarray,
) -> tuple[object, object, int]:
    text_train = components["text_train"]
    text_valid = components["text_valid"]
    if config["use_text"]:
        selector = SelectKBest(f_regression, k=top_k)
        selected_train = selector.fit_transform(text_train, y_train_log)
        selected_valid = selector.transform(text_valid)
        train_blocks = [selected_train]
        valid_blocks = [selected_valid]
    else:
        train_blocks = []
        valid_blocks = []
    if components["structured_train"] is not None:
        train_blocks.append(components["structured_train"])
        valid_blocks.append(components["structured_valid"])
    if not train_blocks:
        raise ValueError("A visao nao possui atributos")
    if len(train_blocks) == 1:
        x_train, x_valid = train_blocks[0], valid_blocks[0]
    else:
        x_train = hstack(train_blocks).tocsr()
        x_valid = hstack(valid_blocks).tocsr()
    return x_train, x_valid, int(x_train.shape[1])

def run_pilot(
    frame: pd.DataFrame,
    *,
    candidate_features: tuple[int, ...] = (5_000,),
    top_k: tuple[int, ...] = (100,),
    views: tuple[str, ...] = tuple(VIEW_CONFIGS),
    models: tuple[str, ...] = ("dummy_log", "ridge_log", "poisson"),
    folds: int = 3,
    random_seed: int = 2027,
    text_variant: str = "raw",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if folds < 2:
        raise ValueError("folds deve ser >= 2")
    unknown_views = sorted(set(views).difference(VIEW_CONFIGS))
    if unknown_views:
        raise ValueError(f"Visoes desconhecidas: {unknown_views}")
    if any(value < 1 for value in candidate_features + top_k):
        raise ValueError("Limites textuais devem ser positivos")
    if any(k > candidate for candidate in candidate_features for k in top_k):
        raise ValueError("top_k nao pode exceder o pool candidato")
    normalize_text("", text_variant)

    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    records: list[dict[str, object]] = []

    fold_splits = list(splitter.split(frame, y=strata, groups=groups))
    text_cache: dict[tuple[int, int], dict[str, object]] = {}
    structured_cache: dict[tuple[int, str], dict[str, object]] = {}
    for candidate in candidate_features:
        for fold, (train_idx, valid_idx) in enumerate(fold_splits, start=1):
            fold_train = frame.iloc[train_idx].reset_index(drop=True)
            fold_valid = frame.iloc[valid_idx].reset_index(drop=True)
            overlap = set(fold_train["representative_repo_id"]) & set(
                fold_valid["representative_repo_id"]
            )
            if overlap:
                raise RuntimeError(f"Dobra {fold} compartilha grupos: {len(overlap)}")
            fold_y = fold_train["count_target"].to_numpy(dtype=float)
            valid_y = fold_valid["count_target"].to_numpy(dtype=float)
            y_train_log = fold_train["log_count_target"].to_numpy(dtype=float)
            for view_name in views:
                config = VIEW_CONFIGS[view_name]
                components: dict[str, object] = {
                    "text_train": None,
                    "text_valid": None,
                    "structured_train": None,
                    "structured_valid": None,
                    "prep_seconds": 0.0,
                }
                if config["use_text"]:
                    text_key = (candidate, fold)
                    if text_key not in text_cache:
                        text_cache[text_key] = _prepare_text_components(
                            fold_train,
                            fold_valid,
                            candidate=candidate,
                            text_variant=text_variant,
                        )
                    components.update(
                        {
                            "text_train": text_cache[text_key]["text_train"],
                            "text_valid": text_cache[text_key]["text_valid"],
                            "prep_seconds": text_cache[text_key]["prep_seconds"],
                        }
                    )
                if config["use_structural"] or config["use_repository"]:
                    structured_key = (fold, view_name)
                    if structured_key not in structured_cache:
                        structured_cache[structured_key] = _prepare_structured_components(
                            fold_train, fold_valid, config=config
                        )
                    components.update(
                        {
                            "structured_train": structured_cache[structured_key][
                                "structured_train"
                            ],
                            "structured_valid": structured_cache[structured_key][
                                "structured_valid"
                            ],
                            "prep_seconds": float(components["prep_seconds"])
                            + float(structured_cache[structured_key]["prep_seconds"]),
                        }
                    )
                for k in top_k:
                    select_start = time.monotonic()
                    x_train, x_valid, feature_count = _assemble_matrices(
                        components,
                        config=config,
                        top_k=k,
                        y_train_log=y_train_log,
                    )
                    prep_seconds = float(components["prep_seconds"]) + (
                        time.monotonic() - select_start
                    )
                    for model_name in models:
                        model_start = time.monotonic()
                        pred = _fit_predict(model_name, x_train, fold_y, x_valid)
                        values = _metrics(valid_y, pred)
                        records.append(
                            {
                                "candidate_features": candidate,
                                "top_k_text": k,
                                "view": view_name,
                                "model": model_name,
                                "fold": fold,
                                "train_rows": len(train_idx),
                                "validation_rows": len(valid_idx),
                                "train_groups": fold_train["representative_repo_id"].nunique(),
                                "validation_groups": fold_valid["representative_repo_id"].nunique(),
                                "group_overlap": len(overlap),
                                "feature_count": feature_count,
                                "prep_seconds": prep_seconds,
                                "fit_predict_seconds": time.monotonic() - model_start,
                                **values,
                            }
                        )

    folds_df = pd.DataFrame(records)
    metric_columns = ["rmsle", "mae", "mae_log1p", "spearman"]
    summary = (
        folds_df.groupby(["candidate_features", "top_k_text", "view", "model"], as_index=False)[
            metric_columns
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": (
            "StratifiedGroupKFold on external training reused subset; external test untouched"
        ),
        "target": "repository_count conditional on reused=1",
        "group_column": "representative_repo_id",
        "candidate_features": list(candidate_features),
        "top_k_text": list(top_k),
        "views": list(views),
        "models": list(models),
        "folds": folds,
        "random_seed": random_seed,
        "rows": int(len(frame)),
        "groups": int(frame["representative_repo_id"].nunique()),
        "primary_metric": "rmsle",
        "selection_note": (
            "f_regression over log1p(count) for textual term selection inside each fold"
        ),
        "text_variant": text_variant,
    }
    return folds_df, {"summary": summary, "manifest": manifest}

def subset_frame(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Subconjunto deterministico e estratificado, apenas para diagnosticos rapidos."""
    if max_rows is None:
        return frame
    if max_rows < 1:
        raise ValueError("max_rows deve ser positivo")
    return (
        frame.sort_values(["split_stratum", "content_hash"])
        .groupby("split_stratum", group_keys=False)
        .head(max(1, max_rows // frame["split_stratum"].nunique()))
        .head(max_rows)
        .reset_index(drop=True)
    )


def _holm(p_values: list[float]) -> list[float]:
    """Correcao de Holm-Bonferroni para uma familia de comparacoes."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def paired_test_table(
    folds: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    label: str,
    metrics: tuple[str, ...] = ("rmsle", "mae"),
) -> pd.DataFrame:
    """Comparacao pareada por (repeticao, dobra) entre configuracoes ``a`` e ``b``.

    ``mean_diff = media(a - b)``: negativo significa que ``a`` tem erro menor. Como as
    repeticoes reutilizam as mesmas observacoes, os escores nao sao independentes; por isso
    o teste t usa a correcao de Nadeau-Bengio (variancia * (1/J + n_valid/n_treino)). O
    Wilcoxon pareado e informado como complemento e tende a ser otimista nesse arranjo.
    """
    ratio = float((folds["validation_rows"] / folds["train_rows"]).mean())
    rows: list[dict[str, object]] = []
    for metric in metrics:
        wide = folds.pivot_table(index=["repeat", "fold"], columns=label, values=metric)
        block: list[dict[str, object]] = []
        for a, b in pairs:
            diff = (wide[a] - wide[b]).dropna().to_numpy()
            n = len(diff)
            mean = float(diff.mean())
            var = float(diff.var(ddof=1)) if n > 1 else float("nan")
            se = float(np.sqrt((1.0 / n + ratio) * var)) if n > 1 else float("nan")
            if n < 2:
                p_nb = 1.0
            elif se == 0.0:  # diferenca constante entre as dobras
                p_nb = 0.0 if mean != 0.0 else 1.0
            else:
                p_nb = float(2 * stats.t.sf(abs(mean / se), n - 1))
            crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
            try:
                p_w = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                p_w = 1.0
            block.append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "n_pairs": n,
                    "mean_diff_a_minus_b": mean,
                    "ci95_low": mean - crit * se,
                    "ci95_high": mean + crit * se,
                    "a_better_share": float(np.mean(diff < 0)),
                    "p_wilcoxon": p_w,
                    "p_nadeau_bengio": p_nb,
                }
            )
        adjusted = _holm([float(r["p_nadeau_bengio"]) for r in block])
        for row, adj in zip(block, adjusted, strict=True):
            row["p_nb_holm"] = adj
        rows.extend(block)
    return pd.DataFrame(rows)


def view_pairs(configs: list[str]) -> list[tuple[str, str]]:
    """Pares relevantes: mesmo modelo em visoes distintas e cada modelo contra as referencias."""
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(configs):
        view_a, model_a = a.split("|")
        for b in configs[i + 1 :]:
            view_b, model_b = b.split("|")
            if model_a == model_b and "dummy" not in model_a:
                pairs.append((a, b))
            elif view_a == view_b and ("dummy" in model_a) != ("dummy" in model_b):
                pairs.append((a, b) if "dummy" not in model_a else (b, a))
    return pairs


def views_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-features", type=int, nargs="+", default=[5_000])
    parser.add_argument("--top-k", type=int, nargs="+", default=[100])
    parser.add_argument("--views", nargs="+", default=list(VIEW_CONFIGS))
    parser.add_argument(
        "--models", nargs="+", default=["ridge_log"]
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1, help="repeticoes com sementes seed+r")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--text-variant", choices=("raw", "drop_artifacts"), default="raw")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limite opcional para um spot-check rapido; nao usar no resultado final",
    )
    return parser.parse_args()

def views_main() -> None:
    args = views_parse_args()
    frame = subset_frame(load_count_frame(), args.max_rows)
    parts: list[pd.DataFrame] = []
    manifest: dict[str, object] = {}
    for repeat in range(args.repeats):
        part, artifacts = run_pilot(
            frame,
            candidate_features=tuple(args.candidate_features),
            top_k=tuple(args.top_k),
            views=tuple(args.views),
            models=tuple(args.models),
            folds=args.folds,
            random_seed=args.seed + repeat,
            text_variant=args.text_variant,
        )
        part.insert(0, "repeat", repeat + 1)
        parts.append(part)
        manifest = artifacts["manifest"]
    folds_df = pd.concat(parts, ignore_index=True)
    group_cols = ["candidate_features", "top_k_text", "view", "model"]
    metric_columns = ["rmsle", "mae", "mae_log1p", "spearman"]
    summary = folds_df.groupby(group_cols, as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = ["_".join(p for p in c if p).rstrip("_") for c in summary.columns]
    manifest.update(
        {
            "repeats": args.repeats,
            "seeds": [args.seed + r for r in range(args.repeats)],
            "diagnostic_subset": args.max_rows is not None,
            "subset_note": (
                "deterministic capped subset for pipeline diagnostics"
                if args.max_rows is not None
                else "full reused training subset"
            ),
        }
    )
    # testes pareados entre visoes (mesmo modelo) e contra a referencia constante
    folds_df["config"] = folds_df["view"] + "|" + folds_df["model"]
    pairs = view_pairs(sorted(folds_df["config"].unique()))
    paired = paired_test_table(
        folds_df[folds_df["top_k_text"] == folds_df["top_k_text"].iloc[0]],
        pairs,
        label="config",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_df.drop(columns="config").to_csv(
        args.output_dir / "count_regression_pilot_folds.csv", index=False
    )
    summary.to_csv(args.output_dir / "count_regression_pilot_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_tests.csv", index=False)
    (args.output_dir / "count_regression_pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


# ==========================================================================
# Benchmark de regressores para o trabalho da disciplina.
# ==========================================================================

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT / "discipline_benchmark"

REGRESSION_MODELS = (
    "ridge_log",
    "knn_log",
    "decision_tree_log",
    "random_forest_log",
    "gradient_boosting_log",
    "hist_gradient_boosting_log",
)

def default_regressor(name: str) -> object:
    """Regressor com os hiperparametros fixos do benchmark original (sem busca)."""
    models = {
        "ridge_log": lambda: Ridge(alpha=10.0),
        "knn_log": lambda: KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1),
        "decision_tree_log": lambda: DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=2027
        ),
        "random_forest_log": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=2027,
        ),
        "gradient_boosting_log": lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, loss="huber", random_state=2027
        ),
        "hist_gradient_boosting_log": lambda: HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=1.0,
            random_state=2027,
        ),
    }
    return models[name]()


def _fit_regressor(name: str, x_train: object, y_log: np.ndarray) -> object:
    dense = name != "ridge_log"
    fit_x = x_train.toarray() if dense and hasattr(x_train, "toarray") else x_train
    model = default_regressor(name)
    model.fit(fit_x, y_log)
    return model

def _predict_regressor(name: str, model: object, x_valid: object) -> np.ndarray:
    dense = name != "ridge_log"
    predict_x = x_valid.toarray() if dense and hasattr(x_valid, "toarray") else x_valid
    return np.maximum(np.expm1(model.predict(predict_x)), 0.0)

def run_benchmark(
    frame: pd.DataFrame,
    *,
    views: tuple[str, ...] = ("hybrid",),
    candidate_features: int = 10_000,
    top_k_text: int = 500,
    folds: int = 3,
    random_seed: int = 2027,
    text_variant: str = "raw",
    regression_models: tuple[str, ...] = REGRESSION_MODELS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    unknown_views = sorted(set(views).difference(VIEW_CONFIGS))
    if unknown_views:
        raise ValueError(f"Visoes desconhecidas: {unknown_views}")
    unknown_models = sorted(set(regression_models).difference(REGRESSION_MODELS))
    if unknown_models:
        raise ValueError(f"Modelos desconhecidos: {unknown_models}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].astype(str).to_numpy()
    regression_records: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y=strata, groups=groups), start=1
    ):
        fold_train = frame.iloc[train_idx].reset_index(drop=True)
        fold_valid = frame.iloc[valid_idx].reset_index(drop=True)
        overlap = set(fold_train["representative_repo_id"]) & set(
            fold_valid["representative_repo_id"]
        )
        if overlap:
            raise RuntimeError(f"Dobra {fold} compartilha grupos: {len(overlap)}")
        y_log = fold_train["log_count_target"].to_numpy(dtype=float)
        y_valid = fold_valid["count_target"].to_numpy(dtype=float)
        for view_name in views:
            config = VIEW_CONFIGS[view_name]
            components: dict[str, object] = {
                "text_train": None,
                "text_valid": None,
                "structured_train": None,
                "structured_valid": None,
                "prep_seconds": 0.0,
            }
            if config["use_text"]:
                components.update(
                    _prepare_text_components(
                        fold_train,
                        fold_valid,
                        candidate=candidate_features,
                        text_variant=text_variant,
                    )
                )
            if config["use_structural"] or config["use_repository"]:
                components.update(
                    _prepare_structured_components(fold_train, fold_valid, config=config)
                )
            x_train, x_valid, feature_count = _assemble_matrices(
                components,
                config=config,
                top_k=top_k_text,
                y_train_log=y_log,
            )
            for model_name in regression_models:
                start = time.monotonic()
                model = _fit_regressor(model_name, x_train, y_log)
                predictions = _predict_regressor(model_name, model, x_valid)
                regression_records.append(
                    {
                        "fold": fold,
                        "view": view_name,
                        "model": model_name,
                        "feature_count": feature_count,
                        "train_rows": len(train_idx),
                        "validation_rows": len(valid_idx),
                        "group_overlap": len(overlap),
                        "fit_seconds": time.monotonic() - start,
                        **_metrics(y_valid, predictions),
                    }
                )
    regression = pd.DataFrame(regression_records)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "discipline benchmark; StratifiedGroupKFold; external test untouched",
        "target_regression": "repository_count conditional on reused=1",
        "views": list(views),
        "candidate_features": candidate_features,
        "top_k_text": top_k_text,
        "folds": folds,
        "random_seed": random_seed,
        "text_variant": text_variant,
        "regression_models": list(regression_models),
        "rows": int(len(frame)),
        "groups": int(frame["representative_repo_id"].nunique()),
    }
    return regression, manifest

def benchmark_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", nargs="+", default=["hybrid"])
    parser.add_argument("--models", nargs="+", default=list(REGRESSION_MODELS))
    parser.add_argument("--candidate-features", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--text-variant", choices=("raw", "drop_artifacts"), default="raw")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()

def benchmark_main() -> None:
    args = benchmark_parse_args()
    frame = subset_frame(load_count_frame(), args.max_rows)
    regression, manifest = run_benchmark(
        frame,
        views=tuple(args.views),
        candidate_features=args.candidate_features,
        top_k_text=args.top_k,
        folds=args.folds,
        random_seed=args.seed,
        text_variant=args.text_variant,
        regression_models=tuple(args.models),
    )
    manifest["diagnostic_subset"] = args.max_rows is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regression.to_csv(args.output_dir / "regression_folds.csv", index=False)
    if not regression.empty:
        regression.groupby(["view", "model"], as_index=False)[
            ["rmsle", "mae", "mae_log1p", "spearman"]
        ].agg(["mean", "std"]).to_csv(args.output_dir / "regression_summary.csv")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(regression.to_string(index=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ==========================================================================
# Avaliação final da trilha da disciplina na visão repository-only.
# ==========================================================================

DEFAULT_OUTPUT = Path("results/discipline_benchmark/final_repository_only")

def _build_matrices(train: pd.DataFrame, test: pd.DataFrame) -> tuple[object, object, list[str]]:
    train_predictors = predictor_view(
        train, use_text=False, use_structural=False, use_repository=True
    )
    test_predictors = predictor_view(
        test, use_text=False, use_structural=False, use_repository=True
    )
    preprocessor = build_preprocessor(
        train_predictors, use_text=False, use_structural=False, use_repository=True
    )
    x_train = preprocessor.fit_transform(train_predictors)
    x_test = preprocessor.transform(test_predictors)
    names = list(preprocessor.get_feature_names_out())
    return x_train, x_test, names

def run_evaluation(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    train = load_count_frame(need_text=False, split_name="train")
    test = load_count_frame(need_text=False, split_name="test")
    overlap = set(train["representative_repo_id"]) & set(test["representative_repo_id"])
    if overlap:
        raise RuntimeError(f"Teste compartilha grupos com treino: {len(overlap)}")
    x_train, x_test, feature_names = _build_matrices(train, test)
    fit_train = x_train.toarray() if hasattr(x_train, "toarray") else x_train
    fit_test = x_test.toarray() if hasattr(x_test, "toarray") else x_test
    y_train = train["log_count_target"].to_numpy(dtype=float)
    y_test = test["count_target"].to_numpy(dtype=float)
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=2027,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            loss="huber",
            random_state=2027,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for name, model in models.items():
        model.fit(fit_train, y_train)
        predictions = np.maximum(np.expm1(model.predict(fit_test)), 0.0)
        summary.append({"model": name, **_metrics(y_test, predictions)})
        predictions_frame = test[
            ["content_hash", "representative_repo_id", "count_target"]
        ].copy()
        predictions_frame["prediction"] = predictions
        predictions_frame["absolute_error"] = np.abs(
            predictions_frame["count_target"] - predictions
        )
        predictions_frame.to_csv(output_dir / f"{name}_test_predictions.csv", index=False)
        importances = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importances.to_csv(output_dir / f"{name}_feature_importance.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "test_summary.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "discipline final evaluation; repository-only; external test used once",
        "target": "repository_count conditional on reused=1",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_groups": int(train["representative_repo_id"].nunique()),
        "test_groups": int(test["representative_repo_id"].nunique()),
        "group_overlap": int(len(overlap)),
        "feature_count": len(feature_names),
        "models": list(models),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "manifest": manifest}

def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_evaluation(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
