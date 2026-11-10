"""Piloto agrupado de regressao de contagem para a Fase 3.

O piloto usa apenas a particao externa de treino. Cada dobra ajusta o TF-IDF,
a selecao univariada de termos e os demais transformadores exclusivamente nos
grupos de treino da dobra; o teste externo nao e carregado.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import PoissonRegressor, Ridge, SGDRegressor, TweedieRegressor
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.count_targets import DEFAULT_OUTPUT, VIEW_CONFIGS, load_count_frame
from gitskills_reuse.modeling.preprocess import build_preprocessor, predictor_view
from gitskills_reuse.modeling.text_variants import normalize_text

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-features", type=int, nargs="+", default=[5_000])
    parser.add_argument("--top-k", type=int, nargs="+", default=[100])
    parser.add_argument("--views", nargs="+", default=list(VIEW_CONFIGS))
    parser.add_argument(
        "--models", nargs="+", default=["ridge_log"]
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--text-variant", choices=("raw", "drop_artifacts"), default="raw")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limite opcional para um spot-check rapido; nao usar no resultado final",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_count_frame()
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("max_rows deve ser positivo")
        frame = (
            frame.sort_values(["split_stratum", "content_hash"])
            .groupby("split_stratum", group_keys=False)
            .head(max(1, args.max_rows // frame["split_stratum"].nunique()))
            .head(args.max_rows)
            .reset_index(drop=True)
        )
    folds_df, artifacts = run_pilot(
        frame,
        candidate_features=tuple(args.candidate_features),
        top_k=tuple(args.top_k),
        views=tuple(args.views),
        models=tuple(args.models),
        folds=args.folds,
        random_seed=args.seed,
        text_variant=args.text_variant,
    )
    artifacts["manifest"]["diagnostic_subset"] = args.max_rows is not None
    artifacts["manifest"]["subset_note"] = (
        "deterministic capped subset for pipeline diagnostics"
        if args.max_rows is not None
        else "full reused training subset"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(args.output_dir / "count_regression_pilot_folds.csv", index=False)
    artifacts["summary"].to_csv(args.output_dir / "count_regression_pilot_summary.csv", index=False)
    (args.output_dir / "count_regression_pilot_manifest.json").write_text(
        json.dumps(artifacts["manifest"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(artifacts["summary"].to_string(index=False))
    print(json.dumps(artifacts["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
