"""Benchmark de regressores para o trabalho da disciplina."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from gitskills_reuse.modeling.count_benchmark import (
    _assemble_matrices,
    _metrics,
    _prepare_structured_components,
    _prepare_text_components,
)
from gitskills_reuse.modeling.count_targets import DEFAULT_OUTPUT, VIEW_CONFIGS, load_count_frame

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT / "discipline_benchmark"
REGRESSION_MODELS = (
    "ridge_log",
    "knn_log",
    "decision_tree_log",
    "random_forest_log",
    "gradient_boosting_log",
    "hist_gradient_boosting_log",
)


def _fit_regressor(name: str, x_train: object, y_log: np.ndarray) -> object:
    dense = name != "ridge_log"
    fit_x = x_train.toarray() if dense and hasattr(x_train, "toarray") else x_train
    models = {
        "ridge_log": Ridge(alpha=10.0),
        "knn_log": KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1),
        "decision_tree_log": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=2027
        ),
        "random_forest_log": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=2027,
        ),
        "gradient_boosting_log": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, loss="huber", random_state=2027
        ),
        "hist_gradient_boosting_log": HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=1.0,
            random_state=2027,
        ),
    }
    model = models[name]
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


def parse_args() -> argparse.Namespace:
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


if __name__ == "__main__":
    main()
