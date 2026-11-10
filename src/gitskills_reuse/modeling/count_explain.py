"""Explicabilidade do Ridge de contagem congelado provisoriamente.

O ajuste usa somente skills reutilizadas da partição externa de treino. As
contribuições são calculadas na escala de ``log1p(repository_count)``; a
transformação para a escala de contagem é usada apenas para exibir previsões.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.count_targets import DEFAULT_OUTPUT, VIEW_CONFIGS, load_count_frame
from gitskills_reuse.modeling.preprocess import (
    REPOSITORY_CATEGORICAL_FEATURES,
    REPOSITORY_NUMERIC_FEATURES,
    STRUCTURAL_CATEGORICAL_FEATURES,
    STRUCTURAL_NUMERIC_FEATURES,
    build_preprocessor,
    predictor_view,
)
from gitskills_reuse.modeling.text_variants import normalize_text

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT / "explainability"


def _feature_block(name: str, text_names: set[str]) -> str:
    if name in text_names:
        return "text"
    if name in STRUCTURAL_NUMERIC_FEATURES or name in STRUCTURAL_CATEGORICAL_FEATURES:
        return "structural"
    if name in REPOSITORY_NUMERIC_FEATURES or name in REPOSITORY_CATEGORICAL_FEATURES:
        return "repository"
    return "encoded"


def explain_ridge(
    frame: pd.DataFrame,
    *,
    view_name: str = "hybrid",
    candidate_features: int = 10_000,
    top_k_text: int = 500,
    alpha: float = 10.0,
    local_rows: int = 20,
    text_variant: str = "raw",
) -> dict[str, object]:
    if view_name not in VIEW_CONFIGS:
        raise ValueError(f"Visao desconhecida: {view_name}")
    config = VIEW_CONFIGS[view_name]
    model_frame = frame.copy()
    if config["use_text"]:
        model_frame["content"] = model_frame["content"].map(
            lambda value: normalize_text(value, text_variant)
        )
    else:
        normalize_text("", text_variant)
    predictors = predictor_view(model_frame, **config)
    y_log = model_frame["log_count_target"].to_numpy(dtype=float)
    preprocessor = build_preprocessor(
        predictors,
        **config,
        max_features=candidate_features,
        text_select_k=top_k_text if config["use_text"] else None,
    )
    x_train = preprocessor.fit_transform(predictors, y_log)
    model = Ridge(alpha=alpha).fit(x_train, y_log)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    coefficients = np.asarray(model.coef_, dtype=float)
    if len(feature_names) != len(coefficients):
        raise RuntimeError("Nomes e coeficientes possuem dimensoes diferentes")

    text_names: set[str] = set()
    if config["use_text"]:
        text_pipe = preprocessor.named_transformers_["text_tfidf"]
        text_names = set(text_pipe.get_feature_names_out())
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient_log1p": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "block": [_feature_block(name, text_names) for name in feature_names],
        }
    ).sort_values("absolute_coefficient", ascending=False, ignore_index=True)

    x_mean = np.asarray(x_train.mean(axis=0)).ravel()
    predictions_log = model.predict(x_train)
    predictions = np.maximum(np.expm1(predictions_log), 0.0)
    order = np.argsort(-np.abs(predictions_log - y_log))[: min(local_rows, len(frame))]
    local_records: list[dict[str, object]] = []
    for row_index in order:
        row_matrix = x_train[row_index]
        row = (
            row_matrix.toarray().ravel()
            if hasattr(row_matrix, "toarray")
            else np.asarray(row_matrix).ravel()
        )
        contributions = (row - x_mean) * coefficients
        top_indices = np.argsort(-np.abs(contributions))[:20]
        for rank, feature_index in enumerate(top_indices, start=1):
            local_records.append(
                {
                    "row_index": int(row_index),
                    "content_hash": model_frame.iloc[row_index]["content_hash"],
                    "representative_repo_id": model_frame.iloc[row_index]["representative_repo_id"],
                    "repository_count": int(model_frame.iloc[row_index]["count_target"]),
                    "predicted_count": float(predictions[row_index]),
                    "log_residual": float(predictions_log[row_index] - y_log[row_index]),
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "block": _feature_block(feature_names[feature_index], text_names),
                    "contribution_log1p": float(contributions[feature_index]),
                }
            )

    return {
        "importance": importance,
        "local": pd.DataFrame(local_records),
        "manifest": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "target": "repository_count conditional on reused=1",
            "view": view_name,
            "candidate_features": candidate_features,
            "top_k_text": top_k_text,
            "ridge_alpha": alpha,
            "rows": int(len(frame)),
            "groups": int(frame["representative_repo_id"].nunique()),
            "feature_count": int(len(feature_names)),
            "explanation_scale": "log1p(repository_count)",
            "text_variant": text_variant,
            "test_external_used": False,
            "caveat": "global coefficients are provisional until stability analysis across folds",
        },
    }


def explain_ridge_stability(
    frame: pd.DataFrame,
    *,
    view_name: str = "hybrid",
    candidate_features: int = 10_000,
    top_k_text: int = 500,
    alpha: float = 10.0,
    folds: int = 3,
    random_seed: int = 2027,
    text_variant: str = "raw",
) -> dict[str, object]:
    """Calcula coeficientes fora da dobra e sua estabilidade entre dobras."""
    if view_name not in VIEW_CONFIGS:
        raise ValueError(f"Visao desconhecida: {view_name}")
    config = VIEW_CONFIGS[view_name]
    model_frame = frame.copy()
    if config["use_text"]:
        model_frame["content"] = model_frame["content"].map(
            lambda value: normalize_text(value, text_variant)
        )
    else:
        normalize_text("", text_variant)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    strata = model_frame["split_stratum"].astype(str).to_numpy()
    groups = model_frame["representative_repo_id"].astype(str).to_numpy()
    records: list[dict[str, object]] = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y=strata, groups=groups), start=1
    ):
        fold_train = model_frame.iloc[train_idx].reset_index(drop=True)
        fold_valid = model_frame.iloc[valid_idx].reset_index(drop=True)
        overlap = set(fold_train["representative_repo_id"]) & set(
            fold_valid["representative_repo_id"]
        )
        if overlap:
            raise RuntimeError(f"Dobra {fold} compartilha grupos: {len(overlap)}")
        train_predictors = predictor_view(fold_train, **config)
        preprocessor = build_preprocessor(
            train_predictors,
            **config,
            max_features=candidate_features,
            text_select_k=top_k_text if config["use_text"] else None,
        )
        y_log = fold_train["log_count_target"].to_numpy(dtype=float)
        x_train = preprocessor.fit_transform(train_predictors, y_log)
        model = Ridge(alpha=alpha).fit(x_train, y_log)
        names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
        coefs = np.asarray(model.coef_, dtype=float)
        text_names: set[str] = set()
        if config["use_text"]:
            text_names = set(preprocessor.named_transformers_["text_tfidf"].get_feature_names_out())
        order = np.argsort(-np.abs(coefs))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        for name, coef, rank in zip(names, coefs, ranks, strict=False):
            records.append(
                {
                    "fold": fold,
                    "feature": name,
                    "coefficient_log1p": float(coef),
                    "absolute_coefficient": float(abs(coef)),
                    "rank_absolute": int(rank),
                    "block": _feature_block(name, text_names),
                    "train_rows": len(train_idx),
                    "validation_rows": len(valid_idx),
                    "group_overlap": len(overlap),
                }
            )
    fold_coefficients = pd.DataFrame(records)
    stability = (
        fold_coefficients.groupby(["feature", "block"], as_index=False)
        .agg(
            folds_present=("fold", "nunique"),
            mean_coefficient_log1p=("coefficient_log1p", "mean"),
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
            median_rank_absolute=("rank_absolute", "median"),
            sign_consistency=("coefficient_log1p", lambda values: abs(np.sign(values).mean())),
        )
        .assign(selected_fraction=lambda table: table["folds_present"] / folds)
        .sort_values(
            ["folds_present", "mean_absolute_coefficient"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "target": "repository_count conditional on reused=1",
        "view": view_name,
        "candidate_features": candidate_features,
        "top_k_text": top_k_text,
        "ridge_alpha": alpha,
        "folds": folds,
        "random_seed": random_seed,
        "rows": int(len(frame)),
        "groups": int(frame["representative_repo_id"].nunique()),
        "test_external_used": False,
        "stability_rule": "feature must recur across folds and preserve coefficient sign",
        "text_variant": text_variant,
    }
    return {"fold_coefficients": fold_coefficients, "stability": stability, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view", choices=tuple(VIEW_CONFIGS), default="hybrid")
    parser.add_argument("--candidate-features", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--local-rows", type=int, default=20)
    parser.add_argument("--text-variant", choices=("raw", "drop_artifacts"), default="raw")
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--stability-only", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_rows < 1:
        raise ValueError("local_rows deve ser positivo")
    frame = load_count_frame()
    artifacts = None
    if not args.stability_only:
        artifacts = explain_ridge(
            frame,
            view_name=args.view,
            candidate_features=args.candidate_features,
            top_k_text=args.top_k,
            alpha=args.alpha,
            local_rows=args.local_rows,
            text_variant=args.text_variant,
        )
    if args.stability:
        stability = explain_ridge_stability(
            frame,
            view_name=args.view,
            candidate_features=args.candidate_features,
            top_k_text=args.top_k,
            alpha=args.alpha,
            folds=args.folds,
            text_variant=args.text_variant,
        )
    else:
        stability = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if artifacts is not None:
        artifacts["importance"].to_csv(
            args.output_dir / "global_feature_importance.csv", index=False
        )
        artifacts["local"].to_csv(args.output_dir / "local_explanations.csv", index=False)
        (args.output_dir / "explainability_manifest.json").write_text(
            json.dumps(artifacts["manifest"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if stability is not None:
        stability["fold_coefficients"].to_csv(
            args.output_dir / "fold_feature_coefficients.csv", index=False
        )
        stability["stability"].to_csv(args.output_dir / "feature_stability.csv", index=False)
        (args.output_dir / "stability_manifest.json").write_text(
            json.dumps(stability["manifest"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if artifacts is not None:
        print(artifacts["importance"].head(20).to_string(index=False))
        print(json.dumps(artifacts["manifest"], indent=2, ensure_ascii=False))
    if stability is not None:
        print(stability["stability"].head(20).to_string(index=False))
        print(json.dumps(stability["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
