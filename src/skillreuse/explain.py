"""Explicabilidade, auditoria da cauda e auditoria de artefatos textuais."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from skillreuse.benchmark import (
    _assemble_matrices,
    _metrics,
    _prepare_structured_components,
    _prepare_text_components,
)
from skillreuse.models import (
    DEFAULT_OUTPUT,
    REPOSITORY_CATEGORICAL_FEATURES,
    REPOSITORY_NUMERIC_FEATURES,
    STRUCTURAL_CATEGORICAL_FEATURES,
    STRUCTURAL_NUMERIC_FEATURES,
    VIEW_CONFIGS,
    build_preprocessor,
    load_count_frame,
    normalize_text,
    predictor_view,
)

# ==========================================================================
# Explicabilidade do Ridge de contagem congelado provisoriamente.
# ==========================================================================

EXPLAIN_OUTPUT_DIR = DEFAULT_OUTPUT / "explainability"

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

def explain_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPLAIN_OUTPUT_DIR)
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

def explain_main() -> None:
    args = explain_parse_args()
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


# ==========================================================================
# Auditoria de erro por quantis do alvo de contagem.
# ==========================================================================

TAIL_OUTPUT_DIR = DEFAULT_OUTPUT / "tail_audit"

def _tail_bin(values: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    return pd.cut(
        values,
        bins=[
            -np.inf,
            thresholds["p50"],
            thresholds["p75"],
            thresholds["p90"],
            thresholds["p95"],
            np.inf,
        ],
        labels=["<=p50", "p50-p75", "p75-p90", "p90-p95", ">p95"],
        include_lowest=True,
    ).astype(str)

def run_tail_audit(
    frame: pd.DataFrame,
    *,
    view_name: str = "hybrid",
    candidate_features: int = 10_000,
    top_k_text: int = 500,
    alpha: float = 10.0,
    folds: int = 3,
    random_seed: int = 2027,
    text_variant: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if view_name not in VIEW_CONFIGS:
        raise ValueError(f"Visao desconhecida: {view_name}")
    config = VIEW_CONFIGS[view_name]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].astype(str).to_numpy()
    prediction_records: list[dict[str, object]] = []
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
        components: dict[str, object] = {
            "text_train": None,
            "text_valid": None,
            "structured_train": None,
            "structured_valid": None,
            "prep_seconds": 0.0,
        }
        if config["use_text"]:
            text_components = _prepare_text_components(
                fold_train,
                fold_valid,
                candidate=candidate_features,
                text_variant=text_variant,
            )
            components.update(text_components)
        if config["use_structural"] or config["use_repository"]:
            components.update(
                _prepare_structured_components(fold_train, fold_valid, config=config)
            )
        x_train, x_valid, feature_count = _assemble_matrices(
            components,
            config=config,
            top_k=top_k_text,
            y_train_log=fold_train["log_count_target"].to_numpy(dtype=float),
        )
        model = Ridge(alpha=alpha).fit(
            x_train, fold_train["log_count_target"].to_numpy(dtype=float)
        )
        predictions = np.maximum(np.expm1(model.predict(x_valid)), 0.0)
        thresholds = {
            quantile: float(fold_train["count_target"].quantile(level))
            for quantile, level in [("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95)]
        }
        bins = _tail_bin(fold_valid["count_target"], thresholds)
        for index, (row, prediction, tail_group) in enumerate(
            zip(fold_valid.itertuples(index=False), predictions, bins, strict=False)
        ):
            prediction_records.append(
                {
                    "fold": fold,
                    "row_index_in_fold": index,
                    "content_hash": row.content_hash,
                    "representative_repo_id": row.representative_repo_id,
                    "true_count": float(row.count_target),
                    "predicted_count": float(prediction),
                    "error": float(prediction - row.count_target),
                    "underpredicted": bool(prediction < row.count_target),
                    "tail_bin": tail_group,
                    "feature_count": feature_count,
                    "group_overlap": len(overlap),
                    **{f"train_{key}": value for key, value in thresholds.items()},
                }
            )
    predictions_df = pd.DataFrame(prediction_records)
    summary_records: list[dict[str, object]] = []
    for tail_group, subset in predictions_df.groupby("tail_bin", observed=False):
        metrics = _metrics(
            subset["true_count"].to_numpy(dtype=float),
            subset["predicted_count"].to_numpy(dtype=float),
        )
        summary_records.append(
            {
                "tail_bin": tail_group,
                "rows": int(len(subset)),
                "median_true": float(subset["true_count"].median()),
                "median_predicted": float(subset["predicted_count"].median()),
                "mean_error": float(subset["error"].mean()),
                "underprediction_rate": float(subset["underpredicted"].mean()),
                **metrics,
            }
        )
    summary_df = pd.DataFrame(summary_records)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": (
            "StratifiedGroupKFold on external training reused subset; external test untouched"
        ),
        "target": "repository_count conditional on reused=1",
        "view": view_name,
        "text_variant": text_variant,
        "candidate_features": candidate_features,
        "top_k_text": top_k_text,
        "ridge_alpha": alpha,
        "folds": folds,
        "random_seed": random_seed,
        "rows": int(len(frame)),
        "groups": int(frame["representative_repo_id"].nunique()),
        "group_overlap_max": int(predictions_df["group_overlap"].max()),
        "threshold_source": "training fold only",
        "test_external_used": False,
    }
    return predictions_df, summary_df, manifest

def tail_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TAIL_OUTPUT_DIR)
    parser.add_argument("--view", choices=tuple(VIEW_CONFIGS), default="hybrid")
    parser.add_argument("--candidate-features", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--text-variant", choices=("raw", "drop_artifacts"), default="raw")
    return parser.parse_args()

def tail_main() -> None:
    args = tail_parse_args()
    frame = load_count_frame()
    predictions, summary, manifest = run_tail_audit(
        frame,
        view_name=args.view,
        candidate_features=args.candidate_features,
        top_k_text=args.top_k,
        alpha=args.alpha,
        folds=args.folds,
        random_seed=args.seed,
        text_variant=args.text_variant,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "tail_audit_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "tail_audit_summary.csv", index=False)
    (args.output_dir / "tail_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ==========================================================================
# Audita artefatos lexicais nos termos usados pela explicabilidade.
# ==========================================================================

DEFAULT_INPUT = Path(
    "results/count_regression/explainability_hybrid/global_feature_importance.csv"
)

DEFAULT_OUTPUT = Path("results/count_regression/explainability_hybrid")

URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)

DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")

NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")

IDENTIFIER_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b", re.IGNORECASE)

def audit_terms(input_path: Path = DEFAULT_INPUT) -> tuple[pd.DataFrame, dict[str, object]]:
    table = pd.read_csv(input_path)
    text_mask = table["block"].eq("text")
    text = table.loc[text_mask, "feature"].astype(str)
    result = table.loc[text_mask, ["feature", "coefficient_log1p", "absolute_coefficient"]].copy()
    result["url_artifact"] = text.map(lambda value: bool(URL_PATTERN.search(value)))
    result["date_artifact"] = text.map(lambda value: bool(DATE_PATTERN.search(value)))
    result["numeric_artifact"] = text.map(lambda value: bool(NUMBER_PATTERN.search(value)))
    result["identifier_artifact"] = text.map(lambda value: bool(IDENTIFIER_PATTERN.search(value)))
    result["artifact_any"] = result[
        ["url_artifact", "date_artifact", "numeric_artifact", "identifier_artifact"]
    ].any(axis=1)
    counts = result[
        ["url_artifact", "date_artifact", "numeric_artifact", "identifier_artifact", "artifact_any"]
    ].sum()
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "text_terms": int(len(result)),
        "artifact_counts": {key: int(value) for key, value in counts.items()},
        "artifact_fraction": {
            key: float(value / len(result)) if len(result) else 0.0
            for key, value in counts.items()
        },
        "interpretation": "artifact flags are audit indicators, not proof of leakage",
    }
    return result, manifest

def audit_text_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()

def audit_text_main() -> None:
    args = audit_text_parse_args()
    result, manifest = audit_terms(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "text_artifact_audit.csv", index=False)
    (args.output_dir / "text_artifact_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(result.loc[result["artifact_any"]].head(20).to_string(index=False))
