"""Auditoria de erro por quantis do alvo de contagem."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.count_benchmark import (
    _assemble_matrices,
    _metrics,
    _prepare_structured_components,
    _prepare_text_components,
)
from gitskills_reuse.modeling.count_targets import DEFAULT_OUTPUT, VIEW_CONFIGS, load_count_frame

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT / "tail_audit"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view", choices=tuple(VIEW_CONFIGS), default="hybrid")
    parser.add_argument("--candidate-features", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--text-variant", choices=("raw", "drop_artifacts"), default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
