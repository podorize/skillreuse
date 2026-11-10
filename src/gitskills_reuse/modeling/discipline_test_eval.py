"""Avaliação final da trilha da disciplina na visão repository-only."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from gitskills_reuse.modeling.count_benchmark import _metrics
from gitskills_reuse.modeling.count_targets import load_count_frame
from gitskills_reuse.modeling.preprocess import build_preprocessor, predictor_view

DEFAULT_OUTPUT = Path("reports/modeling/discipline_benchmark/final_repository_only")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_evaluation(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
