"""Construcao do alvo de regressao de contagem para a Fase 3.

O alvo e a quantidade observada de repositorios que reutilizam cada skill. A
populacao de modelagem e condicional a ``reused = 1``; contagens, identificadores
e estratos continuam excluidos do conjunto de preditores.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gitskills_reuse.modeling.preprocess import predictor_view

DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")
DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")
DEFAULT_TARGETS = Path("data/processed/final_targets.parquet")
DEFAULT_OUTPUT = Path("reports/modeling/count_regression")

VIEW_CONFIGS = {
    "skill_only": {"use_text": True, "use_structural": True, "use_repository": False},
    "repository_only": {"use_text": False, "use_structural": False, "use_repository": True},
    "hybrid": {"use_text": True, "use_structural": True, "use_repository": True},
}


def load_count_frame(
    sample_path: Path = DEFAULT_SAMPLE,
    features_path: Path = DEFAULT_FEATURES,
    targets_path: Path = DEFAULT_TARGETS,
    *,
    need_text: bool = True,
    split_name: str = "train",
) -> pd.DataFrame:
    """Carrega skills reutilizadas de uma partição da divisão externa."""
    if split_name not in {"train", "test"}:
        raise ValueError("split_name deve ser 'train' ou 'test'")
    features = pq.read_table(features_path).to_pandas()
    target_columns = ["content_hash", "repository_count", "reused", "split_stratum"]
    targets = pq.read_table(targets_path, columns=target_columns).to_pandas()
    feature_reused = features[["content_hash", "reused"]]
    features = features.drop(columns=["reused"])
    frame = features.merge(targets, on="content_hash", how="inner", validate="one_to_one")
    expected_rows = len(frame)
    if need_text:
        # The modeling sample contains 200k rows, but the count benchmark only
        # needs the reused training subset.  Reading the text column as a
        # single pandas table creates unnecessary memory pressure.  Stream the
        # Parquet row groups instead and retain only hashes that can enter the
        # benchmark.  This preserves the exact source text while making the
        # discipline benchmark reproducible on constrained machines.
        wanted_hashes = set(
            frame.loc[frame["split"].eq(split_name) & frame["reused"].eq(1), "content_hash"]
        )
        batches: list[pd.DataFrame] = []
        parquet_file = pq.ParquetFile(sample_path)
        for batch in parquet_file.iter_batches(
            columns=["content_hash", "content"], batch_size=8192
        ):
            text_batch = batch.to_pandas()
            selected = text_batch[text_batch["content_hash"].isin(wanted_hashes)]
            if not selected.empty:
                batches.append(selected)
        text = (
            pd.concat(batches, ignore_index=True)
            if batches
            else pd.DataFrame(columns=["content_hash", "content"])
        )
        frame = frame.merge(text, on="content_hash", how="inner", validate="one_to_one")
        expected_rows = len(wanted_hashes)
    if len(frame) != expected_rows:
        raise RuntimeError("Nem todas as linhas de atributos encontraram alvo e texto")
    consistency = frame[["content_hash", "reused"]].merge(
        feature_reused,
        on="content_hash",
        how="inner",
        suffixes=("_target", "_feature"),
    )
    if not consistency["reused_target"].eq(consistency["reused_feature"]).all():
        raise ValueError("reused diverge entre features e alvos")

    subset = frame.loc[frame["split"].eq(split_name) & frame["reused"].eq(1)].copy()
    subset = subset.reset_index(drop=True)
    if subset.empty:
        raise RuntimeError(f"O subconjunto reutilizado de {split_name} esta vazio")

    counts = pd.to_numeric(subset["repository_count"], errors="coerce")
    if counts.isna().any() or (counts < 2).any() or (counts != counts.round()).any():
        raise ValueError("repository_count deve ser inteiro e >= 2 entre reutilizadas")
    subset["count_target"] = counts.astype("int32")
    subset["log_count_target"] = np.log1p(counts).astype("float32")
    return subset


def validate_target_exclusion(frame: pd.DataFrame) -> list[str]:
    """Confirma que a visao de preditores nao carrega o alvo de contagem."""
    predictors = predictor_view(
        frame,
        use_text=True,
        use_structural=True,
        use_repository=True,
    )
    leaked = sorted(
        set(predictors.columns).intersection(
            {"repository_count", "count_target", "log_count_target", "reused", "split_stratum"}
        )
    )
    if leaked:
        raise AssertionError(f"Alvo encontrado entre preditores: {leaked}")
    return list(predictors.columns)


def validate_target_views(frame: pd.DataFrame) -> dict[str, int]:
    """Verifica as tres visoes previstas sem ajustar nenhum transformador."""
    counts: dict[str, int] = {}
    for name, config in VIEW_CONFIGS.items():
        predictors = predictor_view(frame, **config)
        leaked = {
            "repository_count",
            "count_target",
            "log_count_target",
            "reused",
            "split_stratum",
        }.intersection(predictors.columns)
        if leaked:
            raise AssertionError(f"Alvo encontrado na visao {name}: {sorted(leaked)}")
        counts[name] = len(predictors.columns)
    return counts


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["count_target"].astype(float)
    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "groups": int(frame["representative_repo_id"].nunique()),
                "mean": float(counts.mean()),
                "median": float(counts.median()),
                "std": float(counts.std(ddof=1)),
                "variance": float(counts.var(ddof=1)),
                "variance_mean_ratio": float(counts.var(ddof=1) / counts.mean()),
                "min": int(counts.min()),
                "p90": float(counts.quantile(0.90)),
                "p95": float(counts.quantile(0.95)),
                "p99": float(counts.quantile(0.99)),
                "max": int(counts.max()),
            }
        ]
    )


def write_target_artifacts(
    frame: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_columns = validate_target_exclusion(frame)
    view_predictor_counts = validate_target_views(frame)
    summary = target_summary(frame)
    summary.to_csv(output_dir / "count_target_summary.csv", index=False)
    frame[
        [
            "content_hash",
            "representative_repo_id",
            "split_stratum",
            "count_target",
            "log_count_target",
        ]
    ].to_csv(output_dir / "count_targets_train_reused.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "target": "repository_count conditional on reused=1",
        "transformations": ["count_target", "log_count_target=log1p(repository_count)"],
        "rows": int(len(frame)),
        "groups": int(frame["representative_repo_id"].nunique()),
        "predictor_columns_checked": len(predictor_columns),
        "view_predictor_counts": view_predictor_counts,
        "target_columns_excluded": [
            "repository_count",
            "count_target",
            "log_count_target",
            "reused",
            "split_stratum",
        ],
        "next_step": "build grouped fold preprocessors and count-regression spot-check",
    }
    (output_dir / "count_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_count_frame(
        sample_path=args.sample,
        features_path=args.features,
        targets_path=args.targets,
    )
    manifest = write_target_artifacts(frame, output_dir=args.output_dir)
    print("Alvo de regressao de contagem preparado com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
