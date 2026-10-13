"""Validacao cruzada agrupada para reutilizacao binaria e intensidade condicional."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import load_npz, save_npz
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.preprocess import build_preprocessor, predictor_view

DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")
DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")
DEFAULT_TARGETS = Path("data/processed/final_targets.parquet")
DEFAULT_OUTPUT = Path("reports/modeling")

ABLATION_CONFIGS = {
    "all": {
        "use_text": True,
        "use_structural": True,
        "use_repository": True,
        "description": "Texto TF-IDF + Estrutura Markdown + Contexto do Repositório",
    },
    "text_only": {
        "use_text": True,
        "use_structural": False,
        "use_repository": False,
        "description": "Apenas texto do arquivo (TF-IDF unigramas e bigramas)",
    },
    "structure_only": {
        "use_text": False,
        "use_structural": True,
        "use_repository": False,
        "description": "Apenas estrutura e textometria do arquivo SKILL.md",
    },
    "repo_only": {
        "use_text": False,
        "use_structural": False,
        "use_repository": True,
        "description": "Apenas contexto e metadados do repositório representativo",
    },
}


def load_modeling_frame(
    sample_path: Path,
    features_path: Path,
    targets_path: Path,
    *,
    need_text: bool = True,
) -> pd.DataFrame:
    """Une atributos, alvos (e texto quando necessario) por hash e conserva treino externo."""
    features = pq.read_table(features_path).to_pandas()
    targets = pq.read_table(
        targets_path,
        columns=["content_hash", "reuse_intensity", "split_stratum"],
    ).to_pandas()
    frame = features.merge(targets, on="content_hash", how="inner", validate="one_to_one")
    if need_text:
        text = pq.read_table(sample_path, columns=["content_hash", "content"]).to_pandas()
        frame = frame.merge(text, on="content_hash", how="inner", validate="one_to_one")
    if len(frame) != len(features):
        raise RuntimeError("Nem todas as linhas de atributos encontraram texto e alvo")
    train = frame.loc[frame["split"].eq("train")].copy().reset_index(drop=True)
    if train.empty:
        raise RuntimeError("O conjunto de treino externo esta vazio")
    return train


def _binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, predictions)),
        "precision_at_0_5": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predictions, zero_division=0)),
    }


def _intensity_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    metrics = {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "ordinal_mae": float(mean_absolute_error(y_true, predictions)),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=np.arange(4), zero_division=0
    )
    for class_id in range(4):
        metrics[f"class_{class_id}_precision"] = float(precision[class_id])
        metrics[f"class_{class_id}_recall"] = float(recall[class_id])
        metrics[f"class_{class_id}_f1"] = float(f1[class_id])
    return metrics


def _source_signature(paths: list[Path]) -> str:
    """Fingerprint rapido das entradas para invalidar cache quando os arquivos mudam."""
    metadata = []
    for path in paths:
        stat = path.stat()
        metadata.append(
            {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_sparse(path: Path, matrix: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    save_npz(temporary, matrix, compressed=True)
    temporary.replace(path)


def _fold_matrices(
    *,
    transformer: object,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    cache_dir: Path,
    fold: int,
) -> tuple[object, object, bool]:
    """Carrega matrizes persistidas ou transforma e salva uma dobra, de modo atomico."""
    train_path = cache_dir / f"fold_{fold}_train.npz"
    valid_path = cache_dir / f"fold_{fold}_validation.npz"
    metadata_path = cache_dir / f"fold_{fold}.json"
    if train_path.is_file() and valid_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("train_rows") == len(train_frame) and metadata.get(
            "validation_rows"
        ) == len(valid_frame):
            try:
                train_matrix = load_npz(train_path)
                valid_matrix = load_npz(valid_path)
                expected_features = metadata.get("n_features")
                if (
                    train_matrix.shape == (len(train_frame), expected_features)
                    and valid_matrix.shape == (len(valid_frame), expected_features)
                ):
                    return train_matrix, valid_matrix, True
            except (OSError, ValueError):
                pass

    train_matrix = transformer.fit_transform(train_frame)
    valid_matrix = transformer.transform(valid_frame)
    _save_sparse(train_path, train_matrix)
    _save_sparse(valid_path, valid_matrix)
    metadata_path.write_text(
        json.dumps(
            {
                "train_rows": train_matrix.shape[0],
                "validation_rows": valid_matrix.shape[0],
                "n_features": train_matrix.shape[1],
                "dtype": str(train_matrix.dtype),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return train_matrix, valid_matrix, False


def run_cross_validation(
    frame: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    n_splits: int = 5,
    random_seed: int = 20261009,
    use_text: bool = True,
    use_structural: bool = True,
    use_repository: bool = True,
    ablation_name: str = "all",
    max_features: int = 50_000,
    min_df: int = 5,
    max_iter: int = 1_000,
    intensity_max_iter: int = 5_000,
    intensity_tolerance: float = 1e-3,
    cache_identity: str | None = None,
) -> dict[str, object]:
    """Compara pesos de classe nas dobras; nunca consulta o teste externo."""
    if n_splits < 2:
        raise ValueError("n_splits deve ser pelo menos 2")
    if intensity_max_iter < 1 or intensity_tolerance <= 0:
        raise ValueError("Parametros de convergencia da intensidade invalidos")
    output_dir.mkdir(parents=True, exist_ok=True)
    y_binary = frame["reused"].astype(int).to_numpy()
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    records: list[dict[str, object]] = []
    if cache_identity is None:
        identifiers = pd.util.hash_pandas_object(
            frame[["content_hash", "representative_repo_id", "split_stratum"]], index=False
        ).values.tobytes()
        cache_identity = hashlib.sha256(identifiers).hexdigest()
    cache_spec = {
        "schema": 2,
        "source_signature": cache_identity,
        "ablation_name": ablation_name,
        "use_text": use_text,
        "use_structural": use_structural,
        "use_repository": use_repository,
        "n_splits": n_splits,
        "random_seed": random_seed,
        "max_features": max_features if use_text else 0,
        "min_df": min_df if use_text else 0,
        "ngram_range": [1, 2] if use_text else None,
        "max_df": 0.95 if use_text else None,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = output_dir / "cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(cache_spec, indent=2) + "\n", encoding="utf-8"
    )

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y=strata, groups=groups), start=1
    ):
        print(f"[{ablation_name}] Dobra {fold}/{n_splits}: ajustando pre-processamento...", flush=True)
        fold_train = frame.iloc[train_idx]
        fold_valid = frame.iloc[valid_idx]
        binary_train = predictor_view(
            fold_train,
            use_text=use_text,
            use_structural=use_structural,
            use_repository=use_repository,
        )
        binary_valid = predictor_view(
            fold_valid,
            use_text=use_text,
            use_structural=use_structural,
            use_repository=use_repository,
        )
        transformer = build_preprocessor(
            binary_train,
            use_text=use_text,
            use_structural=use_structural,
            use_repository=use_repository,
            max_features=max_features,
            min_df=min_df,
        )
        train_matrix, valid_matrix, cache_hit = _fold_matrices(
            transformer=transformer,
            train_frame=binary_train,
            valid_frame=binary_valid,
            cache_dir=cache_dir,
            fold=fold,
        )
        split_group_overlap = len(
            set(fold_train["representative_repo_id"]) & set(fold_valid["representative_repo_id"])
        )
        if split_group_overlap:
            raise RuntimeError(f"Dobra {fold}: ha grupos compartilhados")
        print(
            f"[{ablation_name}] Dobra {fold}/{n_splits}: matriz {train_matrix.shape[0]} x {train_matrix.shape[1]} "
            f"({'cache' if cache_hit else 'vetorizada'}); treinando classificadores...",
            flush=True,
        )

        for class_weight in (None, "balanced"):
            binary_model = LogisticRegression(
                solver="liblinear", class_weight=class_weight, max_iter=max_iter,
                random_state=random_seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                binary_model.fit(train_matrix, y_binary[train_idx])
            binary_converged = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            binary_scores = binary_model.predict_proba(valid_matrix)[:, 1]
            records.append(
                {
                    "ablation": ablation_name,
                    "task": "binary_reuse",
                    "fold": fold,
                    "class_weight": class_weight or "none",
                    "train_rows": len(train_idx),
                    "validation_rows": len(valid_idx),
                    "train_positive_rate": float(y_binary[train_idx].mean()),
                    "validation_positive_rate": float(y_binary[valid_idx].mean()),
                    "group_overlap": split_group_overlap,
                    "iterations": int(np.max(binary_model.n_iter_)),
                    "converged": binary_converged,
                    **_binary_metrics(y_binary[valid_idx], binary_scores),
                }
            )

            positive_train_mask = y_binary[train_idx] == 1
            positive_valid_mask = y_binary[valid_idx] == 1
            intensity_train_idx = train_idx[positive_train_mask]
            intensity_valid_idx = valid_idx[positive_valid_mask]
            y_intensity_train = frame.iloc[intensity_train_idx]["reuse_intensity"].astype(int).to_numpy()
            y_intensity_valid = frame.iloc[intensity_valid_idx]["reuse_intensity"].astype(int).to_numpy()
            intensity_model = LogisticRegression(
                solver="saga",
                class_weight=class_weight,
                max_iter=intensity_max_iter,
                tol=intensity_tolerance,
                random_state=random_seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                intensity_model.fit(train_matrix[positive_train_mask], y_intensity_train)
            intensity_converged = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            intensity_predictions = intensity_model.predict(valid_matrix[positive_valid_mask])
            train_class_rates = np.bincount(y_intensity_train, minlength=4) / len(y_intensity_train)
            valid_class_rates = np.bincount(y_intensity_valid, minlength=4) / len(y_intensity_valid)
            records.append(
                {
                    "ablation": ablation_name,
                    "task": "conditional_intensity",
                    "fold": fold,
                    "class_weight": class_weight or "none",
                    "train_rows": len(intensity_train_idx),
                    "validation_rows": len(intensity_valid_idx),
                    "train_min_class_rate": float(train_class_rates.min()),
                    "validation_min_class_rate": float(valid_class_rates.min()),
                    "train_intensity_class_rates": json.dumps(train_class_rates.tolist()),
                    "validation_intensity_class_rates": json.dumps(valid_class_rates.tolist()),
                    "group_overlap": split_group_overlap,
                    "iterations": int(np.max(intensity_model.n_iter_)),
                    "converged": intensity_converged,
                    "solver": "saga",
                    "tolerance": intensity_tolerance,
                    **_intensity_metrics(y_intensity_valid, intensity_predictions),
                }
            )

        pd.DataFrame(records).to_csv(output_dir / "cross_validation_folds.partial.csv", index=False)

    fold_results = pd.DataFrame(records)
    fold_results.to_csv(output_dir / "cross_validation_folds.csv", index=False)
    (output_dir / "cross_validation_folds.partial.csv").unlink(missing_ok=True)
    metric_columns = [
        column for column in fold_results.select_dtypes(include="number").columns
        if column not in {"fold", "train_rows", "validation_rows", "group_overlap"}
    ]
    summary = (
        fold_results.groupby(["ablation", "task", "class_weight"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "cross_validation_summary.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "StratifiedGroupKFold on external training split; test split untouched",
        "ablation": ablation_name,
        "use_text": use_text,
        "use_structural": use_structural,
        "use_repository": use_repository,
        "feature_pipeline": f"Ablation '{ablation_name}': text={use_text}, structural={use_structural}, repo={use_repository}",
        "model": "LogisticRegression",
        "intensity_solver": "saga",
        "intensity_max_iter": intensity_max_iter,
        "intensity_tolerance": intensity_tolerance,
        "class_weight_comparison": ["none", "balanced"],
        "n_splits": n_splits,
        "random_seed": random_seed,
        "rows_used": len(frame),
        "external_test_rows_used": 0,
        "converged_fits": int(fold_results["converged"].sum()),
        "total_fits": int(len(fold_results)),
        "cache_key": cache_key,
        "cache_directory": str(cache_dir),
        "group_column": "representative_repo_id",
        "results_file": str(output_dir / "cross_validation_folds.csv"),
        "summary_file": str(output_dir / "cross_validation_summary.csv"),
    }
    (output_dir / "cross_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest


def compile_ablation_summary(output_dir: Path) -> pd.DataFrame:
    """Consolida os resumos de todas as ablacoes executadas."""
    summaries: list[pd.DataFrame] = []
    ablation_dir = output_dir if output_dir.name == "ablation" else output_dir / "ablation"
    if ablation_dir.is_dir():
        for sub in sorted(ablation_dir.iterdir()):
            summary_file = sub / "cross_validation_summary.csv"
            if summary_file.is_file():
                df = pd.read_csv(summary_file)
                if "ablation" not in df.columns:
                    df.insert(0, "ablation", sub.name)
                summaries.append(df)

    main_summary = output_dir / "cross_validation_summary.csv"
    if main_summary.is_file():
        df = pd.read_csv(main_summary)
        if "ablation" not in df.columns:
            df.insert(0, "ablation", "all")
        if not any((s["ablation"] == "all").any() for s in summaries if "ablation" in s.columns):
            summaries.append(df)

    refit_summary = output_dir / "convergence_refit" / "cross_validation_summary.csv"
    if refit_summary.is_file():
        if not any((s["ablation"] == "all").any() for s in summaries if "ablation" in s.columns):
            df = pd.read_csv(refit_summary)
            if "ablation" not in df.columns:
                df.insert(0, "ablation", "all")
            summaries.append(df)

    if not summaries:
        return pd.DataFrame()

    combined = pd.concat(summaries, ignore_index=True)
    summary_path = (output_dir if output_dir.name != "ablation" else output_dir.parent) / "ablation_summary.csv"
    combined.to_csv(summary_path, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--intensity-max-iter", type=int, default=5_000)
    parser.add_argument("--intensity-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", "text_only", "structure_only", "repo_only", "suite", "fast_ablation"],
        help="Configuracao de ablacao a executar: all, text_only, structure_only, repo_only, suite, fast_ablation",
    )
    args = parser.parse_args()

    ablations_to_run: list[str] = []
    if args.ablation == "suite":
        ablations_to_run = ["structure_only", "repo_only", "all", "text_only"]
    elif args.ablation == "fast_ablation":
        ablations_to_run = ["structure_only", "repo_only"]
    else:
        ablations_to_run = [args.ablation]

    source_sig = _source_signature([args.sample, args.features, args.targets])
    overall_start = time.monotonic()
    manifests = {}

    for name in ablations_to_run:
        print(f"\n==========================================", flush=True)
        print(f"Iniciando ablacao: {name}", flush=True)
        print(f"==========================================", flush=True)
        cfg = ABLATION_CONFIGS[name]
        start = time.monotonic()
        frame = load_modeling_frame(
            args.sample,
            args.features,
            args.targets,
            need_text=cfg["use_text"],
        )
        target_dir = args.output_dir / "ablation" / name if (args.ablation in ("suite", "fast_ablation") or name != "all") else args.output_dir
        manifest = run_cross_validation(
            frame,
            output_dir=target_dir,
            n_splits=args.folds,
            random_seed=args.seed,
            use_text=cfg["use_text"],
            use_structural=cfg["use_structural"],
            use_repository=cfg["use_repository"],
            ablation_name=name,
            max_features=args.max_features,
            min_df=args.min_df,
            intensity_max_iter=args.intensity_max_iter,
            intensity_tolerance=args.intensity_tolerance,
            cache_identity=source_sig,
        )
        manifest["runtime_seconds"] = round(time.monotonic() - start, 2)
        (target_dir / "cross_validation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        manifests[name] = manifest

    summary_df = compile_ablation_summary(args.output_dir)
    print("\nResumo consolidado das ablacoes:")
    if not summary_df.empty:
        key_cols = [
            c for c in [
                "ablation", "task", "class_weight",
                "average_precision_mean", "roc_auc_mean", "balanced_accuracy_at_0_5_mean",
                "f1_at_0_5_mean", "macro_f1_mean", "balanced_accuracy_mean"
            ] if c in summary_df.columns
        ]
        print(summary_df[key_cols].to_string(index=False))

    print(f"\nTempo total decorrido: {time.monotonic() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
