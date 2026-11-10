"""Benchmark comparativo de modelos alternativos adaptados aos dados.

Avalia o espaco de hipoteses sob 3-Fold StratifiedGroupKFold no espaco hibrido
(243--249 features, incluindo termos selecionados dentro de cada dobra):
1. Regressao Logistica L2 (Baseline)
2. Regressao Logistica ElasticNet (L1 + L2 via SAGA)
3. MLPClassifier (Rede Neural Tabular Regularizada)
4. HistGradientBoosting Regularizado (min_samples_leaf=200, max_depth=3)
5. Ensemble por Votacao Suave / Blend
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, load_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from gitskills_reuse.modeling.train import (
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT,
    DEFAULT_SAMPLE,
    DEFAULT_TARGETS,
    _intensity_metrics,
    load_modeling_frame,
)


def run_benchmark(
    *,
    sample_path: Path = DEFAULT_SAMPLE,
    features_path: Path = DEFAULT_FEATURES,
    targets_path: Path = DEFAULT_TARGETS,
    output_base: Path = DEFAULT_OUTPUT,
    top_k_text: int = 100,
    n_splits: int = 3,
    random_seed: int = 20261009,
) -> pd.DataFrame:
    print("=" * 70)
    print("INICIANDO BENCHMARK DE MODELOS ADAPTADOS AOS DADOS")
    print("=" * 70)

    # 1. Carregar dataset
    print("\n[1/4] Carregando dataset de treino...", flush=True)
    t0 = time.monotonic()
    frame = load_modeling_frame(sample_path, features_path, targets_path, need_text=False)
    print(f"Dataset carregado ({len(frame)} linhas) em {time.monotonic() - t0:.2f}s", flush=True)

    y_binary = frame["reused"].astype(int).to_numpy()
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # 2. Caches
    sr_cache_candidates = list((output_base / "ablation" / "structure_and_repo" / "cache").glob("*/fold_1_train.npz"))
    if not sr_cache_candidates:
        raise FileNotFoundError("Cache de structure_and_repo nao encontrado.")
    sr_cache_dir = max(sr_cache_candidates, key=lambda path: path.stat().st_mtime).parent

    txt_cache_candidates = list((output_base / "ablation" / "text_only" / "cache").glob("*/fold_1_train.npz"))
    if not txt_cache_candidates:
        raise FileNotFoundError("Cache de text_only nao encontrado.")
    txt_cache_dir = max(txt_cache_candidates, key=lambda path: path.stat().st_mtime).parent

    print(f"Cache tabular: {sr_cache_dir}")
    print(f"Cache texto:   {txt_cache_dir}")

    # Definir modelos a testar
    # Observacao: SGDClassifier com loss='log_loss' implementa LogisticRegression com ElasticNet de forma ultra-rapida
    models = {
        "1. Logistic L2 (Baseline)": LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1_000, random_state=random_seed
        ),
        "2. ElasticNet (L1+L2 via SGD)": SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5, alpha=1e-4, max_iter=500, random_state=random_seed
        ),
        "3. MLP Neural Net (2-layer, L2)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_seed,
        ),
        "4. Constrained GBDT (Anti-Atalho)": HistGradientBoostingClassifier(
            max_depth=3,
            min_samples_leaf=200,
            learning_rate=0.05,
            max_iter=60,
            random_state=random_seed,
        ),
    }

    results = []
    intensity_results = []
    feature_counts: list[int] = []
    oof_rows: list[pd.DataFrame] = []

    print("\n[2/4] Executando 3-Fold StratifiedGroupKFold...", flush=True)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(frame, y=strata, groups=groups), start=1):
        print(f"\n>>> DOBRA {fold}/{n_splits} (Treino: {len(train_idx)} | Validação: {len(valid_idx)}) <<<", flush=True)

        sr_train = load_npz(sr_cache_dir / f"fold_{fold}_train.npz")
        sr_valid = load_npz(sr_cache_dir / f"fold_{fold}_validation.npz")
        txt_train = load_npz(txt_cache_dir / f"fold_{fold}_train.npz")
        txt_valid = load_npz(txt_cache_dir / f"fold_{fold}_validation.npz")

        # Chi-2 Feature Selection em cada dobra
        selector = SelectKBest(score_func=chi2, k=top_k_text)
        txt_train_sel = selector.fit_transform(txt_train, y_binary[train_idx])
        txt_valid_sel = selector.transform(txt_valid)

        X_tr_sp = hstack([sr_train, txt_train_sel]).tocsr()
        X_val_sp = hstack([sr_valid, txt_valid_sel]).tocsr()
        feature_counts.append(int(X_tr_sp.shape[1]))

        # Versao densa float32 para modelos nao-esparsos
        X_tr_dense = X_tr_sp.toarray().astype(np.float32)
        X_val_dense = X_val_sp.toarray().astype(np.float32)

        y_tr = y_binary[train_idx]
        y_val = y_binary[valid_idx]

        fold_probs = {}

        for name, clf in models.items():
            t_start = time.monotonic()
            is_dense = "MLP" in name or "GBDT" in name
            X_tr = X_tr_dense if is_dense else X_tr_sp
            X_val = X_val_dense if is_dense else X_val_sp

            clf.fit(X_tr, y_tr)
            fit_time = time.monotonic() - t_start

            probs = clf.predict_proba(X_val)[:, 1]
            fold_probs[name] = probs

            roc = roc_auc_score(y_val, probs)
            pr = average_precision_score(y_val, probs)
            f1_default = f1_score(y_val, probs >= 0.50, zero_division=0)

            print(
                f"  [{name:32s}] ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1(0.50): {f1_default:.4f} | Tempo: {fit_time:.2f}s",
                flush=True,
            )

            results.append({
                "fold": fold,
                "model": name,
                "roc_auc": roc,
                "pr_auc": pr,
                "f1_at_0_50": f1_default,
                "feature_count": int(X_tr_sp.shape[1]),
                "fit_time_seconds": fit_time,
            })
            if name == "1. Logistic L2 (Baseline)":
                oof_rows.append(pd.DataFrame({
                    "content_hash": frame.iloc[valid_idx]["content_hash"].to_numpy(),
                    "fold": fold,
                    "y_true": y_val,
                    "probability": probs,
                }))

        # Modelo 5: Ensemble por Votacao Suave (Blend Linear + MLP + GBDT)
        blend_probs = (fold_probs["1. Logistic L2 (Baseline)"] * 0.5 +
                       fold_probs["3. MLP Neural Net (2-layer, L2)"] * 0.25 +
                       fold_probs["4. Constrained GBDT (Anti-Atalho)"] * 0.25)
        blend_roc = roc_auc_score(y_val, blend_probs)
        blend_pr = average_precision_score(y_val, blend_probs)
        blend_f1_default = f1_score(y_val, blend_probs >= 0.50, zero_division=0)

        print(
            f"  [{'5. Heterogeneous Blend (Lin+MLP+GBDT)':32s}] ROC-AUC: {blend_roc:.4f} | PR-AUC: {blend_pr:.4f} | F1(0.50): {blend_f1_default:.4f} | Tempo: blend",
            flush=True,
        )
        results.append({
            "fold": fold,
            "model": "5. Heterogeneous Blend",
            "roc_auc": blend_roc,
            "pr_auc": blend_pr,
            "f1_at_0_50": blend_f1_default,
            "feature_count": int(X_tr_sp.shape[1]),
            "fit_time_seconds": 0.0,
        })

        train_positive = y_tr == 1
        validation_positive = y_val == 1
        y_int_train = frame.iloc[train_idx].loc[train_positive, "reuse_intensity"].astype(int).to_numpy()
        y_int_valid = frame.iloc[valid_idx].loc[validation_positive, "reuse_intensity"].astype(int).to_numpy()
        intensity_model = LogisticRegression(
            solver="saga", class_weight="balanced", max_iter=5_000,
            tol=1e-3, random_state=random_seed,
        )
        intensity_model.fit(X_tr_sp[train_positive], y_int_train)
        intensity_predictions = intensity_model.predict(X_val_sp[validation_positive])
        intensity_results.append({
            "fold": fold,
            "model": "Hybrid Logistic (balanced conditional intensity)",
            "feature_count": int(X_tr_sp.shape[1]),
            "test_positive_rows": int(validation_positive.sum()),
            **_intensity_metrics(y_int_valid, intensity_predictions),
        })

    df_results = pd.DataFrame(results)
    intensity_folds = pd.DataFrame(intensity_results)

    # Consolidar medias e desvios
    summary = df_results.groupby("model").agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        f1_0_50_mean=("f1_at_0_50", "mean"),
        f1_0_50_std=("f1_at_0_50", "std"),
        fit_time_mean=("fit_time_seconds", "mean"),
    ).reset_index()

    summary = summary.sort_values(by="roc_auc_mean", ascending=False)

    print("\n" + "=" * 70)
    print("RESUMO CONSOLIDADO DO BENCHMARK (3-FOLD STRATIFIED GROUP K-FOLD)")
    print("=" * 70)
    print(summary.to_string(index=False))

    out_csv = output_base / "model_exploration_benchmark.csv"
    summary.to_csv(out_csv, index=False)
    intensity_folds_path = output_base / "model_exploration_intensity_folds.csv"
    intensity_folds.to_csv(intensity_folds_path, index=False)
    intensity_metric_columns = [
        column for column in intensity_folds.select_dtypes(include="number").columns
        if column not in {"fold", "feature_count", "test_positive_rows"}
    ]
    intensity_summary = intensity_folds.groupby("model", as_index=False)[intensity_metric_columns].agg(["mean", "std"])
    intensity_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in intensity_summary.columns
    ]
    intensity_summary.to_csv(output_base / "model_exploration_intensity_summary.csv", index=False)
    folds_path = output_base / "model_exploration_benchmark_folds.csv"
    df_results.to_csv(folds_path, index=False)
    oof_predictions = pd.concat(oof_rows, ignore_index=True)
    oof_path = output_base / "model_exploration_oof_predictions.csv"
    oof_predictions.to_csv(oof_path, index=False)
    precision, recall, thresholds = precision_recall_curve(
        oof_predictions["y_true"], oof_predictions["probability"]
    )
    oof_f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_idx = int(np.argmax(oof_f1))
    oof_threshold = float(thresholds[best_idx])
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "Three-fold StratifiedGroupKFold on the 160,000-row external training partition; external test rows excluded",
        "n_splits": n_splits,
        "random_seed": random_seed,
        "group_column": "representative_repo_id",
        "stratification_column": "split_stratum",
        "rows_used": int(len(frame)),
        "external_test_rows_used": 0,
        "tabular_cache": str(sr_cache_dir),
        "text_cache": str(txt_cache_dir),
        "text_selection": {"method": "chi2", "features_per_fold": top_k_text},
        "hybrid_feature_count_by_fold": feature_counts,
        "model_families": list(models) + ["5. Heterogeneous Blend"],
        "threshold_for_f1": 0.5,
        "oof_selected_threshold": oof_threshold,
        "oof_selected_f1": float(oof_f1[best_idx]),
        "oof_threshold_selection": "Maximum F1 over three-fold out-of-fold probabilities from the Logistic L2 model; labels are training-split labels only",
        "summary_file": str(out_csv),
        "fold_results_file": str(folds_path),
        "intensity_fold_results_file": str(intensity_folds_path),
        "intensity_summary_file": "reports/modeling/model_exploration_intensity_summary.csv",
        "oof_predictions_file": str(oof_path),
        "runtime_seconds": round(time.monotonic() - t0, 2),
    }
    (output_base / "model_exploration_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSalvo com sucesso em: {out_csv}")
    return summary


if __name__ == "__main__":
    run_benchmark()
