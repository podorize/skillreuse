"""Script de reproducao oficial a partir dos caches esparsos.

Carrega as matrizes esparsas pre-processadas em data/processed/cache_sparse,
treina os modelos finalistas campeoes, avalia as metricas cegas no conjunto
de teste externo de 40.000 instancias e gera todas as tabelas e figuras oficiais.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from gitskills_reuse.modeling.cache_builder import DEFAULT_CACHE_DIR, build_and_save_cache
from gitskills_reuse.modeling.train import DEFAULT_FEATURES, _intensity_metrics

DEFAULT_OUTPUT_DIR = Path("reports/modeling/final_test_evaluation")
DEFAULT_FIGURES_DIR = Path("reports/modeling/figures")
FROZEN_OPTIMAL_THRESHOLD = 0.18872562172994545
DEFAULT_OOF_PREDICTIONS = Path("reports/modeling/model_exploration_oof_predictions.csv")


def _oof_thresholds(
    oof_path: Path,
    *,
    fallback_threshold: float,
    recall_targets: tuple[float, ...] = (0.80, 0.90),
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Escolhe limiares apenas com probabilidades OOF do treino externo."""
    if not oof_path.is_file():
        return {"oof_max_f1": fallback_threshold}, {}
    oof = pd.read_csv(oof_path)
    if not {"y_true", "probability"}.issubset(oof.columns):
        raise ValueError(f"Arquivo OOF sem colunas requeridas: {oof_path}")
    y = oof["y_true"].astype(int).to_numpy()
    p = oof["probability"].astype(float).to_numpy()
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_idx = int(np.nanargmax(f1))
    selected = {"oof_max_f1": float(thresholds[best_idx])}
    summaries = {
        "oof_max_f1": {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1[best_idx]),
        }
    }
    for target in recall_targets:
        eligible = np.flatnonzero(recall[:-1] >= target)
        if eligible.size == 0:
            continue
        idx = int(eligible[-1])  # maior limiar que ainda atinge a sensibilidade alvo
        key = f"oof_recall_ge_{target:.2f}"
        selected[key] = float(thresholds[idx])
        summaries[key] = {
            "target_recall": float(target),
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }
    return selected, summaries


def _binary_operating_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def _cluster_bootstrap_explanation_stability(
    X_train: object,
    y_train: np.ndarray,
    feature_names: list[str],
    repo_ids: np.ndarray,
    reference_model: LogisticRegression,
    reference_rows: np.ndarray,
    *,
    repeats: int = 20,
    random_seed: int = 20261009,
) -> pd.DataFrame:
    """Estima estabilidade de coeficientes/SHAP com bootstrap por repositorio.

    A lista de atributos ja foi selecionada no treino completo; a analise e
    condicional a esses 100 termos e nao mede estabilidade da selecao chi2.
    """
    if repeats < 2:
        raise ValueError("Use ao menos duas replicacoes para estabilidade")
    if len(repo_ids) != X_train.shape[0]:
        raise ValueError("Identificadores de repositorio nao estao alinhados ao treino")
    unique_repos, inverse = np.unique(repo_ids.astype(str), return_inverse=True)
    rng = np.random.default_rng(random_seed)
    n_features = X_train.shape[1]
    reference_coef = reference_model.coef_[0]
    mean_train = np.asarray(X_train.mean(axis=0)).ravel()
    x_reference = X_train[reference_rows].toarray()
    centered_reference = x_reference - mean_train
    coefficient_samples = np.empty((repeats, n_features), dtype=np.float32)
    importance_samples = np.empty((repeats, n_features), dtype=np.float32)
    for rep in range(repeats):
        sampled = rng.integers(0, len(unique_repos), size=len(unique_repos))
        repo_counts = np.bincount(sampled, minlength=len(unique_repos))
        sample_weight = repo_counts[inverse].astype(np.float64)
        model = LogisticRegression(
            solver="liblinear", max_iter=1_000, class_weight=None,
            random_state=random_seed + rep,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        coef = model.coef_[0]
        coefficient_samples[rep] = coef
        importance_samples[rep] = np.abs(centered_reference * coef).mean(axis=0)
    ref_sign = np.sign(reference_coef)
    same_sign = (np.sign(coefficient_samples) == ref_sign[None, :]).mean(axis=0)
    ranks = np.argsort(np.argsort(-importance_samples, axis=1), axis=1) + 1
    return pd.DataFrame({
        "feature": feature_names,
        "full_train_coefficient": reference_coef,
        "bootstrap_coefficient_median": np.median(coefficient_samples, axis=0),
        "bootstrap_coefficient_ci_2_5": np.quantile(coefficient_samples, 0.025, axis=0),
        "bootstrap_coefficient_ci_97_5": np.quantile(coefficient_samples, 0.975, axis=0),
        "coefficient_sign_consistency": same_sign,
        "full_train_mean_abs_shap": np.abs(centered_reference * reference_coef).mean(axis=0),
        "bootstrap_mean_abs_shap_median": np.median(importance_samples, axis=0),
        "bootstrap_shap_ci_2_5": np.quantile(importance_samples, 0.025, axis=0),
        "bootstrap_shap_ci_97_5": np.quantile(importance_samples, 0.975, axis=0),
        "bootstrap_top20_frequency": (ranks <= 20).mean(axis=0),
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "representative repository",
    }).sort_values("full_train_mean_abs_shap", ascending=False)


def _linear_contributions_by_outcome(
    X_test: object,
    y_test: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    feature_names: list[str],
    model: LogisticRegression,
    mean_train: np.ndarray,
    *,
    threshold_rule: str,
    batch_size: int = 5_000,
) -> pd.DataFrame:
    """Resume contribuicoes aditivas do modelo por TP/FP/FN/TN no teste.

    Valores sao contribuicoes lineares relativas a media do treino, nao efeitos causais.
    """
    pred = probabilities >= threshold
    outcome = np.select(
        [(y_test == 1) & pred, (y_test == 0) & pred,
         (y_test == 1) & ~pred, (y_test == 0) & ~pred],
        ["true_positive", "false_positive", "false_negative", "true_negative"],
        default="unclassified",
    )
    coef = model.coef_[0]
    groups = ("true_positive", "false_positive", "false_negative", "true_negative")
    sum_abs = {name: np.zeros(X_test.shape[1], dtype=np.float64) for name in groups}
    sum_signed = {name: np.zeros(X_test.shape[1], dtype=np.float64) for name in groups}
    counts = {name: 0 for name in groups}
    for start in range(0, X_test.shape[0], batch_size):
        stop = min(start + batch_size, X_test.shape[0])
        dense = X_test[start:stop].toarray()
        contributions = (dense - mean_train) * coef
        for name in groups:
            mask = outcome[start:stop] == name
            if mask.any():
                sum_abs[name] += np.abs(contributions[mask]).sum(axis=0)
                sum_signed[name] += contributions[mask].sum(axis=0)
                counts[name] += int(mask.sum())
    rows = []
    for name in groups:
        if counts[name] == 0:
            continue
        for idx, feature in enumerate(feature_names):
            rows.append({
                "prediction_outcome": name,
                "n_test_rows": counts[name],
                "feature": feature,
                "mean_abs_linear_contribution": sum_abs[name][idx] / counts[name],
                "mean_signed_linear_contribution": sum_signed[name][idx] / counts[name],
                "threshold": float(threshold),
                "threshold_rule": threshold_rule,
            })
    return pd.DataFrame(rows).sort_values(
        ["prediction_outcome", "mean_abs_linear_contribution"],
        ascending=[True, False],
    )


def reproduce_study(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    from_scratch: bool = False,
    text_max_features: int = 10_000,
    frozen_threshold: float | None = None,
    oof_predictions_path: Path | None = None,
    stability_repeats: int = 20,
    random_seed: int = 20261009,
) -> pd.DataFrame:
    """Executa novamente a avaliacao externa e atualiza tabelas, figuras e manifesto."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    print("=" * 70, flush=True)
    print("SKILLREUSE: REPRODUCAO RAPIDA DOS RESULTADOS DO ARTIGO CIENTIFICO", flush=True)
    print("=" * 70, flush=True)

    if from_scratch or not (cache_dir / "X_train_hybrid.npz").exists():
        print("Cache ausente ou flag --from-scratch ativada. Construindo cache...", flush=True)
        build_and_save_cache(
            cache_dir=cache_dir,
            text_max_features=text_max_features,
        )

    print("\n[1/4] Carregando matrizes esparsas pre-processadas...", flush=True)
    t_load = time.monotonic()
    X_tr_st = load_npz(cache_dir / "X_train_st.npz")
    X_te_st = load_npz(cache_dir / "X_test_st.npz")

    X_tr_sr = load_npz(cache_dir / "X_train_sr.npz")
    X_te_sr = load_npz(cache_dir / "X_test_sr.npz")

    X_tr_hyb = load_npz(cache_dir / "X_train_hybrid.npz")
    X_te_hyb = load_npz(cache_dir / "X_test_hybrid.npz")

    y_tr_bin = np.load(cache_dir / "y_train_binary.npy")
    y_te_bin = np.load(cache_dir / "y_test_binary.npy")
    y_tr_int = np.load(cache_dir / "y_train_intensity.npy")
    y_te_int = np.load(cache_dir / "y_test_intensity.npy")
    train_ids_path = cache_dir / "train_identifiers.csv"
    test_ids_path = cache_dir / "test_identifiers.csv"
    if not train_ids_path.exists() or not test_ids_path.exists():
        split_ids = pq.read_table(
            DEFAULT_FEATURES,
            columns=["content_hash", "representative_repo_id", "split"],
        ).to_pandas()
        for split_name in ("train", "test"):
            split_ids.loc[split_ids["split"].eq(split_name),
                          ["content_hash", "representative_repo_id"]].to_csv(
                cache_dir / f"{split_name}_identifiers.csv", index=False
            )
    train_ids = pd.read_csv(train_ids_path) if train_ids_path.exists() else None
    test_ids = pd.read_csv(test_ids_path) if test_ids_path.exists() else None
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if oof_predictions_path is None:
        oof_predictions_path = DEFAULT_OOF_PREDICTIONS
    oof_threshold_map, oof_threshold_summary = _oof_thresholds(
        oof_predictions_path,
        fallback_threshold=FROZEN_OPTIMAL_THRESHOLD,
    )
    if frozen_threshold is None:
        frozen_threshold = oof_threshold_map["oof_max_f1"]
    else:
        oof_threshold_map["user_supplied_threshold"] = float(frozen_threshold)
        oof_threshold_summary["user_supplied_threshold"] = {
            "threshold": float(frozen_threshold),
            "selection": "explicit command-line override",
        }

    print(
        f"Matrizes carregadas em {time.monotonic() - t_load:.2f}s! "
        f"Treino: {X_tr_hyb.shape}, Teste: {X_te_hyb.shape}",
        flush=True,
    )

    # Modelos a avaliar: (nome, X_train, X_test)
    configs = [
        ("structure_only", X_tr_st, X_te_st),
        ("structure_and_repo", X_tr_sr, X_te_sr),
        ("hybrid_screened", X_tr_hyb, X_te_hyb),
    ]

    print("\n[2/4] Treinando modelos lineares e avaliando no Teste Cego...", flush=True)
    results: list[dict[str, object]] = []
    predictions_dict: dict[str, np.ndarray] = {}
    hybrid_linear_model: LogisticRegression | None = None

    for cfg_name, X_tr, X_te in configs:
        t_model = time.monotonic()
        # Modelo sem ponderacao; conserva a prevalencia natural do treino.
        lr_unweighted = LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight=None, random_state=random_seed
        )
        lr_unweighted.fit(X_tr, y_tr_bin)
        if cfg_name == "hybrid_screened":
            hybrid_linear_model = lr_unweighted
        probs_unw = lr_unweighted.predict_proba(X_te)[:, 1]
        predictions_dict[f"{cfg_name}_unweighted"] = probs_unw

        roc = float(roc_auc_score(y_te_bin, probs_unw))
        pr = float(average_precision_score(y_te_bin, probs_unw))
        brier = float(brier_score_loss(y_te_bin, probs_unw))

        # Avaliacao no limiar padrao (0.50)
        pred_def = (probs_unw >= 0.50).astype(int)
        f1_def = float(f1_score(y_te_bin, pred_def, zero_division=0))
        rec_def = float(recall_score(y_te_bin, pred_def, zero_division=0))
        prec_def = float(precision_score(y_te_bin, pred_def, zero_division=0))
        bal_acc_def = float(balanced_accuracy_score(y_te_bin, pred_def))

        results.append(
            {
                "model": cfg_name,
                "weighting": "unweighted",
                "threshold_rule": "default_0.50",
                "threshold": 0.50,
                "roc_auc": roc,
                "pr_auc": pr,
                "f1": f1_def,
                "precision": prec_def,
                "recall": rec_def,
                "balanced_accuracy": bal_acc_def,
                "brier_score": brier,
            }
        )

        # Sensitivity analysis at the threshold selected from grouped OOF predictions.
        pred_opt = (probs_unw >= frozen_threshold).astype(int)
        f1_opt = float(f1_score(y_te_bin, pred_opt, zero_division=0))
        rec_opt = float(recall_score(y_te_bin, pred_opt, zero_division=0))
        prec_opt = float(precision_score(y_te_bin, pred_opt, zero_division=0))
        bal_acc_opt = float(balanced_accuracy_score(y_te_bin, pred_opt))

        results.append(
            {
                "model": cfg_name,
                "weighting": "unweighted",
                "threshold_rule": f"frozen_optimal_{frozen_threshold:.2f}",
                "threshold": frozen_threshold,
                "roc_auc": roc,
                "pr_auc": pr,
                "f1": f1_opt,
                "precision": prec_opt,
                "recall": rec_opt,
                "balanced_accuracy": bal_acc_opt,
                "brier_score": brier,
            }
        )

        # Modelo ponderado (class_weight='balanced')
        lr_balanced = LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced", random_state=random_seed
        )
        lr_balanced.fit(X_tr, y_tr_bin)
        probs_bal = lr_balanced.predict_proba(X_te)[:, 1]
        pred_bal = (probs_bal >= 0.50).astype(int)
        roc_bal = float(roc_auc_score(y_te_bin, probs_bal))
        pr_bal = float(average_precision_score(y_te_bin, probs_bal))
        brier_bal = float(brier_score_loss(y_te_bin, probs_bal))

        results.append(
            {
                "model": cfg_name,
                "weighting": "balanced",
                "threshold_rule": "default_0.50",
                "threshold": 0.50,
                "roc_auc": roc_bal,
                "pr_auc": pr_bal,
                "f1": float(f1_score(y_te_bin, pred_bal, zero_division=0)),
                "precision": float(precision_score(y_te_bin, pred_bal, zero_division=0)),
                "recall": float(recall_score(y_te_bin, pred_bal, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_te_bin, pred_bal)),
                "brier_score": brier_bal,
            }
        )
        print(
            f"[{cfg_name}] ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | "
            f"F1 (t*={frozen_threshold}): {f1_opt:.4f} (Rec: {rec_opt*100:.1f}%) "
            f"em {time.monotonic() - t_model:.2f}s",
            flush=True,
        )

    # Limiares de alta sensibilidade definidos no treino via OOF; teste apenas reporta.
    hybrid_scores = predictions_dict["hybrid_screened_unweighted"]
    threshold_operating_rows: list[dict[str, object]] = []
    error_audit_frames: list[pd.DataFrame] = []
    threshold_audit: dict[str, float] = {
        "oof_max_f1": float(frozen_threshold),
        **{
            key: value
            for key, value in oof_threshold_map.items()
            if key.startswith("oof_recall_ge_")
        },
    }
    for rule, threshold in threshold_audit.items():
        metrics = _binary_operating_metrics(y_te_bin, hybrid_scores, threshold)
        oof_meta = oof_threshold_summary.get(rule, {})
        threshold_operating_rows.append({
            "model": "hybrid_screened",
            "threshold_rule": rule,
            "threshold_source": "grouped out-of-fold predictions on outer training split",
            "oof_target_recall": oof_meta.get("target_recall"),
            "oof_precision": oof_meta.get("precision"),
            "oof_recall": oof_meta.get("recall"),
            "oof_f1": oof_meta.get("f1"),
            **metrics,
        })
        pred = (hybrid_scores >= threshold).astype(int)
        if test_ids is not None and len(test_ids) == len(y_te_bin):
            audit = test_ids.copy()
            audit["y_true"] = y_te_bin
            audit["probability"] = hybrid_scores
            audit["threshold_rule"] = rule
            audit["threshold"] = threshold
            audit["y_pred"] = pred
            audit["error_type"] = np.select(
                [(y_te_bin == 1) & (pred == 1), (y_te_bin == 0) & (pred == 1),
                 (y_te_bin == 1) & (pred == 0)],
                ["true_positive", "false_positive", "false_negative"],
                default="true_negative",
            )
            error_audit_frames.append(audit[audit["y_true"] != audit["y_pred"]])
    pd.DataFrame(threshold_operating_rows).to_csv(
        output_dir / "threshold_sensitivity_analysis.csv", index=False
    )
    if error_audit_frames:
        error_audit = pd.concat(error_audit_frames, ignore_index=True)
        error_audit.to_csv(output_dir / "hybrid_test_error_audit.csv", index=False)
        error_summary = (
            error_audit.groupby(["threshold_rule", "error_type"], as_index=False)
            .size().rename(columns={"size": "n_test_rows"})
        )
        error_summary.to_csv(output_dir / "hybrid_test_error_summary.csv", index=False)

    if test_ids is not None and len(test_ids) == len(y_te_bin):
        test_prediction_frame = test_ids.copy()
        test_prediction_frame["y_true"] = y_te_bin
        for prediction_name, values in predictions_dict.items():
            test_prediction_frame[f"probability_{prediction_name}"] = values
        test_prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    # [3/4] Avaliacao da Intensidade Condicional Multiclasse
    print("\n[3/4] Avaliando Intensidade Condicional (Classes 0 a 3)...", flush=True)
    pos_tr_mask = y_tr_bin == 1
    pos_te_mask = y_te_bin == 1

    y_int_tr_pos = y_tr_int[pos_tr_mask]
    y_int_te_pos = y_te_int[pos_te_mask]

    intensity_results: list[dict[str, object]] = []

    for cfg_name, X_tr, X_te in configs:
        lr_int = LogisticRegression(
            solver="saga", max_iter=5000, tol=1e-3,
            class_weight="balanced", random_state=random_seed
        )
        lr_int.fit(X_tr[pos_tr_mask], y_int_tr_pos)
        int_preds = lr_int.predict(X_te[pos_te_mask])
        int_m = _intensity_metrics(y_int_te_pos, int_preds)

        intensity_results.append(
            {
                "model": cfg_name,
                "task": "conditional_intensity",
                "class_weight": "balanced",
                "test_positive_samples": len(y_int_te_pos),
                **int_m,
            }
        )
        c3_rec = int_m["class_3_recall"] * 100
        print(
            f"[{cfg_name} - Intensidade] Macro F1: {int_m['macro_f1']:.4f} | "
            f"Class 3 F1 (10+ repos): {int_m['class_3_f1']:.4f} (Rec: {c3_rec:.1f}%)",
            flush=True,
        )

    # [4/4] Geracao de Graficos e Tabelas Finais
    print("\n[4/4] Salvando relatorios e plotando figuras...", flush=True)
    if hybrid_linear_model is not None:
        feature_names = json.loads((cache_dir / "hybrid_feature_names.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(random_seed)
        explain_idx = rng.choice(X_tr_hyb.shape[0], size=min(1_000, X_tr_hyb.shape[0]), replace=False)
        x_explain = X_tr_hyb[explain_idx].toarray()
        mean_train = np.asarray(X_tr_hyb.mean(axis=0)).ravel()
        contribution_rule = (
            "oof_recall_ge_0.80"
            if "oof_recall_ge_0.80" in threshold_audit
            else "oof_max_f1"
        )
        contribution_threshold = threshold_audit[contribution_rule]
        contributions = _linear_contributions_by_outcome(
            X_te_hyb,
            y_te_bin,
            predictions_dict["hybrid_screened_unweighted"],
            contribution_threshold,
            feature_names,
            hybrid_linear_model,
            mean_train,
            threshold_rule=contribution_rule,
        )
        contributions.to_csv(
            output_dir / "hybrid_linear_contributions_by_outcome.csv", index=False
        )
        linear_shap = (x_explain - mean_train) * hybrid_linear_model.coef_[0]
        shap_summary = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(linear_shap).mean(axis=0),
            "mean_shap": linear_shap.mean(axis=0),
            "coefficient": hybrid_linear_model.coef_[0],
        }).sort_values("mean_abs_shap", ascending=False)
        shap_summary.to_csv(output_dir / "hybrid_linear_shap_summary.csv", index=False)
        if train_ids is not None and "representative_repo_id" in train_ids:
            stability = _cluster_bootstrap_explanation_stability(
                X_tr_hyb,
                y_tr_bin,
                feature_names,
                train_ids["representative_repo_id"].astype(str).to_numpy(),
                hybrid_linear_model,
                explain_idx,
                repeats=stability_repeats,
                random_seed=random_seed,
            )
            stability.to_csv(output_dir / "hybrid_explanation_stability.csv", index=False)
            (output_dir / "hybrid_explanation_stability_manifest.json").write_text(
                json.dumps({
                    "method": "cluster bootstrap by representative repository",
                    "bootstrap_repeats": stability_repeats,
                    "feature_selection": "fixed at the full outer-training chi-square top-100 set",
                    "interpretation": "Coefficient and linear SHAP stability conditional on the selected features; does not estimate stability of text candidate generation or chi-square selection.",
                    "reference_rows": len(explain_idx),
                    "random_seed": random_seed,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        top_shap = shap_summary.head(20).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top_shap["feature"], top_shap["mean_abs_shap"], color="#3b6ea8")
        ax.set_xlabel("Mean absolute SHAP value on the log-odds scale (training sample)")
        ax.set_ylabel("")
        ax.set_title("Top 20 features in the fitted hybrid logistic model")
        fig.tight_layout()
        fig.savefig(figures_dir / "hybrid_shap_top20_en.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "test_binary_results.csv", index=False)

    int_df = pd.DataFrame(intensity_results)
    int_df.to_csv(output_dir / "test_intensity_results.csv", index=False)

    # Matriz de confusao do modelo campeao
    opt_hybrid_preds = (
        predictions_dict["hybrid_screened_unweighted"] >= frozen_threshold
    ).astype(int)
    cm = confusion_matrix(y_te_bin, opt_hybrid_preds)
    cm_dict = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }
    (output_dir / "test_confusion_matrix.json").write_text(
        json.dumps(cm_dict, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "Held-out evaluation on the fixed external split; no model fitting uses test labels",
        "train_rows": int(len(y_tr_bin)),
        "test_rows": int(len(y_te_bin)),
        "test_positive_rows": int(y_te_bin.sum()),
        "frozen_sensitivity_threshold": float(frozen_threshold),
        "threshold_note": (
            "All reported thresholds are selected from grouped out-of-fold predictions on the "
            "outer-training split; the external test labels are used only for evaluation. "
            "The OOF artifact must use the same text configuration as the cache."
        ),
        "oof_predictions_file": str(oof_predictions_path),
        "oof_thresholds": oof_threshold_summary,
        "cache_manifest": str(cache_manifest_path),
        "cache_text_candidate_terms": cache_manifest.get("n_text_candidate_terms"),
        "cache_text_candidate_mode": cache_manifest.get("text_candidate_mode", "outer_training_tfidf"),
        "cache_text_candidate_source": cache_manifest.get("text_candidate_source"),
        "feature_dimensions": {
            "structure": int(X_tr_st.shape[1]),
            "structure_and_repository": int(X_tr_sr.shape[1]),
            "hybrid": int(X_tr_hyb.shape[1]),
        },
        "binary_solver": "liblinear",
        "binary_max_iter": 1000,
        "intensity_solver": "saga",
        "intensity_max_iter": 5000,
        "intensity_tolerance": 0.001,
        "binary_results_file": str(output_dir / "test_binary_results.csv"),
        "intensity_results_file": str(output_dir / "test_intensity_results.csv"),
        "confusion_matrix_file": str(output_dir / "test_confusion_matrix.json"),
        "test_predictions_file": str(output_dir / "test_predictions.csv") if test_ids is not None else None,
        "threshold_sensitivity_file": str(output_dir / "threshold_sensitivity_analysis.csv"),
        "error_audit_file": str(output_dir / "hybrid_test_error_audit.csv") if error_audit_frames else None,
        "explanation_stability_file": str(output_dir / "hybrid_explanation_stability.csv") if (output_dir / "hybrid_explanation_stability.csv").exists() else None,
        "contributions_by_outcome_file": str(output_dir / "hybrid_linear_contributions_by_outcome.csv") if (output_dir / "hybrid_linear_contributions_by_outcome.csv").exists() else None,
        "hybrid_unweighted_default_threshold": res_df.loc[
            (res_df["model"] == "hybrid_screened")
            & (res_df["threshold_rule"] == "default_0.50")
        ].iloc[0].to_dict(),
        "hybrid_unweighted_sensitivity_threshold": res_df.loc[
            (res_df["model"] == "hybrid_screened")
            & (res_df["threshold_rule"] == f"frozen_optimal_{frozen_threshold:.2f}")
        ].iloc[0].to_dict(),
        "hybrid_intensity_results": int_df.loc[
            int_df["model"] == "hybrid_screened"
        ].iloc[0].to_dict(),
        "confusion_matrix": cm_dict,
    }
    (output_dir / "test_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n", encoding="utf-8"
    )

    _plot_figures(figures_dir, y_te_bin, predictions_dict, cm, frozen_threshold)

    elapsed = time.monotonic() - t_start
    print("\n" + "=" * 70, flush=True)
    print(f"REPRODUCAO CONCLUIDA COM SUCESSO EM {elapsed:.2f} SEGUNDOS!", flush=True)
    print("=" * 70, flush=True)
    print("\nTABELA CONSOLIDADA DO TESTE CEGO (Conforme Artigo):")
    cols = [
        "model", "weighting", "threshold", "roc_auc", "pr_auc", "f1", "recall", "balanced_accuracy"
    ]
    sub_df = res_df[
        res_df["threshold_rule"].isin(["default_0.50", f"frozen_optimal_{frozen_threshold:.2f}"])
    ]
    print(sub_df[cols].to_string(index=False))

    return res_df


def _plot_figures(
    figures_dir: Path,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    cm: np.ndarray,
    threshold: float,
) -> None:
    """Gera figuras finais para o artigo em alta resolucao."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = [
        ("structure_only_unweighted", "1. Structure Only", "#94a3b8"),
        ("structure_and_repo_unweighted", "2. Structure + Repo", "#0284c7"),
        ("hybrid_screened_unweighted", "3. Hybrid (Campeão)", "#2563eb"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    # Curva ROC
    for key, label, color in palette:
        fpr, tpr, _ = roc_curve(y_test, predictions[key])
        auc_score = roc_auc_score(y_test, predictions[key])
        axes[0].plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.4f})", color=color, lw=2.2)

    axes[0].plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.5000)", alpha=0.6)
    axes[0].set_title("Curva ROC no Teste Cego Externo", fontsize=13, weight="bold")
    axes[0].set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=11)
    axes[0].set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)

    # Curva Precision-Recall
    baseline_pr = y_test.mean()
    for key, label, color in palette:
        p, r, _ = precision_recall_curve(y_test, predictions[key])
        prauc = average_precision_score(y_test, predictions[key])
        axes[1].plot(r, p, label=f"{label} (PR-AUC = {prauc:.4f})", color=color, lw=2.2)

    axes[1].axhline(
        y=baseline_pr,
        color="k",
        linestyle="--",
        label=f"Prevalência Basal ({baseline_pr:.4f})",
        alpha=0.6,
    )
    axes[1].set_title("Curva Precision-Recall no Teste Cego", fontsize=13, weight="bold")
    axes[1].set_xlabel("Revocação (Recall)", fontsize=11)
    axes[1].set_ylabel("Precisão (Precision)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(figures_dir / "final_test_roc_pr_curves.png", bbox_inches="tight")
    plt.close()

    # Matriz de Confusao
    plt.figure(figsize=(6, 5), dpi=300)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    labels = np.array([
        [f"{cm[0, 0]:,}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]:,}\n({cm_norm[0, 1]:.1%})"],
        [f"{cm[1, 0]:,}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]:,}\n({cm_norm[1, 1]:.1%})"],
    ])
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Não Reusado (0)", "Reusado (1)"],
        yticklabels=["Não Reusado (0)", "Reusado (1)"],
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title(f"Matriz de Confusão no Teste (t* = {threshold:.2f})", fontsize=12, weight="bold")
    plt.xlabel("Classe Predita", fontsize=11)
    plt.ylabel("Classe Real", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_test_confusion_matrix.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillReuse: Reproducao Rapida Oficial")
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Diretorio do cache de matrizes"
    )
    parser.add_argument(
        "--from-scratch", action="store_true", help="Recriar o cache de texto do zero"
    )
    parser.add_argument(
        "--text-max-features", type=int, default=10_000,
        help="Limite de candidatos TF-IDF gerados no treino",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="optional threshold override; default is selected from OOF probabilities"
    )
    args = parser.parse_args()

    reproduce_study(
        cache_dir=args.cache_dir,
        from_scratch=args.from_scratch,
        text_max_features=args.text_max_features,
        frozen_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
