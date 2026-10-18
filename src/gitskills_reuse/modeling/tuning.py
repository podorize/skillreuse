"""Otimizacao de Limiar de Decisao (Threshold Tuning) e Calibracao de Probabilidade.

Avalia as probabilidades preditas fora da dobra (out-of-fold) sob StratifiedGroupKFold,
encontra os limiares operacionais otimos (max F1 e Youden J) e gera curvas de calibracao.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack, load_npz
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.train import (
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT,
    DEFAULT_SAMPLE,
    DEFAULT_TARGETS,
    load_modeling_frame,
)


def run_threshold_tuning(
    *,
    sample_path: Path = DEFAULT_SAMPLE,
    features_path: Path = DEFAULT_FEATURES,
    targets_path: Path = DEFAULT_TARGETS,
    output_base: Path = DEFAULT_OUTPUT,
    sr_cache_dir: Path | None = None,
    text_cache_dir: Path | None = None,
    top_k_text: int = 100,
    n_splits: int = 3,
    random_seed: int = 20261009,
) -> dict[str, object]:
    """Executa a coleta out-of-fold, varredura de thresholds e calibracao."""
    figures_dir = output_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dataset para calibracao e threshold tuning...", flush=True)
    t0 = time.monotonic()
    frame = load_modeling_frame(sample_path, features_path, targets_path, need_text=False)
    y_binary = frame["reused"].astype(int).to_numpy()
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].to_numpy()

    if sr_cache_dir is None:
        sr_candidates = list(
            (output_base / "ablation" / "structure_and_repo" / "cache").glob("*/fold_1_train.npz")
        )
        if not sr_candidates:
            raise FileNotFoundError("Cache de structure_and_repo nao encontrado.")
        sr_cache_dir = sr_candidates[0].parent

    if text_cache_dir is None:
        txt_candidates = list(
            (output_base / "ablation" / "text_only" / "cache").glob("*/fold_1_train.npz")
        )
        if not txt_candidates:
            raise FileNotFoundError("Cache de text_only nao encontrado.")
        text_cache_dir = txt_candidates[0].parent

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    oof_y: list[int] = []
    oof_p_none: list[float] = []
    oof_p_bal: list[float] = []

    print(f"Gerando previsoes fora da dobra (OOF) em {n_splits} folds...", flush=True)

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y=strata, groups=groups), start=1
    ):
        f_start = time.monotonic()
        sr_tr = load_npz(sr_cache_dir / f"fold_{fold}_train.npz")
        sr_val = load_npz(sr_cache_dir / f"fold_{fold}_validation.npz")
        txt_tr = load_npz(text_cache_dir / f"fold_{fold}_train.npz")
        txt_val = load_npz(text_cache_dir / f"fold_{fold}_validation.npz")

        sel = SelectKBest(score_func=chi2, k=top_k_text)
        t_tr = sel.fit_transform(txt_tr, y_binary[train_idx])
        t_val = sel.transform(txt_val)

        X_tr = hstack([sr_tr, t_tr])
        X_val = hstack([sr_val, t_val])

        # Modelo Unweighted
        lr_none = LogisticRegression(max_iter=1000, class_weight=None, random_state=random_seed)
        lr_none.fit(X_tr, y_binary[train_idx])
        p_none = lr_none.predict_proba(X_val)[:, 1]

        # Modelo Balanced
        lr_bal = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_seed
        )
        lr_bal.fit(X_tr, y_binary[train_idx])
        p_bal = lr_bal.predict_proba(X_val)[:, 1]

        oof_y.extend(y_binary[valid_idx])
        oof_p_none.extend(p_none)
        oof_p_bal.extend(p_bal)

        print(f"Fold {fold} OOF computado em {time.monotonic() - f_start:.2f}s", flush=True)

    y_true = np.array(oof_y)
    p_none = np.array(oof_p_none)
    p_bal = np.array(oof_p_bal)

    # 1. Metricas de Calibracao (Brier Score)
    brier_none = float(brier_score_loss(y_true, p_none))
    brier_bal = float(brier_score_loss(y_true, p_bal))
    roc_none = float(roc_auc_score(y_true, p_none))
    pr_none = float(average_precision_score(y_true, p_none))

    print(f"\nBrier Score Loss (None):     {brier_none:.4f}")
    print(f"Brier Score Loss (Balanced): {brier_bal:.4f}")
    print(f"Overall OOF ROC-AUC:         {roc_none:.4f}")
    print(f"Overall OOF PR-AUC:          {pr_none:.4f}")

    # 2. Varredura de Limiares (Threshold Sweep)
    thresholds = np.linspace(0.05, 0.90, 86)
    sweep_records: list[dict[str, object]] = []

    for t in thresholds:
        # Modelo unweighted
        pred_none = p_none >= t
        f1_n = float(f1_score(y_true, pred_none, zero_division=0))
        prec_n = float(precision_score(y_true, pred_none, zero_division=0))
        rec_n = float(recall_score(y_true, pred_none, zero_division=0))
        bacc_n = float(balanced_accuracy_score(y_true, pred_none))
        tn, fp, fn, tp = confusion_matrix(y_true, pred_none).ravel()
        spec_n = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Modelo balanced
        pred_bal = p_bal >= t
        f1_b = float(f1_score(y_true, pred_bal, zero_division=0))
        prec_b = float(precision_score(y_true, pred_bal, zero_division=0))
        rec_b = float(recall_score(y_true, pred_bal, zero_division=0))
        bacc_b = float(balanced_accuracy_score(y_true, pred_bal))
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, pred_bal).ravel()
        spec_b = float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0

        sweep_records.append(
            {
                "threshold": round(float(t), 3),
                "none_f1": f1_n,
                "none_precision": prec_n,
                "none_recall": rec_n,
                "none_balanced_accuracy": bacc_n,
                "none_specificity": spec_n,
                "none_youden_j": (rec_n + spec_n - 1.0),
                "balanced_f1": f1_b,
                "balanced_precision": prec_b,
                "balanced_recall": rec_b,
                "balanced_balanced_accuracy": bacc_b,
                "balanced_specificity": spec_b,
                "balanced_youden_j": (rec_b + spec_b - 1.0),
            }
        )

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(output_base / "threshold_tuning_curves.csv", index=False)

    # Identificar pontos otimos
    best_f1_idx_none = sweep_df["none_f1"].idxmax()
    opt_t_f1_none = float(sweep_df.loc[best_f1_idx_none, "threshold"])
    opt_f1_none = float(sweep_df.loc[best_f1_idx_none, "none_f1"])

    best_j_idx_none = sweep_df["none_youden_j"].idxmax()
    opt_t_j_none = float(sweep_df.loc[best_j_idx_none, "threshold"])
    opt_j_bacc_none = float(sweep_df.loc[best_j_idx_none, "none_balanced_accuracy"])

    best_f1_idx_bal = sweep_df["balanced_f1"].idxmax()
    opt_t_f1_bal = float(sweep_df.loc[best_f1_idx_bal, "threshold"])
    opt_f1_bal = float(sweep_df.loc[best_f1_idx_bal, "balanced_f1"])

    def_f1_none = float(sweep_df.loc[sweep_df["threshold"] == 0.50, "none_f1"].iloc[0])
    def_f1_bal = float(sweep_df.loc[sweep_df["threshold"] == 0.50, "balanced_f1"].iloc[0])

    print("\n=== RESULTADOS DO THRESHOLD TUNING ===")
    print("Modelo Unweighted (calibrado natural):")
    print(f"  Threshold padrao 0.50:  F1 = {def_f1_none:.4f}")
    print(
        f"  Threshold otimo max F1 ({opt_t_f1_none:.2f}): F1 = {opt_f1_none:.4f} "
        f"(+{opt_f1_none - def_f1_none:.4f})"
    )
    print(f"  Threshold otimo Youden J ({opt_t_j_none:.2f}): Balanced Acc = {opt_j_bacc_none:.4f}")
    print("Modelo Balanced:")
    print(f"  Threshold padrao 0.50:  F1 = {def_f1_bal:.4f}")
    print(f"  Threshold otimo max F1 ({opt_t_f1_bal:.2f}): F1 = {opt_f1_bal:.4f}")

    # Salvar Sumario Consolidado
    summary_data = [
        {
            "model": "unweighted_natural",
            "default_threshold": 0.50,
            "default_f1": def_f1_none,
            "default_precision": float(
                sweep_df.loc[sweep_df["threshold"] == 0.50, "none_precision"].iloc[0]
            ),
            "default_recall": float(
                sweep_df.loc[sweep_df["threshold"] == 0.50, "none_recall"].iloc[0]
            ),
            "default_balanced_accuracy": float(
                sweep_df.loc[sweep_df["threshold"] == 0.50, "none_balanced_accuracy"].iloc[0]
            ),
            "optimal_f1_threshold": opt_t_f1_none,
            "optimal_f1": opt_f1_none,
            "optimal_f1_precision": float(sweep_df.loc[best_f1_idx_none, "none_precision"]),
            "optimal_f1_recall": float(sweep_df.loc[best_f1_idx_none, "none_recall"]),
            "optimal_f1_balanced_accuracy": float(
                sweep_df.loc[best_f1_idx_none, "none_balanced_accuracy"]
            ),
            "brier_score": brier_none,
        },
        {
            "model": "balanced_weighted",
            "default_threshold": 0.50,
            "default_f1": def_f1_bal,
            "default_precision": float(
                sweep_df.loc[sweep_df["threshold"] == 0.50, "balanced_precision"].iloc[0]
            ),
            "default_recall": float(
                sweep_df.loc[sweep_df["threshold"] == 0.50, "balanced_recall"].iloc[0]
            ),
            "default_balanced_accuracy": float(
                sweep_df.loc[sweep_df["threshold"] == 0.50, "balanced_balanced_accuracy"].iloc[0]
            ),
            "optimal_f1_threshold": opt_t_f1_bal,
            "optimal_f1": opt_f1_bal,
            "optimal_f1_precision": float(sweep_df.loc[best_f1_idx_bal, "balanced_precision"]),
            "optimal_f1_recall": float(sweep_df.loc[best_f1_idx_bal, "balanced_recall"]),
            "optimal_f1_balanced_accuracy": float(
                sweep_df.loc[best_f1_idx_bal, "balanced_balanced_accuracy"]
            ),
            "brier_score": brier_bal,
        },
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_base / "threshold_tuning_summary.csv", index=False)

    # 3. Gerar Graficos Academicos
    _plot_tuning_curves(figures_dir, sweep_df, opt_t_f1_none, opt_f1_none)
    _plot_calibration_curves(figures_dir, y_true, p_none, p_bal, brier_none, brier_bal)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "OutOfFold threshold tuning on 160k train instances; test set untouched",
        "optimal_threshold_unweighted": opt_t_f1_none,
        "optimal_f1_unweighted": opt_f1_none,
        "optimal_threshold_balanced": opt_t_f1_bal,
        "optimal_f1_balanced": opt_f1_bal,
        "brier_score_unweighted": brier_none,
        "brier_score_balanced": brier_bal,
        "frozen_decision_rule": (
            f"predict 1 if P(reused=1|x) >= {opt_t_f1_none:.2f} on unweighted calibrated model"
        ),
    }
    (output_base / "threshold_tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"\nThreshold tuning concluido em {time.monotonic() - t0:.2f}s! Salvo em {figures_dir}",
        flush=True,
    )
    return manifest


def _plot_tuning_curves(
    figures_dir: Path,
    sweep_df: pd.DataFrame,
    opt_t: float,
    opt_f1: float,
) -> None:
    """Plota as curvas de Precision, Recall, F1 e Balanced Accuracy por threshold."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    t = sweep_df["threshold"]
    ax.plot(t, sweep_df["none_f1"], label="F1-Score", color="#2563eb", linewidth=2.5)
    ax.plot(
        t,
        sweep_df["none_precision"],
        label="Precisão",
        color="#10b981",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        t,
        sweep_df["none_recall"],
        label="Revocação (Recall)",
        color="#f59e0b",
        linewidth=2,
        linestyle=":",
    )
    ax.plot(
        t,
        sweep_df["none_balanced_accuracy"],
        label="Acurácia Balanceada",
        color="#8b5cf6",
        linewidth=1.8,
        linestyle="-.",
    )

    # Marcar o threshold otimo
    ax.axvline(
        x=opt_t,
        color="#dc2626",
        linestyle="--",
        alpha=0.8,
        label=f"Threshold Ótimo ($t^* = {opt_t:.2f}$)",
    )
    ax.axvline(
        x=0.50, color="#64748b", linestyle=":", alpha=0.6, label="Threshold Padrão ($t = 0.50$)"
    )

    ax.scatter([opt_t], [opt_f1], color="#dc2626", s=80, zorder=5)
    ax.annotate(
        f"Máx F1 = {opt_f1:.3f}\n(em t={opt_t:.2f})",
        xy=(opt_t, opt_f1),
        xytext=(opt_t + 0.05, opt_f1 + 0.05),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#dc2626", alpha=0.9),
    )

    ax.set_xlabel("Limiar de Decisão de Probabilidade ($t$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Métrica de Avaliação", fontsize=12, fontweight="bold")
    ax.set_title(
        "Otimização de Limiar de Decisão (Threshold Tuning) no Modelo Híbrido",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_tuning_curves.png")
    plt.close()


def _plot_calibration_curves(
    figures_dir: Path,
    y_true: np.ndarray,
    p_none: np.ndarray,
    p_bal: np.ndarray,
    brier_none: float,
    brier_bal: float,
) -> None:
    """Gera o diagrama de confiabilidade (Reliability Diagram) comparando a calibracao."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Diagonal ideal perfeitamente calibrada
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfeitamente Calibrado")

    prob_true_none, prob_pred_none = calibration_curve(y_true, p_none, n_bins=10)
    ax.plot(
        prob_pred_none,
        prob_true_none,
        marker="o",
        linewidth=2.2,
        color="#2563eb",
        label=f"Modelo Natural (Brier Loss = {brier_none:.4f})",
    )

    prob_true_bal, prob_pred_bal = calibration_curve(y_true, p_bal, n_bins=10)
    ax.plot(
        prob_pred_bal,
        prob_true_bal,
        marker="s",
        linewidth=2.0,
        color="#f59e0b",
        linestyle="-.",
        label=f"Modelo Ponderado/Balanced (Brier Loss = {brier_bal:.4f})",
    )

    ax.set_xlabel("Probabilidade Média Predita", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fração Real de Positivos Observada", fontsize=12, fontweight="bold")
    ax.set_title(
        "Diagrama de Confiabilidade e Calibração de Probabilidade",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_reliability_diagram.png")
    plt.close()


def main() -> None:
    run_threshold_tuning()


if __name__ == "__main__":
    main()
