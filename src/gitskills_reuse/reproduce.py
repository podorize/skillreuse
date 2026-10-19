"""Script de Reproducao Rapida e Oficial (< 45 segundos).

Carrega as matrizes esparsas pre-processadas em data/processed/cache_sparse,
treina os modelos finalistas campeoes, avalia as metricas cegas no conjunto
de teste externo de 40.000 instancias e gera todas as tabelas e figuras oficiais.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from gitskills_reuse.modeling.train import _intensity_metrics

DEFAULT_OUTPUT_DIR = Path("reports/modeling/final_test_evaluation")
DEFAULT_FIGURES_DIR = Path("reports/modeling/figures")
FROZEN_OPTIMAL_THRESHOLD = 0.21


def reproduce_study(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    from_scratch: bool = False,
    frozen_threshold: float = FROZEN_OPTIMAL_THRESHOLD,
    random_seed: int = 20261009,
) -> pd.DataFrame:
    """Executa a reproducao completa dos resultados em menos de 45 segundos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    print("=" * 70, flush=True)
    print("SKILLREUSE: REPRODUCAO RAPIDA DOS RESULTADOS DO ARTIGO CIENTIFICO", flush=True)
    print("=" * 70, flush=True)

    if from_scratch or not (cache_dir / "X_train_hybrid.npz").exists():
        print("Cache ausente ou flag --from-scratch ativada. Construindo cache...", flush=True)
        build_and_save_cache(cache_dir=cache_dir)

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

    for cfg_name, X_tr, X_te in configs:
        t_model = time.monotonic()
        # Modelo sem ponderacao (natural, calibrado)
        lr_unweighted = LogisticRegression(
            max_iter=1000, class_weight=None, random_state=random_seed
        )
        lr_unweighted.fit(X_tr, y_tr_bin)
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

        # Avaliacao no limiar otimo congelado (t* = 0.21)
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
            max_iter=1000, class_weight="balanced", random_state=random_seed
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

    # [3/4] Avaliacao da Intensidade Condicional Multiclasse
    print("\n[3/4] Avaliando Intensidade Condicional (Classes 0 a 3)...", flush=True)
    pos_tr_mask = y_tr_bin == 1
    pos_te_mask = y_te_bin == 1

    y_int_tr_pos = y_tr_int[pos_tr_mask]
    y_int_te_pos = y_te_int[pos_te_mask]

    intensity_results: list[dict[str, object]] = []

    for cfg_name, X_tr, X_te in configs:
        lr_int = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_seed
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
        "--threshold", type=float, default=FROZEN_OPTIMAL_THRESHOLD, help="Limiar otimo congelado"
    )
    args = parser.parse_args()

    reproduce_study(
        cache_dir=args.cache_dir,
        from_scratch=args.from_scratch,
        frozen_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
