"""Avaliacao Final Cega no Conjunto de Teste Externo (40.000 Instancias).

Treina os modelos campeoes em 100% da base de treino (160.000 instancias) e avalia
completamente as metricas cegas no split de teste externo (40.000 instancias de 14.568
repositorios nunca antes vistos), utilizando os limiares operacionais congelados na validacao.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
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

from gitskills_reuse.modeling.preprocess import build_preprocessor, predictor_view
from gitskills_reuse.modeling.train import (
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT,
    DEFAULT_SAMPLE,
    DEFAULT_TARGETS,
    _intensity_metrics,
)


def load_train_and_test_frames(
    sample_path: Path,
    features_path: Path,
    targets_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega treino e teste completos garantindo alinhamento e isolamento estrito."""
    features = pq.read_table(features_path).to_pandas()
    targets = pq.read_table(
        targets_path,
        columns=["content_hash", "reuse_intensity", "split_stratum"],
    ).to_pandas()
    text = pq.read_table(sample_path, columns=["content_hash", "content"]).to_pandas()

    frame = features.merge(targets, on="content_hash", how="inner", validate="one_to_one")
    frame = frame.merge(text, on="content_hash", how="inner", validate="one_to_one")

    train_frame = frame[frame["split"] == "train"].copy().reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].copy().reset_index(drop=True)

    if len(train_frame) != 160_000 or len(test_frame) != 40_000:
        raise RuntimeError(
            f"Particoes com tamanho inesperado: train={len(train_frame)}, test={len(test_frame)}"
        )

    train_repos = set(train_frame["representative_repo_id"])
    test_repos = set(test_frame["representative_repo_id"])
    overlap = train_repos.intersection(test_repos)
    if overlap:
        raise RuntimeError(
            f"Vazamento detectado! {len(overlap)} repositorios compartilhados entre treino e teste."
        )

    return train_frame, test_frame


def run_final_test_evaluation(
    *,
    sample_path: Path = DEFAULT_SAMPLE,
    features_path: Path = DEFAULT_FEATURES,
    targets_path: Path = DEFAULT_TARGETS,
    output_base: Path = DEFAULT_OUTPUT,
    frozen_optimal_threshold: float = 0.21,
    top_k_text: int = 100,
    random_seed: int = 20261009,
) -> dict[str, object]:
    """Executa a avaliacao final dos modelos campeoes no conjunto de teste cego."""
    out_dir = output_base / "final_test_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print("=" * 70, flush=True)
    print("INICIANDO AVALIACAO FINAL CEGA NO CONJUNTO DE TESTE (40.000 INSTANCIAS)", flush=True)
    print("=" * 70, flush=True)

    train_df, test_df = load_train_and_test_frames(sample_path, features_path, targets_path)
    print(f"Treino carregado: {len(train_df)} amostras (57.808 repos)", flush=True)
    print(f"Teste carregado:  {len(test_df)} amostras (14.568 repos 100% ineditos)", flush=True)

    y_train_binary = train_df["reused"].astype(int).to_numpy()
    y_test_binary = test_df["reused"].astype(int).to_numpy()

    tr_rate = y_train_binary.mean()
    te_rate = y_test_binary.mean()
    tr_sum = int(y_train_binary.sum())
    te_sum = int(y_test_binary.sum())
    print(
        f"Taxa de positivos no Treino: {tr_rate:.4f} ({tr_sum}/{len(y_train_binary)})",
        flush=True,
    )
    print(
        f"Taxa de positivos no Teste:  {te_rate:.4f} ({te_sum}/{len(y_test_binary)})",
        flush=True,
    )

    # 1. Pre-processamento Tabular (Estrutura e Repositorio)
    print("\nAjustando transformador tabular de Estrutura + Repositorio no Treino...", flush=True)
    sr_train_pred = predictor_view(
        train_df, use_text=False, use_structural=True, use_repository=True
    )
    sr_test_pred = predictor_view(test_df, use_text=False, use_structural=True, use_repository=True)

    sr_trans = build_preprocessor(
        sr_train_pred, use_text=False, use_structural=True, use_repository=True
    )
    sr_train_sp = sr_trans.fit_transform(sr_train_pred)
    sr_test_sp = sr_trans.transform(sr_test_pred)
    print(f"Matriz Tabular: {sr_train_sp.shape[1]} features", flush=True)

    # 1b. Pre-processamento Estrutural Puro (para baseline structure_only)
    st_train_pred = predictor_view(
        train_df, use_text=False, use_structural=True, use_repository=False
    )
    st_test_pred = predictor_view(
        test_df, use_text=False, use_structural=True, use_repository=False
    )
    st_trans = build_preprocessor(
        st_train_pred, use_text=False, use_structural=True, use_repository=False
    )
    st_train_sp = st_trans.fit_transform(st_train_pred)
    st_test_sp = st_trans.transform(st_test_pred)

    # 2. Pre-processamento e Selecao Textual Chi-2
    print(
        "Iniciando TF-IDF nos 160.000 documentos de Treino (estimativa: ~14 min)...",
        flush=True,
    )
    t_tfidf = time.monotonic()
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    txt_train_raw = vec.fit_transform(train_df["content"])
    print(f"TF-IDF Treino concluido em {time.monotonic() - t_tfidf:.1f}s", flush=True)

    print("Vetorizando 40.000 documentos de Teste (estimativa: ~3 min)...", flush=True)
    t_trans = time.monotonic()
    txt_test_raw = vec.transform(test_df["content"])
    print(f"TF-IDF Teste concluido em {time.monotonic() - t_trans:.1f}s", flush=True)

    vocab = np.array(vec.get_feature_names_out())
    print(f"Selecionando Top-{top_k_text} termos no Treino via Chi-2...", flush=True)
    selector = SelectKBest(score_func=chi2, k=top_k_text)
    txt_train_sel = selector.fit_transform(txt_train_raw, y_train_binary)
    txt_test_sel = selector.transform(txt_test_raw)
    selected_text_names = [f"text__{name}" for name in vocab[selector.get_support()]]
    print(f"Termos selecionados: {len(selected_text_names)} features", flush=True)

    # 3. Empilhar Matrizes do Modelo Hibrido
    X_train_hybrid = hstack([sr_train_sp, txt_train_sel])
    X_test_hybrid = hstack([sr_test_sp, txt_test_sel])
    print(
        f"Matriz Hibrida Final: Treino {X_train_hybrid.shape}, Teste {X_test_hybrid.shape}",
        flush=True,
    )

    test_results: list[dict[str, object]] = []

    # Modelos a avaliar no teste: (nome, X_tr, X_te)
    configs = [
        ("structure_only", st_train_sp, st_test_sp),
        ("structure_and_repo", sr_train_sp, sr_test_sp),
        ("hybrid_screened", X_train_hybrid, X_test_hybrid),
    ]

    predictions_dict: dict[str, np.ndarray] = {}

    print("\n" + "=" * 50, flush=True)
    print("TREINANDO E AVALIANDO MODELOS BINARIOS NO TESTE CEGO", flush=True)
    print("=" * 50, flush=True)

    for cfg_name, X_tr, X_te in configs:
        # A. Modelo Unweighted (Probabilidades Calibradas)
        lr_none = LogisticRegression(max_iter=1000, class_weight=None, random_state=random_seed)
        lr_none.fit(X_tr, y_train_binary)
        probs_none = lr_none.predict_proba(X_te)[:, 1]
        predictions_dict[f"{cfg_name}_unweighted"] = probs_none

        roc_n = float(roc_auc_score(y_test_binary, probs_none))
        pr_n = float(average_precision_score(y_test_binary, probs_none))
        brier_n = float(brier_score_loss(y_test_binary, probs_none))

        # Avaliacao em corte padrao 0.50
        pred_def = probs_none >= 0.50
        f1_def = float(f1_score(y_test_binary, pred_def, zero_division=0))
        prec_def = float(precision_score(y_test_binary, pred_def, zero_division=0))
        rec_def = float(recall_score(y_test_binary, pred_def, zero_division=0))
        bacc_def = float(balanced_accuracy_score(y_test_binary, pred_def))

        test_results.append(
            {
                "model": cfg_name,
                "weighting": "unweighted",
                "threshold_rule": "default_0.50",
                "threshold": 0.50,
                "roc_auc": roc_n,
                "pr_auc": pr_n,
                "f1": f1_def,
                "precision": prec_def,
                "recall": rec_def,
                "balanced_accuracy": bacc_def,
                "brier_score": brier_n,
            }
        )

        # Avaliacao no limiar congelado otimo t* = 0.21
        pred_opt = probs_none >= frozen_optimal_threshold
        f1_opt = float(f1_score(y_test_binary, pred_opt, zero_division=0))
        prec_opt = float(precision_score(y_test_binary, pred_opt, zero_division=0))
        rec_opt = float(recall_score(y_test_binary, pred_opt, zero_division=0))
        bacc_opt = float(balanced_accuracy_score(y_test_binary, pred_opt))

        test_results.append(
            {
                "model": cfg_name,
                "weighting": "unweighted",
                "threshold_rule": f"frozen_optimal_{frozen_optimal_threshold:.2f}",
                "threshold": frozen_optimal_threshold,
                "roc_auc": roc_n,
                "pr_auc": pr_n,
                "f1": f1_opt,
                "precision": prec_opt,
                "recall": rec_opt,
                "balanced_accuracy": bacc_opt,
                "brier_score": brier_n,
            }
        )

        # B. Modelo Balanced
        lr_bal = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_seed
        )
        lr_bal.fit(X_tr, y_train_binary)
        probs_bal = lr_bal.predict_proba(X_te)[:, 1]
        predictions_dict[f"{cfg_name}_balanced"] = probs_bal

        roc_b = float(roc_auc_score(y_test_binary, probs_bal))
        pr_b = float(average_precision_score(y_test_binary, probs_bal))
        brier_b = float(brier_score_loss(y_test_binary, probs_bal))

        pred_bal_def = probs_bal >= 0.50
        test_results.append(
            {
                "model": cfg_name,
                "weighting": "balanced",
                "threshold_rule": "default_0.50",
                "threshold": 0.50,
                "roc_auc": roc_b,
                "pr_auc": pr_b,
                "f1": float(f1_score(y_test_binary, pred_bal_def, zero_division=0)),
                "precision": float(precision_score(y_test_binary, pred_bal_def, zero_division=0)),
                "recall": float(recall_score(y_test_binary, pred_bal_def, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_binary, pred_bal_def)),
                "brier_score": brier_b,
            }
        )

        print(
            f"[{cfg_name}] ROC-AUC: {roc_n:.4f} | PR-AUC: {pr_n:.4f} | "
            f"F1 (t*={frozen_optimal_threshold:.2f}): {f1_opt:.4f} "
            f"(Prec: {prec_opt:.4f}, Rec: {rec_opt:.4f})",
            flush=True,
        )

    # 4. Avaliacao da Intensidade Condicional no Teste Cego
    print("\n" + "=" * 50, flush=True)
    print("AVALIANDO INTENSIDADE CONDICIONAL NO TESTE CEGO", flush=True)
    print("=" * 50, flush=True)

    pos_tr_mask = y_train_binary == 1
    pos_te_mask = y_test_binary == 1

    y_int_tr = train_df.iloc[np.where(pos_tr_mask)[0]]["reuse_intensity"].astype(int).to_numpy()
    y_int_te = test_df.iloc[np.where(pos_te_mask)[0]]["reuse_intensity"].astype(int).to_numpy()

    print(f"Amostras positivas: Treino={len(y_int_tr)}, Teste={len(y_int_te)}", flush=True)

    intensity_test_results: list[dict[str, object]] = []

    for cfg_name, X_tr, X_te in configs:
        X_int_tr = X_tr[pos_tr_mask]
        X_int_te = X_te[pos_te_mask]

        lr_int = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_seed
        )
        lr_int.fit(X_int_tr, y_int_tr)
        int_preds = lr_int.predict(X_int_te)
        int_m = _intensity_metrics(y_int_te, int_preds)

        intensity_test_results.append(
            {
                "model": cfg_name,
                "task": "conditional_intensity",
                "class_weight": "balanced",
                "test_positive_samples": len(y_int_te),
                **int_m,
            }
        )
        print(
            f"[{cfg_name} - Intensidade] Macro F1: {int_m['macro_f1']:.4f} | "
            f"BalAcc: {int_m['balanced_accuracy']:.4f} | Class 3 F1: {int_m['class_3_f1']:.4f}",
            flush=True,
        )

    # 5. Salvar Tabelas de Resultados
    test_binary_df = pd.DataFrame(test_results)
    test_binary_df.to_csv(out_dir / "test_binary_results.csv", index=False)

    test_int_df = pd.DataFrame(intensity_test_results)
    test_int_df.to_csv(out_dir / "test_intensity_results.csv", index=False)

    # Matriz de confusao do modelo campeao
    opt_hybrid_preds = (
        predictions_dict["hybrid_screened_unweighted"] >= frozen_optimal_threshold
    ).astype(int)
    cm = confusion_matrix(y_test_binary, opt_hybrid_preds)
    cm_dict = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }
    (out_dir / "test_confusion_matrix.json").write_text(
        json.dumps(cm_dict, indent=2) + "\n", encoding="utf-8"
    )

    # 6. Gerar Graficos Finais para o Paper
    _plot_final_test_figures(
        figures_dir, y_test_binary, predictions_dict, cm, frozen_optimal_threshold
    )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": (
            "Strict blind evaluation on untouched external test set of 40,000 instances "
            "(14,568 unseen repos)"
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_repos": len(set(train_df["representative_repo_id"])),
        "test_repos": len(set(test_df["representative_repo_id"])),
        "repo_overlap": 0,
        "frozen_decision_threshold": frozen_optimal_threshold,
        "champion_model": "hybrid_screened",
        "champion_test_roc_auc": float(
            test_binary_df.loc[
                (test_binary_df["model"] == "hybrid_screened")
                & (
                    test_binary_df["threshold_rule"]
                    == f"frozen_optimal_{frozen_optimal_threshold:.2f}"
                ),
                "roc_auc",
            ].iloc[0]
        ),
        "champion_test_pr_auc": float(
            test_binary_df.loc[
                (test_binary_df["model"] == "hybrid_screened")
                & (
                    test_binary_df["threshold_rule"]
                    == f"frozen_optimal_{frozen_optimal_threshold:.2f}"
                ),
                "pr_auc",
            ].iloc[0]
        ),
        "champion_test_f1": float(
            test_binary_df.loc[
                (test_binary_df["model"] == "hybrid_screened")
                & (
                    test_binary_df["threshold_rule"]
                    == f"frozen_optimal_{frozen_optimal_threshold:.2f}"
                ),
                "f1",
            ].iloc[0]
        ),
        "champion_test_intensity_macro_f1": float(
            test_int_df.loc[test_int_df["model"] == "hybrid_screened", "macro_f1"].iloc[0]
        ),
    }
    (out_dir / "test_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70, flush=True)
    print(f"AVALIACAO FINAL CONCLUIDA COM SUCESSO EM {time.monotonic() - t0:.2f}s!", flush=True)
    print(f"Artefatos salvos em {out_dir}", flush=True)
    print("=" * 70, flush=True)

    return manifest


def _plot_final_test_figures(
    figures_dir: Path,
    y_test: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    cm: np.ndarray,
    frozen_t: float,
) -> None:
    """Gera curvas ROC, Precision-Recall e Matriz de Confusao da avaliacao cega final."""
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Curvas ROC e PR Lado a Lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    models_to_plot = [
        ("structure_only_unweighted", "1. Structure Only", "#94a3b8"),
        ("structure_and_repo_unweighted", "2. Structure + Repo", "#f59e0b"),
        ("hybrid_screened_unweighted", "3. Hybrid (Campeão)", "#2563eb"),
    ]

    # Curva ROC
    for key, label, color in models_to_plot:
        probs = preds_dict[key]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax1.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=2.2)

    ax1.plot([0, 1], [0, 1], linestyle="--", color="#64748b", alpha=0.7)
    ax1.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Curva ROC no Conjunto de Teste Cego (40k Amostras)", fontsize=12, fontweight="bold"
    )
    ax1.legend(loc="lower right", frameon=True, fontsize=10)

    # Curva Precision-Recall
    for key, label, color in models_to_plot:
        probs = preds_dict[key]
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax2.plot(rec, prec, label=f"{label} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.2)

    no_skill = y_test.mean()
    ax2.plot(
        [0, 1],
        [no_skill, no_skill],
        linestyle="--",
        color="#dc2626",
        alpha=0.7,
        label=f"Baseline Nulo ({no_skill:.1%})",
    )
    ax2.set_xlabel("Revocação (Recall)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precisão", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Curva Precision-Recall no Conjunto de Teste Cego", fontsize=12, fontweight="bold"
    )
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "final_test_roc_pr_curves.png")
    plt.close()

    # 2. Matriz de Confusao no Threshold Congelado
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Singleton (=1)", "Reutilizada (>=2)"],
        yticklabels=["Singleton (=1)", "Reutilizada (>=2)"],
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Classe Predita", fontsize=11, fontweight="bold")
    ax.set_ylabel("Classe Real Observada", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Matriz de Confusão no Teste Cego ($t^* = {frozen_t:.2f}$)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "final_test_confusion_matrix.png")
    plt.close()


def main() -> None:
    run_final_test_evaluation()


if __name__ == "__main__":
    main()
