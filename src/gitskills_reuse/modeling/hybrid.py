"""Modelo hibrido: Estrutura + Contexto de Repositorio + Top-K N-grams de Texto (Chi-Square).

Avalia a combinacao multimodal tabular + selecao supervisionada de texto sob
StratifiedGroupKFold, demonstrando a recuperacao de sinal textual sem explosao de dimensionalidade.
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
import shap
from scipy.sparse import hstack, load_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.preprocess import (
    build_preprocessor,
    predictor_view,
)
from gitskills_reuse.modeling.train import (
    DEFAULT_FEATURES,
    DEFAULT_OUTPUT,
    DEFAULT_SAMPLE,
    DEFAULT_TARGETS,
    compile_ablation_summary,
    load_modeling_frame,
)


def run_hybrid_evaluation(
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
    shap_samples_per_fold: int = 500,
) -> dict[str, object]:
    """Executa a validacao cruzada agrupada de 3 dobras para o modelo hibrido."""
    out_dir = output_base / "ablation" / "hybrid_screened"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dataset de modelagem...", flush=True)
    t0 = time.monotonic()
    frame = load_modeling_frame(sample_path, features_path, targets_path, need_text=True)
    print(f"Dataset carregado ({len(frame)} linhas) em {time.monotonic() - t0:.2f}s", flush=True)

    y_binary = frame["reused"].astype(int).to_numpy()
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Identificar diretorios de cache preexistentes
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

    fold_records: list[dict[str, object]] = []
    importance_records: list[dict[str, object]] = []
    shap_records: list[dict[str, object]] = []

    print(
        f"\nAvaliacao hibrida (struct + repo + top-{top_k_text} text) em {n_splits} dobras...",
        flush=True,
    )

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y=strata, groups=groups), start=1
    ):
        fold_start = time.monotonic()
        print(f"\n--- Dobra {fold}/{n_splits} ---", flush=True)

        sr_train = load_npz(sr_cache_dir / f"fold_{fold}_train.npz")
        sr_valid = load_npz(sr_cache_dir / f"fold_{fold}_validation.npz")
        txt_train = load_npz(text_cache_dir / f"fold_{fold}_train.npz")
        txt_valid = load_npz(text_cache_dir / f"fold_{fold}_validation.npz")

        # Obter nomes das features estruturais/repo
        fold_train_frame = frame.iloc[train_idx]
        sr_predictor_train = predictor_view(
            fold_train_frame, use_text=False, use_structural=True, use_repository=True
        )
        sr_trans = build_preprocessor(
            sr_predictor_train, use_text=False, use_structural=True, use_repository=True
        )
        sr_trans.fit(sr_predictor_train)
        sr_feature_names = list(sr_trans.get_feature_names_out())

        # Obter vocabulario de texto
        print(
            f"[Dobra {fold}] Ajustando vetorizador de texto para obter nomes dos n-grams...",
            flush=True,
        )
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        vec.fit(fold_train_frame["content"])
        raw_vocab = np.array(vec.get_feature_names_out())

        # Selecao Chi-2 estritamente no treino
        print(
            f"[Dobra {fold}] Selecionando as Top-{top_k_text} features de texto via Chi-2...",
            flush=True,
        )
        selector = SelectKBest(score_func=chi2, k=top_k_text)
        txt_train_sel = selector.fit_transform(txt_train, y_binary[train_idx])
        txt_valid_sel = selector.transform(txt_valid)

        selected_text_mask = selector.get_support()
        selected_text_names = [f"text__{term}" for term in raw_vocab[selected_text_mask]]

        combined_feature_names = sr_feature_names + selected_text_names
        X_train_sp = hstack([sr_train, txt_train_sel])
        X_valid_sp = hstack([sr_valid, txt_valid_sel])

        # 1. Regressao Logistica (Balanced e None)
        for cw in ["balanced", "none"]:
            cw_param = "balanced" if cw == "balanced" else None
            lr = LogisticRegression(max_iter=1000, class_weight=cw_param, random_state=random_seed)
            lr.fit(X_train_sp, y_binary[train_idx])
            probs = lr.predict_proba(X_valid_sp)[:, 1]
            preds = probs >= 0.5

            roc = float(roc_auc_score(y_binary[valid_idx], probs))
            pr = float(average_precision_score(y_binary[valid_idx], probs))
            bacc = float(balanced_accuracy_score(y_binary[valid_idx], preds))
            prec = float(precision_score(y_binary[valid_idx], preds, zero_division=0))
            rec = float(recall_score(y_binary[valid_idx], preds, zero_division=0))
            f1 = float(f1_score(y_binary[valid_idx], preds, zero_division=0))

            fold_records.append(
                {
                    "ablation": "hybrid_screened",
                    "task": "binary_reuse",
                    "fold": fold,
                    "class_weight": cw,
                    "train_rows": len(train_idx),
                    "validation_rows": len(valid_idx),
                    "train_positive_rate": float(y_binary[train_idx].mean()),
                    "validation_positive_rate": float(y_binary[valid_idx].mean()),
                    "iterations": int(np.max(lr.n_iter_)),
                    "converged": bool(lr.n_iter_[0] < 1000),
                    "average_precision": pr,
                    "roc_auc": roc,
                    "balanced_accuracy_at_0_5": bacc,
                    "precision_at_0_5": prec,
                    "recall_at_0_5": rec,
                    "f1_at_0_5": f1,
                }
            )

            # Coeficientes
            for fname, coef in zip(combined_feature_names, lr.coef_[0], strict=False):
                importance_records.append(
                    {
                        "ablation": "hybrid_screened",
                        "task": "binary_reuse",
                        "fold": fold,
                        "class_weight": cw,
                        "feature": fname,
                        "coefficient": float(coef),
                    }
                )

            if cw == "balanced":
                print(
                    f"[Dobra {fold}] ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1: {f1:.4f}",
                    flush=True,
                )

        # 2. HistGradientBoosting + SHAP
        print(f"[Dobra {fold}] Treinando HistGradientBoostingClassifier...", flush=True)
        X_train_dense = X_train_sp.toarray().astype(np.float32)
        X_valid_dense = X_valid_sp.toarray().astype(np.float32)

        hgb = HistGradientBoostingClassifier(
            max_iter=50,
            learning_rate=0.05,
            max_depth=3,
            class_weight="balanced",
            random_state=random_seed,
        )
        hgb.fit(X_train_dense, y_binary[train_idx])
        hgb_probs = hgb.predict_proba(X_valid_dense)[:, 1]
        hgb_roc = float(roc_auc_score(y_binary[valid_idx], hgb_probs))
        hgb_pr = float(average_precision_score(y_binary[valid_idx], hgb_probs))
        print(
            f"[Dobra {fold} - HGB (Grouped)] ROC-AUC: {hgb_roc:.4f} | PR-AUC: {hgb_pr:.4f}",
            flush=True,
        )

        # Calcular SHAP em subamostra de validacao
        sample_size = min(shap_samples_per_fold, len(valid_idx))
        print(
            f"[Dobra {fold}] Computando TreeExplainer SHAP em {sample_size} amostras...", flush=True
        )
        explainer = shap.TreeExplainer(hgb)
        shap_vals = explainer(X_valid_dense[:sample_size])
        mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)

        for fname, val in zip(combined_feature_names, mean_abs_shap, strict=False):
            shap_records.append(
                {
                    "fold": fold,
                    "feature": fname,
                    "mean_abs_shap": float(val),
                }
            )

        print(f"Dobra {fold} concluida em {time.monotonic() - fold_start:.2f}s", flush=True)

    # Consolidar dobras
    folds_df = pd.DataFrame(fold_records)
    folds_df.to_csv(out_dir / "cross_validation_folds.csv", index=False)

    numeric_cols = [
        c
        for c in folds_df.select_dtypes(include="number").columns
        if c not in {"fold", "train_rows", "validation_rows"}
    ]
    summary = folds_df.groupby(["ablation", "task", "class_weight"], as_index=False)[
        numeric_cols
    ].agg(["mean", "std"])
    summary.columns = [
        "_".join(str(p) for p in col if p).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(out_dir / "cross_validation_summary.csv", index=False)

    # Consolidar Feature Importances (Regressao Logistica)
    imp_df = pd.DataFrame(importance_records)
    imp_summary = imp_df.groupby(["ablation", "task", "class_weight", "feature"], as_index=False)[
        "coefficient"
    ].agg(mean="mean", std="std")
    imp_summary["odds_ratio"] = np.exp(imp_summary["mean"])
    imp_summary.sort_values(
        by=["task", "class_weight", "mean"], ascending=[True, True, False], inplace=True
    )
    imp_summary.to_csv(out_dir / "feature_importances.csv", index=False)

    # Consolidar SHAP Summary
    shap_df = pd.DataFrame(shap_records)
    shap_summary = (
        shap_df.groupby("feature", as_index=False)["mean_abs_shap"]
        .agg(mean="mean", std="std")
        .sort_values(by="mean", ascending=False)
    )
    shap_summary.to_csv(out_dir / "shap_summary.csv", index=False)

    # Salvar Manifest
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "StratifiedGroupKFold on external training split; test split untouched",
        "ablation": "hybrid_screened",
        "top_k_text": top_k_text,
        "n_splits": n_splits,
        "random_seed": random_seed,
        "total_features": len(combined_feature_names),
        "structure_repo_features": len(sr_feature_names),
        "text_screened_features": len(selected_text_names),
        "linear_mean_roc_auc": float(
            summary[summary["class_weight"] == "balanced"]["roc_auc_mean"].iloc[0]
        ),
        "linear_mean_pr_auc": float(
            summary[summary["class_weight"] == "balanced"]["average_precision_mean"].iloc[0]
        ),
    }
    (out_dir / "cross_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Atualizar resumo global de ablacoes
    print("\nAtualizando tabela global reports/modeling/ablation_summary.csv...", flush=True)
    compile_ablation_summary(output_base)

    # Gerar figuras
    print("Gerando graficos comparativos e SHAP em reports/modeling/figures/...", flush=True)
    _generate_paper_plots(output_base, shap_summary, imp_summary)

    print(f"\nAvaliacao hibrida concluida com sucesso! Artefatos salvos em {out_dir}", flush=True)
    return manifest


def _generate_paper_plots(
    output_base: Path,
    shap_summary: pd.DataFrame,
    imp_summary: pd.DataFrame,
) -> None:
    """Gera graficos academicos de alta resolucao para inclusao no artigo/relatorio."""
    figures_dir = output_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Grafico Comparativo de Todas as Configuracoes de Ablacao
    ablation_summary_path = output_base / "ablation_summary.csv"
    if ablation_summary_path.is_file():
        df_all = pd.read_csv(ablation_summary_path)
        bin_df = df_all[
            (df_all["task"] == "binary_reuse") & (df_all["class_weight"] == "balanced")
        ].copy()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        names_map = {
            "text_only": "1. Text Only (TF-IDF 50k)",
            "repo_only": "2. Repo Only (~97)",
            "all": "3. Multimodal All (50k)",
            "structure_only": "4. Structure Only (46)",
            "structure_and_repo": "5. Structure + Repo (143)",
            "hybrid_screened": "6. Hybrid (Struct+Repo+Top100 Text)",
        }
        bin_df["display_name"] = bin_df["ablation"].map(names_map).fillna(bin_df["ablation"])
        bin_df.sort_values(by="roc_auc_mean", inplace=True)

        y_pos = np.arange(len(bin_df))
        colors = [
            "#94a3b8" if "Hybrid" not in name else "#2563eb" for name in bin_df["display_name"]
        ]

        ax.barh(
            y_pos,
            bin_df["roc_auc_mean"],
            xerr=bin_df["roc_auc_std"],
            align="center",
            color=colors,
            alpha=0.9,
            capsize=5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bin_df["display_name"], fontsize=11, fontweight="medium")
        ax.set_xlabel("ROC-AUC Médio (3-Fold Grouped CV)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Estudo de Ablação de Features para Reutilização de Skills (binary_reuse)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlim(0.50, 0.75)

        auc_means = bin_df["roc_auc_mean"]
        auc_stds = bin_df["roc_auc_std"]
        for i, (v, s) in enumerate(zip(auc_means, auc_stds, strict=False)):
            ax.text(
                v + s + 0.005,
                i,
                f"{v:.3f} ± {s:.3f}",
                va="center",
                fontsize=10,
                fontweight="bold" if "Hybrid" in bin_df["display_name"].iloc[i] else "normal",
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "ablation_roc_auc_comparison.png")
        plt.close()

    # 2. Grafico das Top-20 Features de SHAP do Modelo Hibrido
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    top_shap = shap_summary.head(20).copy()
    top_shap.sort_values(by="mean", ascending=True, inplace=True)

    colors = []
    for f in top_shap["feature"]:
        if f.startswith("text__"):
            colors.append("#10b981")  # Verde para texto
        elif f.startswith("repo_"):
            colors.append("#f59e0b")  # Amarelo/Laranja para repo
        else:
            colors.append("#3b82f6")  # Azul para estrutura

    y_pos = np.arange(len(top_shap))
    ax.barh(y_pos, top_shap["mean"], color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_shap["feature"], fontsize=10)
    ax.set_xlabel("Impacto Médio |SHAP| na Predição", fontsize=11, fontweight="bold")
    ax.set_title(
        "Top-20 Features por Importância SHAP no Modelo Híbrido",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Legenda manual
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3b82f6", label="Estrutura do Arquivo"),
        Patch(facecolor="#f59e0b", label="Contexto do Repositório"),
        Patch(facecolor="#10b981", label="Top N-grams de Texto (Chi-2)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(figures_dir / "hybrid_shap_top20.png")
    plt.close()


def main() -> None:
    run_hybrid_evaluation()


if __name__ == "__main__":
    main()
