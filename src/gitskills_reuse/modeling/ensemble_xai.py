"""Decomposição de Explicabilidade (XAI) do Ensemble Heterogêneo.

Implementa a triangulação de explicabilidade conforme os axiomas de Lloyd Shapley e Lundberg & Lee:
1. Modelo Linear (50% do Blend): Odds Ratios exatos (e^beta) e Linear SHAP analítico.
2. Modelo Baseado em Árvores GBDT (25% do Blend): Tree SHAP exato via TreeExplainer em tempo polinomial.
3. Rede Neural MLP (25% do Blend): Sampling/Gradient SHAP sobre espaço contínuo.
4. Ensemble Heterogêneo (Blend): Unified Ensemble SHAP via Axioma da Aditividade de Shapley:
   phi_i(Blend) = 0.50 * phi_i(Linear) + 0.25 * phi_i(Tree) + 0.25 * phi_i(MLP).

Exporta métricas de fidelidade, tabelas comparativas de ranking e figura diagnóstica.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
import shap

DEFAULT_CACHE_DIR = Path("data/processed/cache_sparse")
DEFAULT_OUTPUT_DIR = Path("reports/modeling")
DEFAULT_FIGURE_DIR = Path("reports/modeling/figures")


def run_ensemble_xai(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    eval_samples: int = 150,
    mlp_samples: int = 60,
    bg_samples: int = 30,
    random_seed: int = 20261009,
) -> dict[str, object]:
    t_total = time.monotonic()
    print("=" * 70, flush=True)
    print("INICIANDO DECOMPOSIÇÃO DE EXPLICABILIDADE (XAI) DO ENSEMBLE", flush=True)
    print("=" * 70, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar cache
    print("\n[1/5] Carregando matrizes pré-processadas do cache...", flush=True)
    t0 = time.monotonic()
    X_tr_sp = load_npz(cache_dir / "X_train_hybrid.npz")
    y_tr = np.load(cache_dir / "y_train_binary.npy")
    X_te_sp = load_npz(cache_dir / "X_test_hybrid.npz")
    y_te = np.load(cache_dir / "y_test_binary.npy")

    with open(cache_dir / "hybrid_feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    n_features = len(feature_names)
    print(f"Treino: {X_tr_sp.shape} | Teste: {X_te_sp.shape} | Features: {n_features} ({time.monotonic() - t0:.2f}s)", flush=True)

    # Matrizes densas float32 para GBDT, MLP e SHAP
    print("[2/5] Convertendo subconjuntos para representação densa float32...", flush=True)
    X_tr_dense = X_tr_sp.toarray().astype(np.float32)
    X_te_dense = X_te_sp.toarray().astype(np.float32)

    # 2. Ajuste dos 3 Modelos
    print("\n[3/5] Ajustando os modelos do Ensemble no conjunto de treino...", flush=True)
    
    # 2.1 Regressão Logística L2
    t_m = time.monotonic()
    lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500, random_state=random_seed)
    lr.fit(X_tr_sp, y_tr)
    lr_time = time.monotonic() - t_m
    probs_lr = lr.predict_proba(X_te_sp)[:, 1]
    roc_lr = roc_auc_score(y_te, probs_lr)
    pr_lr = average_precision_score(y_te, probs_lr)
    f1_lr = f1_score(y_te, probs_lr >= 0.21, zero_division=0)
    print(f"  [1. Logistic L2] ROC-AUC: {roc_lr:.4f} | PR-AUC: {pr_lr:.4f} | F1(0.21): {f1_lr:.4f} ({lr_time:.2f}s)", flush=True)

    # 2.2 HistGradientBoosting
    t_m = time.monotonic()
    hgb = HistGradientBoostingClassifier(
        max_depth=3, min_samples_leaf=200, learning_rate=0.05, max_iter=60, random_state=random_seed
    )
    hgb.fit(X_tr_dense, y_tr)
    hgb_time = time.monotonic() - t_m
    probs_hgb = hgb.predict_proba(X_te_dense)[:, 1]
    roc_hgb = roc_auc_score(y_te, probs_hgb)
    pr_hgb = average_precision_score(y_te, probs_hgb)
    f1_hgb = f1_score(y_te, probs_hgb >= 0.21, zero_division=0)
    print(f"  [2. GBDT Árvores] ROC-AUC: {roc_hgb:.4f} | PR-AUC: {pr_hgb:.4f} | F1(0.21): {f1_hgb:.4f} ({hgb_time:.2f}s)", flush=True)

    # 2.3 MLP Neural Network
    t_m = time.monotonic()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_seed,
    )
    mlp.fit(X_tr_dense, y_tr)
    mlp_time = time.monotonic() - t_m
    probs_mlp = mlp.predict_proba(X_te_dense)[:, 1]
    roc_mlp = roc_auc_score(y_te, probs_mlp)
    pr_mlp = average_precision_score(y_te, probs_mlp)
    f1_mlp = f1_score(y_te, probs_mlp >= 0.21, zero_division=0)
    print(f"  [3. MLP Neural]   ROC-AUC: {roc_mlp:.4f} | PR-AUC: {pr_mlp:.4f} | F1(0.21): {f1_mlp:.4f} ({mlp_time:.2f}s)", flush=True)

    # 2.4 Heterogeneous Blend
    probs_blend = 0.50 * probs_lr + 0.25 * probs_hgb + 0.25 * probs_mlp
    roc_blend = roc_auc_score(y_te, probs_blend)
    pr_blend = average_precision_score(y_te, probs_blend)
    f1_blend = f1_score(y_te, probs_blend >= 0.21, zero_division=0)
    print(f"  [4. Blend 50/25/25] ROC-AUC: {roc_blend:.4f} | PR-AUC: {pr_blend:.4f} | F1(0.21): {f1_blend:.4f}", flush=True)

    # 3. Computação da Explicabilidade Estratificada
    print("\n[4/5] Computando atribuições XAI para cada componente e unificando via Aditividade...", flush=True)

    # Selecionar subamostra de teste para avaliação explicativa
    rng = np.random.RandomState(random_seed)
    eval_indices = rng.choice(len(y_te), size=min(eval_samples, len(y_te)), replace=False)
    X_eval = X_te_dense[eval_indices]
    
    # 3.1 Linear SHAP analítico exato e Odds Ratios
    print("  -> Computando Linear SHAP exato e Odds Ratios...", flush=True)
    mean_X_tr = X_tr_dense.mean(axis=0)
    std_X_tr = X_tr_dense.std(axis=0) + 1e-9
    lr_weights = lr.coef_[0]
    
    # Odds Ratios brutos e padronizados
    odds_ratios_raw = np.exp(lr_weights)
    odds_ratios_std = np.exp(lr_weights * std_X_tr)
    
    # phi_i(Linear) = w_i * (x_i - E[x_i])
    shap_linear_eval = (X_eval - mean_X_tr) * lr_weights
    mean_abs_shap_linear = np.abs(shap_linear_eval).mean(axis=0)

    # 3.2 Tree SHAP exato via TreeExplainer
    print("  -> Computando Tree SHAP exato via TreeExplainer (GBDT)...", flush=True)
    tree_explainer = shap.TreeExplainer(hgb)
    shap_tree_obj = tree_explainer(X_eval)
    shap_tree_eval = shap_tree_obj.values
    mean_abs_shap_tree = np.abs(shap_tree_eval).mean(axis=0)

    # 3.3 MLP SHAP via SamplingExplainer
    print(f"  -> Computando Neural SHAP via SamplingExplainer (MLP em {mlp_samples} amostras)...", flush=True)
    bg_indices = rng.choice(len(y_tr), size=bg_samples, replace=False)
    bg_dense = X_tr_dense[bg_indices]
    mlp_eval_subset = X_eval[:mlp_samples]
    
    sampling_explainer = shap.SamplingExplainer(mlp.predict_proba, bg_dense)
    shap_mlp_raw = sampling_explainer.shap_values(mlp_eval_subset)
    if isinstance(shap_mlp_raw, list):
        shap_mlp_eval = shap_mlp_raw[1]
    elif shap_mlp_raw.ndim == 3:
        shap_mlp_eval = shap_mlp_raw[:, :, 1]
    else:
        shap_mlp_eval = shap_mlp_raw
    mean_abs_shap_mlp = np.abs(shap_mlp_eval).mean(axis=0)

    # 3.4 Unified Ensemble SHAP via Axioma da Aditividade de Shapley
    # phi_i(Blend) = 0.50 * phi_i(Linear) + 0.25 * phi_i(Tree) + 0.25 * phi_i(MLP)
    print("  -> Unificando valores SHAP sob Axioma da Aditividade de Shapley...", flush=True)
    # Alinhando no subconjunto comum de instâncias
    shap_linear_aligned = shap_linear_eval[:mlp_samples]
    shap_tree_aligned = shap_tree_eval[:mlp_samples]
    shap_mlp_aligned = shap_mlp_eval
    
    shap_blend_aligned = (
        0.50 * shap_linear_aligned +
        0.25 * shap_tree_aligned +
        0.25 * shap_mlp_aligned
    )
    mean_abs_shap_blend = np.abs(shap_blend_aligned).mean(axis=0)

    # 4. Consolidar Tabela Diagnóstica
    print("\n[5/5] Consolidando tabelas, correlações e gerando figura comparativa...", flush=True)
    df_xai = pd.DataFrame({
        "feature": feature_names,
        "lr_coef": lr_weights,
        "odds_ratio_raw": odds_ratios_raw,
        "odds_ratio_std": odds_ratios_std,
        "shap_linear_mean_abs": mean_abs_shap_linear,
        "shap_tree_mean_abs": mean_abs_shap_tree,
        "shap_mlp_mean_abs": mean_abs_shap_mlp,
        "shap_blend_mean_abs": mean_abs_shap_blend,
    })

    # Ranks (1 = mais importante)
    df_xai["rank_linear"] = df_xai["shap_linear_mean_abs"].rank(ascending=False).astype(int)
    df_xai["rank_tree"] = df_xai["shap_tree_mean_abs"].rank(ascending=False).astype(int)
    df_xai["rank_mlp"] = df_xai["shap_mlp_mean_abs"].rank(ascending=False).astype(int)
    df_xai["rank_blend"] = df_xai["shap_blend_mean_abs"].rank(ascending=False).astype(int)

    # Ordenar por importância no Blend
    df_xai = df_xai.sort_values(by="shap_blend_mean_abs", ascending=False).reset_index(drop=True)
    csv_path = output_dir / "ensemble_xai_decomposition.csv"
    df_xai.to_csv(csv_path, index=False)
    print(f"Tabela de explicabilidade salva em: {csv_path}", flush=True)

    # Calcular correlações de Spearman (Efeito Rashomon)
    corr_matrix = df_xai[
        ["shap_linear_mean_abs", "shap_tree_mean_abs", "shap_mlp_mean_abs", "shap_blend_mean_abs"]
    ].corr(method="spearman")
    print("\nMatriz de Correlação de Spearman entre os Modelos XAI (Quantificação do Efeito Rashomon):")
    print(corr_matrix.round(4))

    # 5. Gerar Gráfico Comparativo de Alta Resolução
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=False)

    top20_blend = df_xai.head(20).copy().iloc[::-1]

    # Cores temáticas
    palette = {
        "Linear": "#1f77b4",
        "GBDT": "#2ca02c",
        "MLP": "#ff7f0e",
        "Blend": "#9467bd",
    }

    # Painel 1: Odds Ratios (Regressão Logística - Visão SE Linter)
    top20_lr = df_xai.sort_values(by="shap_linear_mean_abs", ascending=False).head(20).iloc[::-1]
    axes[0].barh(top20_lr["feature"], top20_lr["shap_linear_mean_abs"], color=palette["Linear"], alpha=0.85)
    axes[0].set_title("(a) Linear SHAP (50% do Blend)\nÂncora Analítica (Linter / Odds Ratios)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mean |Linear SHAP|", fontsize=11)

    # Painel 2: Tree SHAP (GBDT - Visão Não-Linear / Divisões Ótimas)
    top20_tree = df_xai.sort_values(by="shap_tree_mean_abs", ascending=False).head(20).iloc[::-1]
    axes[1].barh(top20_tree["feature"], top20_tree["shap_tree_mean_abs"], color=palette["GBDT"], alpha=0.85)
    axes[1].set_title("(b) Tree SHAP (25% do Blend)\nÁrvores Polinomiais Exatas (GBDT)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Mean |Tree SHAP|", fontsize=11)

    # Painel 3: Neural SHAP (MLP - Representações Latentes)
    top20_mlp = df_xai.sort_values(by="shap_mlp_mean_abs", ascending=False).head(20).iloc[::-1]
    axes[2].barh(top20_mlp["feature"], top20_mlp["shap_mlp_mean_abs"], color=palette["MLP"], alpha=0.85)
    axes[2].set_title("(c) Neural SHAP (25% do Blend)\nAtivação Não-Linear Latente (MLP)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Mean |Neural SHAP|", fontsize=11)

    # Painel 4: Unified Ensemble SHAP (Blend 50/25/25)
    axes[3].barh(top20_blend["feature"], top20_blend["shap_blend_mean_abs"], color=palette["Blend"], alpha=0.85)
    axes[3].set_title("(d) Unified Ensemble SHAP\nAxioma da Aditividade de Shapley", fontsize=13, fontweight="bold")
    axes[3].set_xlabel("Mean |Blend SHAP|", fontsize=11)

    plt.tight_layout()
    fig_path = figure_dir / "ensemble_xai_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura comparativa de alta resolução salva em: {fig_path}", flush=True)

    # Manifesto de execução
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_instances": len(y_te),
        "eval_instances_evaluated": len(eval_indices),
        "mlp_instances_evaluated": mlp_samples,
        "background_samples": bg_samples,
        "n_features": n_features,
        "performance": {
            "logistic_l2": {"roc_auc": roc_lr, "pr_auc": pr_lr, "f1_0_21": f1_lr, "time_s": lr_time},
            "constrained_gbdt": {"roc_auc": roc_hgb, "pr_auc": pr_hgb, "f1_0_21": f1_hgb, "time_s": hgb_time},
            "mlp_neural_net": {"roc_auc": roc_mlp, "pr_auc": pr_mlp, "f1_0_21": f1_mlp, "time_s": mlp_time},
            "heterogeneous_blend": {"roc_auc": roc_blend, "pr_auc": pr_blend, "f1_0_21": f1_blend},
        },
        "spearman_correlations": corr_matrix.to_dict(),
        "total_runtime_seconds": round(time.monotonic() - t_total, 2),
    }

    manifest_path = output_dir / "ensemble_xai_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifesto salvo em: {manifest_path}", flush=True)
    print(f"\nDECOMPOSIÇÃO DE EXPLICABILIDADE CONCLUÍDA EM {time.monotonic() - t_total:.2f}s!", flush=True)
    print("=" * 70, flush=True)

    return manifest


if __name__ == "__main__":
    run_ensemble_xai()
