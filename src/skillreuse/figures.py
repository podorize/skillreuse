"""Figuras do relatorio a partir dos artefatos em ``results/owner`` (nenhum modelo e treinado)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

RESULTS = Path("results/owner")
OUT = RESULTS / "figures"
PRIMARY = RESULTS / "primary"
LABEL = {
    "skill": "skill", "repo": "repositório", "repo+catalog": "repositório+catálogo",
    "path+catalog": "caminho+catálogo", "clean_full": "limpo completo",
    "clean_full+shortcuts": "limpo + atalhos", "shortcuts_only": "só atalhos",
}
COLORS = {"limpo": "#1f77b4", "atalho": "#d62728", "base": "#7f7f7f"}


def _view_color(view: str) -> str:
    return COLORS["atalho"] if "shortcuts" in view else COLORS["limpo"]


def fig_views() -> None:
    clf = pd.read_csv(PRIMARY / "classification_summary.csv")
    reg = pd.read_csv(PRIMARY / "regression_summary.csv")
    clf = clf[(clf.subset == "all") & clf.config.str.endswith("|random_forest")]
    reg = reg[(reg.subset == "all") & reg.config.str.endswith("|random_forest_log")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for a, df, mean, std, base, title, xlabel in (
        (ax[0], clf, "roc_auc_mean", "roc_auc_std", 0.5, "Reúso entre donos (cross_owner)",
         "AUC-ROC (RF; média ± dp de 9 dobras)"),
        (ax[1], reg, "rmsle_mean", "rmsle_std", float(
            pd.read_csv(PRIMARY / "regression_summary.csv").query(
                "subset=='all' and config=='dummy_mean_log'")["rmsle_mean"].iloc[0]),
         "Regressão: log1p(owner_count)", "RMSLE (RF; menor é melhor)"),
    ):
        df = df.assign(view=df.config.str.split("|").str[0]).sort_values(
            mean, ascending=(title.startswith("Reúso")))
        a.barh([LABEL[v] for v in df.view], df[mean], xerr=df[std],
               color=[_view_color(v) for v in df.view], alpha=0.85)
        a.axvline(base, color=COLORS["base"], ls="--", lw=1.2, label="baseline")
        lo, hi = (0.45, 0.76) if title.startswith("Reúso") else (0.6, 0.74)
        a.set_xlim(lo, hi)
        a.set_xlabel(xlabel)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8, loc="lower right" if title.startswith("Reúso") else "lower left")
    fig.tight_layout()
    fig.savefig(OUT / "views_primary.png", dpi=160)
    plt.close(fig)


SENSITIVITY = [
    ("sem limite", RESULTS / "models"),
    ("excl. >1000", RESULTS / "robustness" / "excl1000"),
    ("excl. >500", RESULTS / "models_excl_big_owners"),
    ("excl. >200", RESULTS / "robustness" / "excl200"),
    ("50 por dono", PRIMARY),
]


def sensitivity_table() -> pd.DataFrame:
    rows = []
    for label, directory in SENSITIVITY:
        clf = pd.read_csv(directory / "classification_summary.csv")
        reg = pd.read_csv(directory / "regression_summary.csv")
        clf, reg = clf[clf.subset == "all"].set_index("config"), reg[reg.subset == "all"].set_index(
            "config")
        base = reg.loc["dummy_mean_log", "rmsle_mean"]
        rows.append({
            "setting": label,
            "auc_clean_rf": clf.loc["clean_full|random_forest", "roc_auc_mean"],
            "auc_std": clf.loc["clean_full|random_forest", "roc_auc_std"],
            "rmsle_baseline": base,
            "rmsle_gain_clean_rf": base - reg.loc["clean_full|random_forest_log", "rmsle_mean"],
            "rmsle_gain_shortcuts_only_rf": base
            - reg.loc["shortcuts_only|random_forest_log", "rmsle_mean"],
        })
    return pd.DataFrame(rows)


def fig_sensitivity() -> None:
    table = sensitivity_table()
    table.to_csv(OUT / "sensitivity_owner_limit.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(table["setting"], table["auc_clean_rf"], yerr=table["auc_std"], color="#1f77b4")
    ax[0].axhline(0.5, color=COLORS["base"], ls="--")
    ax[0].set_ylim(0.45, 0.95)
    ax[0].set_ylabel("AUC-ROC (RF limpo)")
    ax[1].bar(table["setting"], table["rmsle_gain_clean_rf"], color="#2ca02c")
    ax[1].set_ylabel("redução do RMSLE vs. baseline")
    for a in ax:
        a.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_owner_limit.png", dpi=160)
    plt.close(fig)


def fig_importance() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for a, task, title, xlabel in (
        (ax[0], "classification", "Reúso entre donos", "queda do AUC ao permutar"),
        (ax[1], "regression", "log1p(owner_count)", "aumento do RMSLE ao permutar"),
    ):
        df = pd.read_csv(RESULTS / "importance" / f"{task}_importance.csv")
        grp = df[df.feature.str.startswith("grupo:")]
        col = df[~df.feature.str.startswith("grupo:")].head(6)
        sel = pd.concat([grp, col]).sort_values("mean")
        a.barh(sel.feature.str.replace("grupo:", "[grupo] "), sel["mean"], xerr=sel["std"],
               color=["#9467bd" if f.startswith("grupo:") else "#1f77b4" for f in sel.feature])
        a.set_xlabel(xlabel)
        a.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=160)
    plt.close(fig)


def fig_holdout() -> None:
    pred = pd.read_csv(RESULTS / "holdout" / "classification_predictions.csv")
    fpr, tpr, _ = roc_curve(pred["y"], pred["prob"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.7))
    ax[0].plot(fpr, tpr, color="#1f77b4", label="RF (holdout)")
    ax[0].plot([0, 1], [0, 1], "--", color=COLORS["base"], label="acaso")
    ax[0].set_xlabel("taxa de falsos positivos")
    ax[0].set_ylabel("taxa de verdadeiros positivos")
    ax[0].legend(fontsize=8)
    bins = np.linspace(pred["prob"].min(), pred["prob"].max(), 11)
    idx = np.clip(np.digitize(pred["prob"], bins) - 1, 0, 9)
    cal = pred.assign(bin=idx).groupby("bin").agg(p=("prob", "mean"), y=("y", "mean"),
                                                   n=("y", "size"))
    ax[1].plot(cal["p"], cal["y"], "o-", color="#1f77b4")
    ax[1].plot([0.3, 1], [0.3, 1], "--", color=COLORS["base"])
    ax[1].set_xlabel("probabilidade prevista")
    ax[1].set_ylabel("frequência observada")
    fig.tight_layout()
    fig.savefig(OUT / "holdout_roc_calibration.png", dpi=160)
    plt.close(fig)


def figures_main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_views()
    fig_sensitivity()
    fig_importance()
    if (RESULTS / "holdout" / "classification_predictions.csv").exists():
        fig_holdout()
    print(f"Figuras em {OUT}")
