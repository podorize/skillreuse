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
    "skill": "skill", "repo": "repository", "repo+catalog": "repository+catalog",
    "path+catalog": "path+catalog", "clean_full": "clean (full)",
    "clean_full+shortcuts": "clean + shortcuts", "shortcuts_only": "shortcuts only",
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
        (ax[0], clf, "roc_auc_mean", "roc_auc_std", 0.5, "Cross-owner reuse (cross_owner)",
         "ROC AUC (RF; mean ± sd over 9 folds)"),
        (ax[1], reg, "rmsle_mean", "rmsle_std", float(
            pd.read_csv(PRIMARY / "regression_summary.csv").query(
                "subset=='all' and config=='dummy_mean_log'")["rmsle_mean"].iloc[0]),
         "Regression: log1p(owner_count)", "RMSLE (RF; lower is better)"),
    ):
        df = df.assign(view=df.config.str.split("|").str[0]).sort_values(
            mean, ascending=(title.startswith("Cross")))
        a.barh([LABEL[v] for v in df.view], df[mean], xerr=df[std],
               color=[_view_color(v) for v in df.view], alpha=0.85)
        a.axvline(base, color=COLORS["base"], ls="--", lw=1.2, label="baseline")
        lo, hi = (0.45, 0.76) if title.startswith("Cross") else (0.6, 0.74)
        a.set_xlim(lo, hi)
        a.set_xlabel(xlabel)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8, loc="lower right" if title.startswith("Cross") else "lower left")
    fig.tight_layout()
    fig.savefig(OUT / "views_primary.png", dpi=160)
    plt.close(fig)


SENSITIVITY = [
    ("no limit", RESULTS / "models"),
    ("excl. >1000", RESULTS / "robustness" / "excl1000"),
    ("excl. >500", RESULTS / "models_excl_big_owners"),
    ("excl. >200", RESULTS / "robustness" / "excl200"),
    ("50 per owner", PRIMARY),
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
    ax[0].set_ylabel("ROC AUC (clean RF)")
    ax[1].bar(table["setting"], table["rmsle_gain_clean_rf"], color="#2ca02c")
    ax[1].set_ylabel("RMSLE reduction vs. baseline")
    for a in ax:
        a.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_owner_limit.png", dpi=160)
    plt.close(fig)


def fig_importance() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for a, task, title, xlabel in (
        (ax[0], "classification", "Cross-owner reuse", "AUC drop when permuted"),
        (ax[1], "regression", "log1p(owner_count)", "RMSLE increase when permuted"),
    ):
        df = pd.read_csv(RESULTS / "importance" / f"{task}_importance.csv")
        grp = df[df.feature.str.startswith("grupo:")]
        col = df[~df.feature.str.startswith("grupo:")].head(6)
        sel = pd.concat([grp, col]).sort_values("mean")
        a.barh(sel.feature.str.replace("grupo:", "[group] "), sel["mean"], xerr=sel["std"],
               color=["#9467bd" if f.startswith("grupo:") else "#1f77b4" for f in sel.feature])
        a.set_xlabel(xlabel)
        a.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=160)
    plt.close(fig)


INTERP = RESULTS / "interpretation"


def fig_pdp() -> None:
    d = pd.read_csv(INTERP / "pdp_oof.csv", keep_default_na=False)
    spec = [("classification", ["log_stars", "log_forks", "push_recency_days"],
             "P(cross-owner reuse)"),
            ("regression", ["log_skills_in_repo", "path_depth", "repo_license"],
             "mean log1p(owner_count)")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for row, (task, feats, ylabel) in enumerate(spec):
        obs = d[(d.task == task) & (d.feature == "(observed)")]["mean_prediction"].mean()
        for col, feat in enumerate(feats):
            ax = axes[row, col]
            g = d[(d.task == task) & (d.feature == feat)].copy()
            if feat == "repo_license":
                a = g.groupby("value", sort=False)["mean_prediction"].agg(["mean", "std"])
                a.index = ["none" if i == "" else i for i in a.index]
                ax.bar(a.index, a["mean"], yerr=a["std"], color="#1f77b4")
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.set_ylim(a["mean"].min() * 0.9, a["mean"].max() * 1.05)
            else:
                g["value"] = g["value"].astype(float)
                a = g.groupby("value")["mean_prediction"].agg(["mean", "std"])
                ax.plot(a.index, a["mean"], "o-", color="#1f77b4", ms=3)
                ax.fill_between(a.index, a["mean"] - a["std"], a["mean"] + a["std"], alpha=0.25)
            ax.axhline(obs, color=COLORS["base"], ls="--", lw=1)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=8)
    fig.suptitle("Partial dependence (out-of-fold, owner-grouped): classification RF (top), "
                 "regression HistGB (bottom)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "pdp.png", dpi=160)
    plt.close(fig)


def fig_holdout() -> None:
    import json

    pred = pd.read_csv(RESULTS / "holdout" / "classification_predictions.csv")
    thr = json.loads((INTERP / "threshold_results.json").read_text())
    hold = thr["holdout"]
    fpr, tpr, _ = roc_curve(pred["y"], pred["prob"])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].plot(fpr, tpr, color="#1f77b4", label="RF (holdout)")
    ax[0].plot([0, 1], [0, 1], "--", color=COLORS["base"], label="chance")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].legend(fontsize=8)
    bins = np.linspace(pred["prob"].min(), pred["prob"].max(), 11)
    idx = np.clip(np.digitize(pred["prob"], bins) - 1, 0, 9)
    cal = pred.assign(bin=idx).groupby("bin").agg(p=("prob", "mean"), y=("y", "mean"))
    ax[1].plot(cal["p"], cal["y"], "o-", color="#1f77b4")
    ax[1].plot([0.3, 1], [0.3, 1], "--", color=COLORS["base"])
    ax[1].set_xlabel("predicted probability")
    ax[1].set_ylabel("observed frequency")
    curve = pd.read_csv(INTERP / "holdout_minority_pr_curve.csv")
    ax[2].plot(curve["recall"], curve["precision"], color="#1f77b4")
    ax[2].axhline(hold["minority_prevalence"], color=COLORS["base"], ls="--",
                  label=f"prevalence {hold['minority_prevalence']:.2f}")
    ax[2].plot(hold["minority_recall"], hold["minority_precision"], "o", color="#d62728",
               label=f"threshold {thr['threshold']:.2f}")
    ax[2].set_xlabel("recall (single-owner class)")
    ax[2].set_ylabel("precision")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "holdout_roc_calibration.png", dpi=160)
    plt.close(fig)


def figures_main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_views()
    fig_sensitivity()
    fig_importance()
    if (INTERP / "pdp_oof.csv").exists():
        fig_pdp()
    if (RESULTS / "holdout" / "classification_predictions.csv").exists() and (
            INTERP / "threshold_results.json").exists():
        fig_holdout()
    print(f"Figuras em {OUT}")
