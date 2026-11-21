"""Figures of the paper, drawn from the result files in ``results/`` (command ``skillreuse figures``).

Nothing is trained here: every figure reads files written by the earlier phases (and, for
the SHAP values, the frozen text model). The figures, in the order of the paper:

    eda.png               time to first copy, first appearance by month, dating outcome
    eda_attributes.png    skewness of the attributes and correlation between them
    eda_univariate.png    share of copied skills by quartile of each attribute
    model_selection.png   spot-check of 7 algorithms and nested tuning of the best 3
    results.png           ROC and precision-recall curves, confusion matrix on the holdout
    importance.png        permutation importance (families and attributes)
    shap.png              SHAP values and dependence plots of the text model
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from skillreuse.collect import DATES, OCCURRENCES, SNAPSHOT  # noqa: E402
from skillreuse.study import (  # noqa: E402
    OUT,
    TEXT,
    TREE_ALGOS,
    interp_sample,
    load_frame,
    write_json,
)

FIG = OUT / "figures"
BLUE, LIGHT, GREY = "#2f5d8a", "#9bb7d4", "#999999"
# Readable names of the algorithms and of the attributes, as written in the paper.
NAMES = {"logistic": "Logistic reg.", "naive_bayes": "Naive Bayes", "knn": "k-NN",
         "decision_tree": "Decision tree", "random_forest": "Random forest",
         "hist_gb": "HistGB", "mlp": "MLP"}
LABEL = {
    "text_average_sentence_words": "words per sentence", "text_lexical_diversity":
    "lexical diversity", "text_average_word_length": "average word length",
    "markdown_inline_code_count": "inline code spans", "markdown_heading_count": "headings",
    "text_unique_word_count": "unique words", "frontmatter_description_char_count":
    "description length", "frontmatter_name_char_count": "name length", "text_line_count":
    "lines", "text_nonempty_line_count": "non-empty lines", "markdown_url_count": "URLs",
    "markdown_link_count": "links", "text_sentence_count": "sentences", "text_char_count":
    "characters", "text_word_count": "words", "body_chars": "body characters",
    "frontmatter_valid": "valid front matter", "markdown_max_heading_depth": "heading depth",
    "frontmatter_description_missing": "description missing",
    "markdown_fenced_code_blocks": "code blocks", "markdown_unordered_list_items":
    "bullet items", "markdown_ordered_list_items": "numbered items",
    "markdown_table_row_count": "table rows", "markdown_shell_prompt_count": "shell prompts",
    "markdown_blockquote_count": "quotes", "markdown_checkbox_count": "checkboxes",
    "frontmatter_name_missing": "name missing",
}


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure in ``results/figures/`` and close it."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)


def fig_eda() -> None:
    """Figure 1: (a) time to first copy and the window W, (b) skills by month of first
    appearance and the eligibility limit, (c) outcome of the dating of the copies."""
    win = json.loads((OUT / "phase3" / "window.json").read_text())
    dev_all = load_frame("dev", eligible_only=False)
    t0 = pd.to_datetime(dev_all["t0"], utc=True)
    # Skills with 90 days of observation that were copied within 90 days (as in the rule of W).
    obs = dev_all[t0 <= SNAPSHOT - pd.Timedelta(days=90)]
    lags = obs.loc[obs["first_copy_lag_days"] <= 90, "first_copy_lag_days"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
    xs = np.arange(0, 91)
    ax[0].plot(xs, [(lags <= x).mean() for x in xs], color=BLUE)
    ax[0].axvline(win["W"], color="#c0504d", ls="--", lw=1, label=f"W = {win['W']} days")
    ax[0].axhline(0.8, color=GREY, lw=0.7)
    ax[0].set_xlabel("days from first appearance to first copy")
    ax[0].set_ylabel("cumulative share")
    ax[0].set_title("(a) Time to first copy by another developer")
    ax[0].legend(fontsize=8)
    # Skills before October 2025 (when the format was introduced) are grouped as "earlier".
    m = t0.dt.strftime("%Y-%m")
    m = m.where(m >= "2025-10", "earlier")
    months = m.value_counts()
    months = months.reindex(["earlier"] + sorted(k for k in months.index if k != "earlier"))
    ax[1].bar(months.index, months.values, color=LIGHT)
    cut = (SNAPSHOT - pd.Timedelta(days=win["W"])).strftime("%Y-%m")
    ax[1].axvline(list(months.index).index(cut) + 0.5 if cut in months.index else len(months),
                  color="#c0504d", ls="--", lw=1, label="eligibility limit")
    ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    ax[1].set_title("(b) First appearance (T0) by month, development")
    ax[1].legend(fontsize=8)
    occ = pd.read_parquet(OCCURRENCES, columns=["content_hash", "repo_full_name", "path",
                                               "aggregator"])
    status = pd.read_parquet(DATES).merge(occ, on=["content_hash", "repo_full_name", "path"])
    status = status.loc[~status["aggregator"], "status"].value_counts(normalize=True)
    # Labels of the status codes, which are stored in Portuguese in the data files.
    lab = {"datada": "dated", "repositorio_inacessivel": "repository gone",
           "arquivo_ausente": "file gone", "hash_nao_encontrado": "hash not found",
           "falha_de_coleta": "collection failure",
           "data_posterior_ao_retrato": "date after snapshot"}
    ax[2].barh([lab.get(s, s) for s in status.index[::-1]], status.values[::-1], color=BLUE)
    for i, v in enumerate(status.values[::-1]):
        ax[2].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)
    ax[2].set_xlim(0, 1.1)
    ax[2].set_title("(c) Dating of copies outside aggregators")
    _save(fig, "eda.png")


def fig_selection() -> None:
    """Figure 4: AUC of each fold for (a) the 7 algorithms of the spot-check and (b) the
    default and tuned versions of the 3 finalists in the nested search."""
    spot = pd.read_csv(OUT / "phase4" / "spot_folds.csv")
    nested = pd.read_csv(OUT / "phase4" / "nested_folds.csv")
    order = spot.groupby("config")["roc_auc"].mean().sort_values().index
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].boxplot([spot.loc[spot.config == c, "roc_auc"] for c in order], vert=False)
    ax[0].set_yticks(range(1, len(order) + 1), [NAMES[c] for c in order])
    ax[0].axvline(0.5, color=GREY, ls="--", lw=0.8)
    ax[0].set_xlabel("ROC AUC (9 author-grouped folds)")
    ax[0].set_title("(a) Spot-checking, default hyperparameters")
    cfg = nested.groupby("config")["roc_auc"].mean().sort_values().index
    ax[1].boxplot([nested.loc[nested.config == c, "roc_auc"] for c in cfg], vert=False)
    ax[1].set_yticks(range(1, len(cfg) + 1),
                     [f"{NAMES[c.split('|')[0]]} ({c.split('|')[1]})" for c in cfg])
    ax[1].set_xlabel("ROC AUC (6 outer folds, nested search)")
    ax[1].set_title("(b) Nested tuning of the three finalists")
    _save(fig, "model_selection.png")


def fig_eda_attributes() -> None:
    """Figure 2: (a) skewness of the attributes before and after the log transform, and
    (b) Spearman correlation between the 27 text attributes."""
    dev = load_frame("dev")
    rules = json.loads((OUT / "phase3" / "eda_rules.json").read_text())
    num = [c for c in TEXT if dev[c].nunique() > 2]
    skew = dev[num].astype(float).skew()
    skew_log = np.log1p(dev[num].clip(lower=0).astype(float)).skew()
    order = skew.sort_values(ascending=False).index[:14]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(order))
    ax[0].barh(y + 0.2, np.clip(skew[order], None, 60), 0.4, label="raw (clipped at 60)",
               color="#cccccc")
    ax[0].barh(y - 0.2, [skew_log[c] if c in rules["skewed_log1p_nontree"] else skew[c]
                         for c in order], 0.4, label="after log1p", color=BLUE)
    ax[0].set_yticks(y, [LABEL.get(c, c) for c in order], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].axvline(2, color="#c0504d", ls="--", lw=0.8, label="rule: skewness > 2")
    ax[0].set_xlabel("skewness (development set)")
    ax[0].set_title("(a) Skewness of the text attributes")
    ax[0].legend(fontsize=7, loc="lower right")
    corr = dev[TEXT].astype(float).corr("spearman")
    im = ax[1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(TEXT)), [LABEL.get(c, c) for c in TEXT], rotation=90, fontsize=7)
    ax[1].set_yticks(range(len(TEXT)), [LABEL.get(c, c) for c in TEXT], fontsize=7)
    fig.colorbar(im, ax=ax[1], shrink=0.75, label="Spearman correlation")
    ax[1].set_title("(b) Correlation between text attributes")
    _save(fig, "eda_attributes.png")


def fig_eda_univariate() -> None:
    """Figure 3: share of copied skills in each quartile of each text attribute (development)."""
    uni = pd.read_csv(OUT / "phase3" / "univariate_rates.csv").set_index("feature")
    uni = uni.loc[[c for c in TEXT if c in uni.index]]
    base = json.loads((OUT / "phase3" / "eda_summary.json").read_text())["replicated_rate"]
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    im = ax.imshow(uni.values, cmap="RdBu_r", vmin=base - 0.05, vmax=base + 0.05, aspect="auto")
    n = uni.shape[1]
    ax.set_xticks(range(n), [f"Q{k + 1}" + (" (low)" if k == 0 else " (high)" if k == n - 1 else "")
                             for k in range(n)])
    ax.set_yticks(range(len(uni)), [LABEL.get(c, c) for c in uni.index], fontsize=8)
    for i in range(uni.shape[0]):
        for k in range(n):
            ax.text(k, i, f"{uni.values[i, k]:.1%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6, label=f"share copied (overall {base:.1%})")
    ax.set_title("Share of copied skills by quartile of each text attribute")
    _save(fig, "eda_univariate.png")


def fig_results() -> None:
    """Figure 5: (a) ROC and (b) precision-recall curves of the final models on the holdout,
    and (c) the confusion matrix of the text model at the frozen threshold.

    The origin models drawn are the ones with the popularity at the snapshot (the control
    reported in the paper), so the numbers agree with Table 5.
    """
    hold = json.loads((OUT / "phase6" / "holdout_results.json").read_text())
    pred = pd.read_parquet(OUT / "phase6" / "holdout_predictions_control.parquet")
    ctrl = json.loads((OUT / "phase6" / "control_popularity.json").read_text())["point"]
    point = {**hold["point"], **ctrl}
    fz = json.loads((OUT / "phase5" / "freeze.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    views = (("text", "text", BLUE), ("size", "size only", LIGHT),
             ("origin_pop", "origin repository", GREY),
             ("origin_pop+text", "origin repository + text", "#c0504d"))
    for view, lab, color in views:
        p = pred[f"prob_{view.replace('+', '_')}"]
        fpr, tpr, _ = roc_curve(pred["y"], p)
        ax[0].plot(fpr, tpr, color=color, lw=1.5,
                   label=f"{lab} (AUC {point[view]['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(pred["y"], p)
        ax[1].plot(rec, prec, color=color, lw=1.5,
                   label=f"{lab} (PR AUC {point[view]['pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) ROC curves")
    ax[0].legend(fontsize=7, loc="lower right")
    prev = hold["prevalence"]
    ax[1].axhline(prev, color=GREY, ls="--", lw=0.8, label=f"prevalence ({prev:.1%})")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_ylim(0, 0.6)
    ax[1].set_title("(b) Precision-recall curves")
    ax[1].legend(fontsize=7, loc="upper right")
    yt = pred["y"].to_numpy()
    cm = confusion_matrix(yt, (pred["prob_text"] >= fz["threshold"]).astype(int))
    share = cm / cm.sum(axis=1, keepdims=True)  # each cell as a share of its row
    ax[2].imshow(share, cmap="Blues", vmin=0, vmax=1)
    for (i, k), v in np.ndenumerate(cm):
        ax[2].text(k, i, f"{v:,}\n({share[i, k]:.0%} of row)", ha="center", va="center",
                   fontsize=9, color="white" if share[i, k] > 0.6 else "black")
    ax[2].set_xticks([0, 1], ["predicted\nnot copied", "predicted\ncopied"])
    ax[2].set_yticks([0, 1], ["not copied", "copied"])
    ax[2].set_title(f"(c) Text model, threshold {fz['threshold']:.2f}")
    _save(fig, "results.png")


def fig_interpretation() -> None:
    """Figures 6 and 7: permutation importance and SHAP values of the text model.

    Figure 6 (``importance.png``) shows the AUC drop for each family and for the ten most
    important attributes. Figure 7 (``shap.png``) shows (a) the distribution of the SHAP
    values of the 12 most influential attributes and (b) how the five most important
    attributes change the prediction as their value changes. The SHAP values are computed
    on the interpretation sample of the holdout. For a logistic regression, the SHAP value
    of an attribute is its coefficient times how far the value is from the average (in
    log-odds), so SHAP shows the effect of the coefficients on each skill.
    """
    perm = pd.read_csv(OUT / "phase7" / "permutation_holdout.csv")
    fam = perm[perm["feature"].str.startswith("family:")].iloc[::-1]
    col = perm[~perm["feature"].str.startswith("family:")].head(10).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    for a, d, lab in ((ax[0], fam, fam["feature"].str.removeprefix("family:")),
                      (ax[1], col, col["feature"].map(lambda c: LABEL.get(c, c)))):
        a.barh(lab, d["auc_drop"], color=BLUE,
               xerr=[(d["auc_drop"] - d["ci95_low"]).clip(lower=0),
                     (d["ci95_high"] - d["auc_drop"]).clip(lower=0)])
        a.set_xlabel("AUC drop when permuted (holdout, 95% CI)")
    ax[0].set_title("(a) Attribute families")
    ax[1].set_title("(b) Top 10 attributes")
    _save(fig, "importance.png")
    # SHAP values of the text model on the interpretation sample.
    fz = json.loads((OUT / "phase5" / "freeze.json").read_text())
    m = joblib.load(fz["models"]["text"]["file"])
    pipe, cols = m["pipeline"], m["columns"]
    sample = interp_sample(load_frame("holdout"))
    import shap  # noqa: PLC0415

    xt = pipe[:-1].transform(sample[cols])  # the data after preprocessing, as the model sees it
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    model = pipe.named_steps["model"]
    # Pick the explainer that fits the algorithm of the model.
    if fz["algo"] in TREE_ALGOS:
        values = shap.TreeExplainer(model).shap_values(xt)
    elif fz["algo"] == "logistic":
        values = shap.LinearExplainer(
            model, shap.maskers.Independent(xt, max_samples=len(xt))).shap_values(xt)
    else:
        sub = np.random.default_rng(0).choice(len(xt), size=min(1000, len(xt)), replace=False)
        xt, sample = xt[sub], sample.iloc[sub].reset_index(drop=True)
        values = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], xt[:100])(xt).values
    values = values[1] if isinstance(values, list) else values
    if values.ndim == 3:
        values = values[:, :, 1]
    # Position of the preprocessed column of each original attribute.
    raw = {c: next(k for k, n in enumerate(names) if n.endswith(f"__{c}")) for c in cols}
    imp = pd.Series({c: np.abs(values[:, j]).mean() for c, j in raw.items()}) \
        .sort_values(ascending=False)
    pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values}) \
        .to_csv(OUT / "phase7" / "shap_holdout.csv", index=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(xt), size=min(5000, len(xt)), replace=False)  # points to draw
    top = list(imp.index[:12])
    fig = plt.figure(figsize=(12, 8.2), layout="constrained")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1])
    # (a) one row of dots per attribute, colored by the value of the attribute.
    a0 = fig.add_subplot(gs[0, :])
    for r, c in enumerate(top[::-1]):
        v = values[pick, raw[c]]
        x = sample[c].to_numpy(dtype=float)[pick]
        q = pd.Series(x).rank(pct=True).to_numpy()
        a0.scatter(v, r + rng.uniform(-0.3, 0.3, len(v)), c=q, cmap="coolwarm", s=2, alpha=0.5)
    a0.set_yticks(range(len(top)), [LABEL.get(c, c) for c in top[::-1]], fontsize=10)
    a0.axvline(0, color=GREY, lw=0.7)
    lim = np.nanpercentile(np.abs(values[pick][:, [raw[c] for c in top]]), 99.9)
    a0.set_xlim(-lim * 1.1, lim * 1.1)
    a0.set_xlabel("SHAP value (log-odds); color: attribute value from low (blue) to high (red)")
    a0.set_title("(a) SHAP values of the 12 most influential attributes")
    # (b) dependence plots of the 5 attributes with the largest permutation importance.
    perm_top = [f for f in perm["feature"] if not f.startswith("family:")][:5]
    dep = {}
    for i, c in enumerate(perm_top):
        a = fig.add_subplot(gs[1, i])
        x = sample[c].to_numpy(dtype=float)
        a.axhline(0, color=GREY, lw=0.7)
        if set(np.unique(x[~np.isnan(x)])) <= {0.0, 1.0}:
            # binary attribute: one point cloud per category, with the number of skills
            sv = values[:, raw[c]]
            for k in (0, 1):
                s = sv[x == k]
                a.scatter(k + rng.uniform(-0.12, 0.12, len(s)), s, s=3, alpha=0.3, color=BLUE)
                a.plot([k - 0.2, k + 0.2], [s.mean()] * 2, color="black", lw=1.5)
                a.annotate(f"{s.mean():+.2f}\nn = {len(s):,}", (k, s.mean()),
                           xytext=(0, 8 if s.mean() >= 0 else -24), textcoords="offset points",
                           ha="center", fontsize=8)
            a.set_xticks([0, 1], ["no", "yes"])
            a.set_xlim(-0.6, 1.6)
            pad = 0.35 * (sv.max() - sv.min() + 1e-9)
            a.set_ylim(min(sv.min(), 0) - pad, max(sv.max(), 0) + pad)
        else:
            # numeric attribute: show the 1st to 99th percentile, so outliers do not squeeze it
            lo, hi = np.nanpercentile(x, [1, 99])
            a.scatter(x[pick], values[pick, raw[c]], s=3, alpha=0.3, color=BLUE)
            a.set_xlim(lo - 0.02 * (hi - lo + 1e-9), hi + 0.02 * (hi - lo + 1e-9))
            inside = (x >= lo) & (x <= hi)
            sv = values[inside, raw[c]]
            pad = 0.05 * (sv.max() - sv.min() + 1e-9)
            a.set_ylim(sv.min() - pad, sv.max() + pad)
        a.set_xlabel(LABEL.get(c, c), fontsize=9)
        a.set_title(("(b) SHAP dependence: " if i == 0 else "") + LABEL.get(c, c), fontsize=9,
                    loc="left" if i == 0 else "center")
        if i == 0:
            a.set_ylabel("SHAP value (log-odds)")
        dep[c] = float(pd.Series(x).corr(pd.Series(values[:, raw[c]]), method="spearman"))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "shap.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "phase7" / "shap_dependence.json", {"rows": len(xt), "top12": top,
                                                         "spearman_value_vs_shap": dep})


def main_figures() -> None:
    """Command ``skillreuse figures``: draw all the figures of the paper."""
    fig_eda()
    fig_eda_attributes()
    fig_eda_univariate()
    fig_selection()
    fig_results()
    fig_interpretation()
    print(sorted(p.name for p in FIG.glob("*.png")))
