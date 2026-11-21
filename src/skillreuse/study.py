"""Phases 3 to 7 of the protocol (paper/protocol.md): all the machine learning of the study.

The file follows the order of the phases:

    Attributes        which columns each model sees (section 3)
    Data              loading the table, the target, the folds and the metrics
    Phase 3           exploratory analysis, choice of the window W, preprocessing rules
    Preprocessing     pipelines of the seven algorithms
    Phase 4           spot-check of 7 algorithms and nested search of the best 3
    Phase 5           final models, reference models, decision threshold and freezing
    Phase 6           single evaluation on the holdout
    Control           control model with the popularity of the origin repository
    Phase 7           robustness checks and interpretation of the model
    Reproduction      check of the frozen models, and command line

All development experiments use the same partitions: 3 repetitions of 3-fold
``StratifiedGroupKFold``, grouped by author and stratified by the target (seeds 2027-2029).
The holdout is read only in ``skillreuse holdout`` (phase 6) and after it (phase 7).

Names that appear everywhere:

* a "view" is a set of columns given to a model: ``text``, ``origin``, ``origin+text``,
  ``size`` (see ``VIEWS``);
* ``replicated`` is the target: 1 if the skill was copied by another developer within W
  days of its first appearance (T0). The word "copied" in the paper is "replicated" here;
* "dev" is the development set and "holdout" is the test set, split by author.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from skillreuse.cohort import POPULARITY_TABLE, TABLE, VARIANTS_TABLE  # noqa: E402
from skillreuse.collect import DATES, OCCURRENCES, SNAPSHOT  # noqa: E402
from skillreuse.stats import N_BOOT, boot_p, holm, nb_test, owner_bootstrap  # noqa: E402

# Some models do not always converge in the iterations given; this is not an error here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = Path("results")  # one folder per phase: results/phase3, results/phase4, ...
MODELS_DIR = Path("models")  # frozen models (phase 5)
SEED = 2027  # seed of the models, the searches and the bootstrap
REPEATS, FOLDS = 3, 3  # repetitions and folds of the cross-validation
WINDOWS = [30, 60, 90]  # candidate windows W, in days

# ---------------------------------------------------------------------------
# Attributes (section 3)
# ---------------------------------------------------------------------------

# The 27 attributes of the text, by family. The families are also used to measure the
# importance of a whole group of attributes (phase 7).
FAMILIES = {
    "Size": ["text_char_count", "text_line_count", "text_nonempty_line_count",
             "text_word_count", "text_unique_word_count", "text_sentence_count", "body_chars"],
    "Text style": ["text_lexical_diversity", "text_average_word_length",
                   "text_average_sentence_words"],
    "Markdown structure": ["markdown_heading_count", "markdown_max_heading_depth",
                           "markdown_unordered_list_items", "markdown_ordered_list_items",
                           "markdown_table_row_count", "markdown_blockquote_count",
                           "markdown_checkbox_count"],
    "Code": ["markdown_fenced_code_blocks", "markdown_inline_code_count",
             "markdown_shell_prompt_count"],
    "Links": ["markdown_link_count", "markdown_url_count"],
    "Front matter": ["frontmatter_valid", "frontmatter_name_char_count",
                     "frontmatter_description_char_count", "frontmatter_name_missing",
                     "frontmatter_description_missing"],
}
TEXT = [c for fam in FAMILIES.values() for c in fam]
# The 14 attributes of the origin repository, used only by the control model.
ORIGIN = ["log_stars_at_t0", "repo_age_days_at_t0", "repo_language", "repo_license",
          "repo_is_fork", "path_depth", "path_is_root", "p_template", "p_plugin", "p_vendor",
          "p_docs", "p_pkg", "p_tool_dir", "p_claude_dir"]
CATEGORICAL = {"repo_language", "repo_license"}
# Deviation of section 13: stars and forks at the snapshot replace the stars at T0.
POPULARITY = ["log_stars_snapshot", "log_forks_snapshot"]
ORIGIN_POP = [c for c in ORIGIN if c != "log_stars_at_t0"] + POPULARITY
# Columns of each view. "size" is the simple reference model (only the size of the text).
VIEWS = {"text": TEXT, "origin": ORIGIN, "origin+text": ORIGIN + TEXT, "size": FAMILIES["Size"],
         "origin_pop": ORIGIN_POP, "origin_pop+text": ORIGIN_POP + TEXT}
MAIN_VIEWS = ["text", "origin", "origin+text"]
assert len(TEXT) == 27

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def window() -> int:
    """The window W (in days) chosen in phase 3."""
    return int(json.loads((OUT / "phase3" / "window.json").read_text())["W"])


def load_frame(split: str, w: int | None = None, variant: str = "",
               eligible_only: bool = True) -> pd.DataFrame:
    """Load the skills of one side of the split, with the target and the eligibility.

    ``split`` is ``"dev"``, ``"holdout"`` or ``"all"``. ``w`` is the window in days
    (default: the window chosen in phase 3). ``variant`` selects the definition of
    aggregator: ``""`` is the main one (100 skills), ``"_agg500"`` and ``"_aggnone"`` are
    the robustness variants.

    Two columns are computed here, from T0 and the time to first copy (section 2.3):

    * ``eligible``: T0 is at least ``w`` days before the snapshot, so the whole window was
      observed. A skill that appeared later could not have been copied within the window;
    * ``replicated``: the target, 1 if the first copy by another developer arrived within
      ``w`` days of T0 (a skill with no copy by another developer has no time and counts
      as 0).

    Only eligible skills are returned, unless ``eligible_only`` is false.
    """
    frame = pd.read_parquet(TABLE)
    if POPULARITY_TABLE.is_file():
        frame = frame.merge(pd.read_parquet(POPULARITY_TABLE), on="content_hash", how="left")
    if variant:
        frame = frame.merge(pd.read_parquet(VARIANTS_TABLE), on="content_hash", how="left")
    frame = frame[frame["split"] == split].copy() if split != "all" else frame.copy()
    t0 = pd.to_datetime(frame[f"t0{variant}"], utc=True)
    lag = frame[f"first_copy_lag_days{variant}"]
    if w is None and eligible_only:
        w = window()
    if w is not None:
        frame["eligible"] = t0.notna() & (t0 <= SNAPSHOT - pd.Timedelta(days=w))
        frame["replicated"] = (lag <= w).fillna(False).astype(int)
        if eligible_only:
            frame = frame[frame["eligible"]]
    frame["t0_used"] = t0
    return frame.sort_values("content_hash").reset_index(drop=True)


def folds(frame: pd.DataFrame, repeats: int = REPEATS, n_folds: int = FOLDS, seed: int = SEED):
    """Generate the cross-validation partitions as ``(repeat, fold, train_idx, valid_idx)``.

    Each repetition is a different shuffle (seed, seed + 1, ...). Folds are grouped by
    author, so an author is never in training and validation at the same time, and
    stratified by the target, so each fold has about the same share of copied skills. The
    same partitions are used for every algorithm, so their results can be compared.
    """
    groups = frame["owner"].astype(str).to_numpy()
    y = frame["replicated"].to_numpy()
    for r in range(1, repeats + 1):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r - 1)
        for f, (tr, va) in enumerate(splitter.split(frame, y=y, groups=groups), 1):
            if set(groups[tr]) & set(groups[va]):
                raise RuntimeError("an author is shared between training and validation")
            yield r, f, tr, va


def clf_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """Metrics of the predicted probabilities ``p`` for the true labels ``y``.

    ROC AUC (the main metric), PR AUC, Brier score, log loss and the prevalence (share of
    positives). ``w`` are optional weights per row. Probabilities are clipped to
    [1e-6, 1 - 1e-6] so that the log loss never becomes infinite.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
            "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
            "prevalence": float(np.average(y, weights=w))}


def sha256(path: Path) -> str:
    """SHA-256 of a file, used to fingerprint the frozen models and inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as readable JSON, creating the folder. Arrays become lists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
                    + "\n", encoding="utf-8")


def ci(samples: np.ndarray) -> list[float]:
    """95% percentile interval of a sample (for example of bootstrap results)."""
    return np.percentile(samples, [2.5, 97.5]).tolist()

# ---------------------------------------------------------------------------
# Phase 3: exploratory analysis, window and preprocessing rules (section 5)
# Uses only the development set, so nothing from the holdout can influence the choices.
# ---------------------------------------------------------------------------


def choose_window(dev: pd.DataFrame) -> dict[str, object]:
    """Choose the window W by the rule written in the protocol.

    Take the development skills observed for at least 90 days (T0 at least 90 days before
    the snapshot) that were copied within 90 days. Among the candidates 30, 60 and 90 days,
    W is the smallest one that covers at least 80% of their times to first copy. Looking
    only at skills with 90 days of observation avoids bias: recent skills have not had
    time to show a late first copy.
    """
    t0 = pd.to_datetime(dev["t0"], utc=True)
    obs = dev[t0 <= SNAPSHOT - pd.Timedelta(days=90)]
    lags = obs.loc[obs["first_copy_lag_days"] <= 90, "first_copy_lag_days"]
    cover = {w: float((lags <= w).mean()) for w in WINDOWS}
    w = next((w for w in WINDOWS if cover[w] >= 0.80), 90)
    # The text of "rule" is stored in window.json, so it is kept as it was written.
    return {"W": w, "coverage": cover, "skills_with_90d": len(obs),
            "copied_within_90d": len(lags), "lag_quantiles": lags.quantile([.25, .5, .75, .8, .9])
            .round(1).to_dict(), "rule": "menor W em {30,60,90} com >= 80% dos tempos"}


def eda_rules(dev: pd.DataFrame) -> dict[str, object]:
    """Derive the preprocessing rules from the development data (section 5).

    * ``constant``: columns with a single value, which carry no information;
    * ``skewed_log1p_nontree``: numeric columns with only non-negative values and skewness
      above 2, which get a log transform in models that are sensitive to scale (the very
      long tail would let a few extreme skills dominate them);
    * ``collinear_pairs``: pairs of columns with absolute Spearman correlation above 0.95;
    * ``drop_collinear_nontree``: columns removed from scale-sensitive models, so that
      only one column of each correlated group stays. A column is kept unless it correlates
      above 0.95 with a column that was already kept (greedy, in the order of ``TEXT``).

    Tree models ignore the last two rules: they do not mind skewness or correlation.
    """
    cols = TEXT + ORIGIN
    numeric = [c for c in cols if c not in CATEGORICAL]
    constant = [c for c in cols if dev[c].nunique(dropna=True) <= 1]
    order = [c for c in numeric if c not in constant]
    skewed = [c for c in order if dev[c].dropna().min() >= 0 and dev[c].nunique() > 2
              and dev[c].dropna().astype(float).skew() > 2]
    corr = dev[order].astype(float).corr("spearman").abs()
    pairs, kept, drop = [], [], []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if corr.loc[a, b] > 0.95:
                pairs.append({"a": a, "b": b, "abs_spearman": float(corr.loc[a, b])})
    for c in order:
        (drop if any(corr.loc[c, k] > 0.95 for k in kept) else kept).append(c)
    return {"constant": constant, "skewed_log1p_nontree": skewed, "collinear_pairs": pairs,
            "drop_collinear_nontree": drop,
            "inconsistent_to_missing": {"repo_age_days_at_t0": "negativo"}}


def run_eda() -> None:
    """Phase 3 (command ``skillreuse eda``): exploratory analysis of the development set.

    Writes to ``results/phase3/``: the window (``window.json``), the preprocessing rules
    (``eda_rules.json``), a profile of every attribute (``feature_profile.csv``), the share
    of copied skills by quartile of each text attribute (``univariate_rates.csv``) and a
    summary of the data (``eda_summary.json``).
    """
    out = OUT / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    dev_all = load_frame("dev", eligible_only=False)
    win = choose_window(dev_all)
    write_json(out / "window.json", win)
    dev = load_frame("dev", w=win["W"])
    # A negative age means T0 is before the creation of the repository: impossible, so missing.
    dev.loc[dev["repo_age_days_at_t0"] < 0, "repo_age_days_at_t0"] = np.nan
    rules = eda_rules(dev)
    write_json(out / "eda_rules.json", rules)
    # Outcome of the dating of the copies outside aggregators, for the paper.
    occ = pd.read_parquet(OCCURRENCES,
                          columns=["content_hash", "repo_full_name", "path", "aggregator"])
    status = pd.read_parquet(DATES).merge(occ, on=["content_hash", "repo_full_name", "path"]) \
        .merge(pd.read_parquet(TABLE, columns=["content_hash", "split"]), on="content_hash")
    status = status[(status["split"] == "dev") & ~status["aggregator"].astype(bool)]
    status = status["status"].value_counts(normalize=True).to_dict()
    # Profile of each attribute: missing values, distribution and skewness.
    prof = []
    for c in TEXT + ORIGIN:
        s = dev[c]
        row = {"feature": c, "missing_share": float(s.isna().mean()), "n_unique": int(s.nunique())}
        if c not in CATEGORICAL:
            v = s.dropna().astype(float)
            row.update({"min": float(v.min()), "median": float(v.median()),
                        "p99": float(v.quantile(.99)), "max": float(v.max()),
                        "skew": float(v.skew())})
        prof.append(row)
    pd.DataFrame(prof).to_csv(out / "feature_profile.csv", index=False)
    # Share of copied skills in each quartile of an attribute. Ranking with "first" breaks
    # ties by order, so zero-heavy attributes are split arbitrarily between quartiles.
    uni = []
    for c in [c for c in TEXT if dev[c].nunique() > 4]:
        q = pd.qcut(dev[c].rank(method="first"), 4, labels=False)
        uni.append({"feature": c, **{f"q{int(k) + 1}": float(v) for k, v in
                                     dev.groupby(q)["replicated"].mean().items()}})
    pd.DataFrame(uni).to_csv(out / "univariate_rates.csv", index=False)
    summary = {"dev_eligible_rows": len(dev), "dev_authors": int(dev["owner"].nunique()),
               "dev_all_rows": len(dev_all), "replicated_rate": float(dev["replicated"].mean()),
               "collection_status_dev": status,
               "t0_by_month": dev["t0_used"].dt.strftime("%Y-%m").value_counts().sort_index()
               .to_dict(),
               "all_copies_dated_share": float(dev["all_copies_dated"].mean())}
    write_json(out / "eda_summary.json", summary)
    print(json.dumps({"window": win, **{k: summary[k] for k in ("dev_eligible_rows",
                                                                "replicated_rate")}}, indent=2))

# ---------------------------------------------------------------------------
# Preprocessing and algorithms (sections 5 and 6)
# ---------------------------------------------------------------------------

# Tree models need no scaling, no log and no removal of correlated columns.
TREE_ALGOS = {"decision_tree", "random_forest", "hist_gb"}
ALGOS = ["logistic", "naive_bayes", "knn", "decision_tree", "random_forest", "hist_gb", "mlp"]


def _log1p_clip(x: object) -> np.ndarray:
    """log(1 + x), with negative values clipped to 0 first.

    This function (and ``_dense``) is stored by name inside the frozen models in
    ``models/*.joblib``, so it must not be renamed or moved.
    """
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def _dense(x: object) -> np.ndarray:
    """Turn a sparse matrix (the output of one-hot encoding) into a dense array."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def load_rules() -> dict[str, object]:
    """Load the preprocessing rules of phase 3, plus those of the popularity columns.

    The popularity columns were added later (deviation of section 13), so their rules are
    in a separate file written by ``run_control`` and merged here when it exists.
    """
    rules = json.loads((OUT / "phase3" / "eda_rules.json").read_text())
    extra = OUT / "phase5" / "control_rules.json"  # rules of the popularity columns (deviation)
    if extra.is_file():
        for k, v in json.loads(extra.read_text()).items():
            if isinstance(v, list):
                rules[k] = rules.get(k, []) + [x for x in v if x not in rules.get(k, [])]
    return rules


def view_columns(view: str, algo: str, rules: dict[str, object]) -> list[str]:
    """Columns of a view for an algorithm: remove constant columns, and for algorithms that
    are not trees also the columns removed for collinearity."""
    cols = [c for c in VIEWS[view] if c not in rules["constant"]]
    if algo not in TREE_ALGOS:
        cols = [c for c in cols if c not in rules["drop_collinear_nontree"]]
    return cols


def preprocessor(columns: list[str], algo: str, rules: dict[str, object]) -> ColumnTransformer:
    """Build the preprocessing of the columns, in up to three blocks.

    * numeric columns: median imputation (with a flag for the missing ones) and, for
      algorithms that are not trees, standardization;
    * skewed numeric columns (not trees): imputation, then log(1 + x), then standardization;
    * categorical columns: constant imputation and one-hot encoding, with categories that
      appear fewer than 10 times grouped together.

    This is only a description: nothing is learned until the pipeline is fitted, and then
    the medians, means and categories come from the training data only.
    """
    tree = algo in TREE_ALGOS
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c not in CATEGORICAL]
    log_cols = [] if tree else [c for c in num if c in rules["skewed_log1p_nontree"]]
    plain = [c for c in num if c not in log_cols]
    blocks: list[tuple[str, object, list[str]]] = []
    if plain:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if not tree:
            steps.append(("scale", StandardScaler()))
        blocks.append(("num", Pipeline(steps), plain))
    if log_cols:
        blocks.append(("num_log", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
            ("scale", StandardScaler())]), log_cols))
    if cat:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat))
    return ColumnTransformer(blocks, remainder="drop")


def estimator(algo: str, threads: int = -1) -> object:
    """A new model with the default hyperparameters of scikit-learn (and fixed seeds)."""
    return {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "naive_bayes": lambda: GaussianNB(),
        "knn": lambda: KNeighborsClassifier(n_jobs=threads),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=threads),
        "hist_gb": lambda: HistGradientBoostingClassifier(random_state=SEED),
        "mlp": lambda: MLPClassifier(random_state=SEED),
    }[algo]()


# Hyperparameters searched in phase 4 (10 random configurations per algorithm).
# ``loguniform`` draws on a log scale, ``randint`` an integer, a list one of its values.
SPACES: dict[str, dict[str, object]] = {
    "logistic": {"C": loguniform(1e-3, 1e2), "class_weight": [None, "balanced"]},
    "naive_bayes": {"var_smoothing": loguniform(1e-12, 1e-6)},
    "knn": {"n_neighbors": randint(5, 102), "weights": ["uniform", "distance"], "p": [1, 2]},
    "decision_tree": {"max_depth": randint(3, 26), "min_samples_leaf": randint(1, 61),
                      "class_weight": [None, "balanced"]},
    "random_forest": {"n_estimators": [100, 200, 300], "max_depth": randint(5, 26),
                      "min_samples_leaf": randint(1, 31), "max_features": ["sqrt", 0.5],
                      "class_weight": [None, "balanced_subsample"]},
    "hist_gb": {"learning_rate": loguniform(0.02, 0.2), "max_iter": randint(100, 501),
                "max_leaf_nodes": randint(8, 65), "min_samples_leaf": randint(5, 101),
                "l2_regularization": loguniform(1e-3, 10.0), "class_weight": [None, "balanced"]},
    "mlp": {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": loguniform(1e-5, 1e-2),
            "learning_rate_init": loguniform(1e-4, 1e-2), "early_stopping": [True]},
}


def pipeline(algo: str, view: str, rules: dict[str, object], params: dict | None = None,
             threads: int = -1) -> tuple[Pipeline, list[str]]:
    """Build the full model: preprocessing, then (if needed) densifying, then the algorithm.

    Returns the pipeline and the list of columns it expects. Because everything is inside
    one pipeline, it is fitted on the training data of each fold only, so no information of
    the validation data leaks into training. Logistic regression takes the sparse matrix
    directly; the other algorithms need it dense.
    """
    cols = view_columns(view, algo, rules)
    steps = [("pre", preprocessor(cols, algo, rules))]
    if algo != "logistic":
        steps.append(("dense", FunctionTransformer(_dense, accept_sparse=True)))
    steps.append(("model", estimator(algo, threads)))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
    return pipe, cols


def _ratio(rows: pd.DataFrame) -> float:
    """Mean ratio (validation size / training size) of the folds, used by ``nb_test``."""
    return float((rows["validation_rows"] / rows["train_rows"]).mean())


def compare_to_best(folds_df: pd.DataFrame, key: str = "config") -> pd.DataFrame:
    """Compare the best configuration (highest mean AUC) with each of the others.

    Uses a paired Nadeau-Bengio test on the AUC of each fold, and adjusts the p-values for
    the number of comparisons with Holm's method.
    """
    wide = folds_df.pivot_table(index=["repeat", "fold"], columns=key, values="roc_auc")
    best = wide.mean().idxmax()
    ratio = _ratio(folds_df)
    rows = [{"best": best, "other": o, **nb_test((wide[best] - wide[o]).to_numpy(), ratio)}
            for o in wide.columns if o != best]
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_nb"].tolist())
    return out

# ---------------------------------------------------------------------------
# Phase 4: spot-check and nested search (sections 6.1 and 6.2)
# ---------------------------------------------------------------------------


def run_spot(repeats: int = REPEATS, out: Path = OUT / "phase4") -> None:
    """Spot-check (command ``skillreuse spot``): the 7 algorithms with default settings.

    Each algorithm is trained and validated on exactly the same 9 partitions of the
    development set, using the text attributes. The 3 with the highest mean AUC go on to
    the nested search. Writes ``spot_folds.csv`` (one row per fold and algorithm),
    ``spot_summary.csv``, ``spot_tests.csv`` and ``selection.json``.
    """
    out.mkdir(parents=True, exist_ok=True)
    dev, rules = load_frame("dev"), load_rules()
    y = dev["replicated"].to_numpy()
    rows = []
    for r, f, tr, va in folds(dev, repeats=repeats):
        for algo in ALGOS:
            t = time.monotonic()
            pipe, cols = pipeline(algo, "text", rules)
            pipe.fit(dev.iloc[tr][cols], y[tr])
            p = pipe.predict_proba(dev.iloc[va][cols])[:, 1]
            rows.append({"repeat": r, "fold": f, "config": algo, "train_rows": len(tr),
                         "validation_rows": len(va), "seconds": time.monotonic() - t,
                         **clf_metrics(y[va], p)})
        print(f"[spot] r{r} f{f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "spot_folds.csv", index=False)
    summary = df.groupby("config")[["roc_auc", "pr_auc", "brier", "seconds"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(out / "spot_summary.csv")
    compare_to_best(df).to_csv(out / "spot_tests.csv", index=False)
    write_json(out / "selection.json", {"ranking": summary.index.tolist(),
                                        "selected_top3": summary.index[:3].tolist()})
    print(summary.round(4).to_string())


N_ITER, INNER_FOLDS, N_JOBS = 10, 3, 4  # configurations tried, inner folds, parallel jobs


def inner_cv(train: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner folds of the nested search: 3 folds of the training data, grouped by author."""
    splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(train, y=train["replicated"], groups=train["owner"].astype(str)))


def search(algo: str, view: str, train: pd.DataFrame, rules: dict, seed: int
           ) -> RandomizedSearchCV:
    """Random search of ``N_ITER`` configurations of ``SPACES[algo]`` on the training data.

    Each configuration is scored by AUC on the inner folds, and the best one is refitted
    on all the training data.
    """
    pipe, cols = pipeline(algo, view, rules, threads=1)
    rs = RandomizedSearchCV(pipe, {f"model__{k}": v for k, v in SPACES[algo].items()},
                            n_iter=N_ITER, cv=inner_cv(train, seed), scoring="roc_auc",
                            n_jobs=N_JOBS, random_state=SEED, refit=True, error_score="raise")
    return rs.fit(train[cols], train["replicated"].to_numpy())


def run_tune() -> None:
    """Nested search (command ``skillreuse tune``) of the 3 best algorithms of the spot-check.

    In each of 6 outer folds (2 repetitions x 3 folds) the search picks its best
    configuration using only the training part, and that configuration is then measured on
    the outer validation fold, which the search never saw. The default version of each
    algorithm is measured on the same folds. The outer result is an honest estimate of
    "search, then train", while the inner score of the search would be optimistic.

    The winner is chosen by the rule of the protocol: the configuration with the highest
    mean AUC wins, unless it is not significantly better than others (Nadeau-Bengio with
    Holm); among the tied configurations, the one that trains fastest wins. Simplicity is
    accepted when the extra complexity does not pay off. Writes ``nested_folds.csv``,
    ``nested_summary.csv``, ``nested_tests.csv`` and ``winner.json``.
    """
    out = OUT / "phase4"
    dev, rules = load_frame("dev"), load_rules()
    top3 = json.loads((out / "selection.json").read_text())["selected_top3"]
    y = dev["replicated"].to_numpy()
    rows = []
    for r, f, tr, va in folds(dev, repeats=2):
        train, valid = dev.iloc[tr], dev.iloc[va]
        base = {"repeat": r, "fold": f, "train_rows": len(tr), "validation_rows": len(va)}
        for algo in top3:
            pipe, cols = pipeline(algo, "text", rules)
            t = time.monotonic()
            pipe.fit(train[cols], y[tr])
            secs = time.monotonic() - t
            rows.append({**base, "config": f"{algo}|default", "fit_seconds": secs,
                         **clf_metrics(y[va], pipe.predict_proba(valid[cols])[:, 1])})
            rs = search(algo, "text", train, rules, seed=SEED + r - 1)
            rows.append({**base, "config": f"{algo}|tuned", "fit_seconds": rs.refit_time_,
                         **clf_metrics(y[va], rs.best_estimator_.predict_proba(valid[cols])[:, 1]),
                         "inner_auc": float(rs.best_score_),
                         "best_params": json.dumps({k.removeprefix("model__"): v for k, v in
                                                    rs.best_params_.items()}, default=str)})
            print(f"[tune] r{r} f{f} {algo}: default {rows[-2]['roc_auc']:.4f} "
                  f"tuned {rows[-1]['roc_auc']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "nested_folds.csv", index=False)
    summary = df.groupby("config").agg(mean=("roc_auc", "mean"), std=("roc_auc", "std"),
                                       fit_seconds=("fit_seconds", "mean")) \
        .sort_values("mean", ascending=False)
    summary.to_csv(out / "nested_summary.csv")
    tests = compare_to_best(df)
    tests.to_csv(out / "nested_tests.csv", index=False)
    best = summary.index[0]
    # Tied configurations: the best one and every one that is not significantly worse.
    tied = [best] + tests.loc[tests["p_holm"] >= 0.05, "other"].tolist()
    winner = summary.loc[tied, "fit_seconds"].idxmin()  # tie-break rule of section 6.2
    algo, variant = winner.split("|")
    # The text of "rule" is stored in winner.json, so it is kept as it was written.
    write_json(out / "winner.json", {"best_auc": best, "tied": tied, "winner": winner,
                                     "algo": algo, "variant": variant,
                                     "rule": "maior AUC externa; entre empatados, menor tempo"})
    print(summary.round(4).to_string(), "\nwinner:", winner)

# ---------------------------------------------------------------------------
# Phase 5: final models, reference model, threshold and freezing (sections 6.3 to 6.6)
# ---------------------------------------------------------------------------


def final_params(view: str, dev: pd.DataFrame, rules: dict, winner: dict) -> dict:
    """Hyperparameters of the final model of a view: none if the winner uses the defaults,
    otherwise the best configuration of a new search on the whole development set."""
    if winner["variant"] == "default":
        return {}
    rs = search(winner["algo"], view, dev, rules, seed=SEED)
    return {k.removeprefix("model__"): v for k, v in rs.best_params_.items()}


def balanced_accuracy(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """Mean of the recall of the positives and of the negatives at threshold ``t``."""
    pred = p >= t
    return float((np.mean(pred[y == 1]) + np.mean(~pred[y == 0])) / 2)


def threshold_report(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    """Metrics that depend on the decision threshold ``t``: precision, recall, balanced
    accuracy and the counts of the confusion matrix."""
    pred = p >= t
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": t, "balanced_accuracy": balanced_accuracy(y, p, t), "precision": prec,
            "recall": rec, "true_positive": tp, "false_positive": fp, "false_negative": fn}


def run_final() -> None:
    """Final models (command ``skillreuse final``), validated on the development folds.

    The winning algorithm is trained on four sets of columns: ``text`` (the model of the
    research question), ``origin`` (repository only), ``origin+text`` and ``size`` (only the
    size of the text, the simple reference). Each one is validated on the 9 partitions.
    Also tests, on the folds, the hypotheses H1 (text above 0.5), C1 (origin + text above
    origin) and text above size. Writes ``final_params.json``, ``dev_views_folds.csv``,
    ``dev_views_summary.csv``, ``dev_oof_text.csv`` (out-of-fold probabilities of the text
    model, used to choose the threshold) and ``dev_tests.json``.
    """
    out = OUT / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    dev, rules = load_frame("dev"), load_rules()
    winner = json.loads((OUT / "phase4" / "winner.json").read_text())
    algo, y = winner["algo"], dev["replicated"].to_numpy()
    params = {v: final_params(v, dev, rules, winner) for v in MAIN_VIEWS}
    params["size"] = params["text"]  # reference: same algorithm and hyperparameters as the text
    write_json(out / "final_params.json", {"algo": algo, "variant": winner["variant"],
                                           "views": params})
    rows, oof = [], []
    for r, f, tr, va in folds(dev):
        for view in [*MAIN_VIEWS, "size"]:
            pipe, cols = pipeline(algo, view, rules, params[view])
            p = pipe.fit(dev.iloc[tr][cols], y[tr]).predict_proba(dev.iloc[va][cols])[:, 1]
            rows.append({"repeat": r, "fold": f, "config": view, "train_rows": len(tr),
                         "validation_rows": len(va), **clf_metrics(y[va], p)})
            if view == "text":
                oof.append(pd.DataFrame({"repeat": r, "fold": f, "y": y[va], "prob": p}))
        print(f"[final] r{r} f{f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "dev_views_folds.csv", index=False)
    pd.concat(oof).to_csv(out / "dev_oof_text.csv", index=False)
    s = df.groupby("config")[["roc_auc", "pr_auc", "brier"]].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    s.to_csv(out / "dev_views_summary.csv")
    wide, ratio = df.pivot_table(index=["repeat", "fold"], columns="config",
                                 values="roc_auc"), _ratio(df)
    write_json(out / "dev_tests.json", {
        "H1_text_vs_0.5": nb_test((wide["text"] - 0.5).to_numpy(), ratio),
        "C1_origin+text_vs_origin": nb_test((wide["origin+text"] - wide["origin"]).to_numpy(),
                                            ratio),
        "text_vs_size": nb_test((wide["text"] - wide["size"]).to_numpy(), ratio)})
    print(s.round(4).to_string())


def run_freeze() -> None:
    """Freeze the models and the threshold (command ``skillreuse freeze``), before the holdout.

    Each model is trained on the whole development set and saved in ``models/`` together
    with its list of columns. The decision threshold is the value between 0.01 and 0.99
    that maximizes the balanced accuracy of the out-of-fold predictions of the text model.
    ``freeze.json`` records a SHA-256 of every model and of the inputs (table, window,
    rules, winner, parameters and protocol), with ``holdout_read`` set to false: from now
    on any change to a model would be detected.
    """
    dev, rules = load_frame("dev"), load_rules()
    fp = json.loads((OUT / "phase5" / "final_params.json").read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y, entries = dev["replicated"].to_numpy(), {}
    for view in [*MAIN_VIEWS, "size"]:
        pipe, cols = pipeline(fp["algo"], view, rules, fp["views"][view])
        pipe.fit(dev[cols], y)
        path = MODELS_DIR / f"{view.replace('+', '_')}.joblib"
        joblib.dump({"pipeline": pipe, "columns": cols, "view": view}, path, compress=3)
        entries[view] = {"file": str(path), "sha256": sha256(path), "params": fp["views"][view]}
    oof = pd.read_csv(OUT / "phase5" / "dev_oof_text.csv")
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    yo, po = oof["y"].to_numpy(), oof["prob"].to_numpy()
    thr = float(grid[np.argmax([balanced_accuracy(yo, po, t) for t in grid])])
    files = {str(p): sha256(p) for p in [TABLE, OUT / "phase3" / "window.json",
                                         OUT / "phase3" / "eda_rules.json",
                                         OUT / "phase4" / "winner.json",
                                         OUT / "phase5" / "final_params.json",
                                         Path("paper/protocol.md")]}
    write_json(OUT / "phase5" / "freeze.json", {
        "frozen_at_local": datetime.now().astimezone().isoformat(), "algo": fp["algo"],
        "models": entries, "threshold": thr, "threshold_dev": threshold_report(yo, po, thr),
        "inputs_sha256": files, "holdout_read": False})
    print("frozen; threshold", thr)

# ---------------------------------------------------------------------------
# Phase 6: single evaluation on the holdout (section 7)
# ---------------------------------------------------------------------------


def frozen() -> dict:
    """Load ``freeze.json`` after checking that no frozen model file was changed."""
    fz = json.loads((OUT / "phase5" / "freeze.json").read_text())
    for view, e in fz["models"].items():
        if sha256(Path(e["file"])) != e["sha256"]:
            raise RuntimeError(f"frozen model was changed: {view}")
    return fz


def predict(fz: dict, view: str, frame: pd.DataFrame) -> np.ndarray:
    """Probability of being copied for each row of ``frame``, from the frozen model of a view."""
    m = joblib.load(fz["models"][view]["file"])
    return m["pipeline"].predict_proba(frame[m["columns"]])[:, 1]


def run_holdout(allow_reevaluation: bool = False) -> None:
    """The single evaluation (command ``skillreuse holdout``): the only look at the holdout.

    Applies the frozen models to the holdout and computes the AUC and the other metrics.
    Confidence intervals come from 2,000 bootstrap resamples of authors. Then tests:

    * H1: the lower end of the interval of the AUC of the text model is above 0.5;
    * C1: the lower end of the interval of AUC(origin + text) - AUC(origin) is above 0.
      The control reported in the paper replaces it in ``run_control``;

    both with p < 0.05 after the Holm correction for the two tests. Also reports the gain
    of the text over the size-only reference and the threshold metrics. After the run, a
    lock file is written and the command refuses to run again unless
    ``--allow-reevaluation`` is given (``make rerun`` uses it, to reproduce the numbers).
    """
    out = OUT / "phase6"
    lock = out / "evaluation_lock.json"
    if lock.exists() and not allow_reevaluation:
        raise SystemExit(f"Holdout already evaluated ({lock}). "
                         "Re-evaluating requires a declaration.")
    fz = frozen()
    hold = load_frame("holdout")
    y, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    probs = {v: predict(fz, v, hold) for v in fz["models"]}
    point = {v: clf_metrics(y, p) for v, p in probs.items()}
    boot = {v: owner_bootstrap(owners, lambda w, p=p: roc_auc_score(y, p, sample_weight=w))
            for v, p in probs.items()}
    diff = boot["origin+text"] - boot["origin"]
    h1 = {"auc": point["text"]["roc_auc"], "ci95": ci(boot["text"]),
          "p_boot": boot_p(boot["text"], 0.5)}
    c1 = {"diff": point["origin+text"]["roc_auc"] - point["origin"]["roc_auc"],
          "ci95": ci(diff), "p_boot": boot_p(diff, 0.0)}
    h1["p_holm"], c1["p_holm"] = holm([h1["p_boot"], c1["p_boot"]])
    h1["success"] = bool(h1["ci95"][0] > 0.5 and h1["p_holm"] < 0.05)
    c1["success"] = bool(c1["ci95"][0] > 0 and c1["p_holm"] < 0.05)
    size = {"diff": point["text"]["roc_auc"] - point["size"]["roc_auc"],
            "ci95": ci(boot["text"] - boot["size"])}
    result = {"evaluated_at_local": datetime.now().astimezone().isoformat(), "window": window(),
              "holdout_rows": len(hold), "holdout_authors": int(len(set(owners))),
              "prevalence": float(y.mean()), "point": point,
              "view_ci95": {v: ci(b) for v, b in boot.items()}, "H1": h1, "C1": c1,
              "text_minus_size": size, "threshold": threshold_report(y, probs["text"],
                                                                     fz["threshold"]),
              "n_boot": N_BOOT}
    write_json(out / "holdout_results.json", result)
    pd.DataFrame({"content_hash": hold["content_hash"], "y": y,
                  **{f"prob_{v.replace('+', '_')}": p for v, p in probs.items()}}) \
        .to_parquet(out / "holdout_predictions.parquet", index=False)
    write_json(lock, {"evaluated_at_local": result["evaluated_at_local"],
                      "freeze_sha256": sha256(OUT / "phase5" / "freeze.json")})
    print(json.dumps({"H1": h1, "C1": c1, "text_minus_size": size}, indent=2,
                     default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))

# ---------------------------------------------------------------------------
# Control with the popularity of the origin repository (deviation of section 13)
# ---------------------------------------------------------------------------


def run_control() -> None:
    """Control model with stars and forks at the snapshot (command ``skillreuse control``).

    The GitHub API no longer reports when each star was given, so the stars at T0 were not
    available and the original control (C1) did not include popularity. Here the control
    is trained again with the stars and forks of the origin repository at the snapshot,
    which is later than T0. This favors the repository, so if the text still adds
    information the conclusion is stronger. Only C1 is evaluated again on the holdout; the
    text model and H1 do not change, and the original results (phase 6) are kept.
    Writes ``control_rules.json``, ``freeze_control.json``, ``control_popularity.json``
    and the holdout predictions with the new views.
    """
    dev = load_frame("dev")
    base = eda_rules(dev.assign(**{c: dev[c] for c in POPULARITY}))
    extra = {"constant": [c for c in base["constant"] if c in POPULARITY],
             "skewed_log1p_nontree": [c for c in base["skewed_log1p_nontree"] if c in POPULARITY],
             "drop_collinear_nontree": [c for c in base["drop_collinear_nontree"]
                                        if c in POPULARITY]}
    write_json(OUT / "phase5" / "control_rules.json", extra)
    rules, fp = load_rules(), json.loads((OUT / "phase5" / "final_params.json").read_text())
    winner = json.loads((OUT / "phase4" / "winner.json").read_text())
    y, views = dev["replicated"].to_numpy(), ["origin_pop", "origin_pop+text"]
    params = {v: final_params(v, dev, rules, winner) for v in views}
    rows = []
    for r, f, tr, va in folds(dev):
        for v in views:
            pipe, cols = pipeline(fp["algo"], v, rules, params[v])
            p = pipe.fit(dev.iloc[tr][cols], y[tr]).predict_proba(dev.iloc[va][cols])[:, 1]
            rows.append({"repeat": r, "fold": f, "config": v, "train_rows": len(tr),
                         "validation_rows": len(va), **clf_metrics(y[va], p)})
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["repeat", "fold"], columns="config", values="roc_auc")
    dev_test = nb_test((wide["origin_pop+text"] - wide["origin_pop"]).to_numpy(), _ratio(df))
    entries = {}
    for v in views:
        pipe, cols = pipeline(fp["algo"], v, rules, params[v])
        pipe.fit(dev[cols], y)
        path = MODELS_DIR / f"{v.replace('+', '_')}.joblib"
        joblib.dump({"pipeline": pipe, "columns": cols, "view": v}, path, compress=3)
        entries[v] = {"file": str(path), "sha256": sha256(path), "params": params[v]}
    write_json(OUT / "phase5" / "freeze_control.json", {
        "frozen_at_local": datetime.now().astimezone().isoformat(), "models": entries,
        "dev_auc": df.groupby("config")["roc_auc"].agg(["mean", "std"]).to_dict("index"),
        "dev_C1": dev_test})
    hold = load_frame("holdout")
    yh, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    probs = {}
    for v, e in entries.items():
        m = joblib.load(e["file"])
        probs[v] = m["pipeline"].predict_proba(hold[m["columns"]])[:, 1]
    fz = frozen()
    p_text = predict(fz, "text", hold)
    boot = {v: owner_bootstrap(owners, lambda w, p=p: roc_auc_score(yh, p, sample_weight=w))
            for v, p in probs.items()}
    b_text = owner_bootstrap(owners, lambda w: roc_auc_score(yh, p_text, sample_weight=w))
    diff = boot["origin_pop+text"] - boot["origin_pop"]
    c1 = {"diff": float(roc_auc_score(yh, probs["origin_pop+text"])
                        - roc_auc_score(yh, probs["origin_pop"])),
          "ci95": ci(diff), "p_boot": boot_p(diff, 0.0)}
    # Holm over the same two tests as before: H1 (text) and this C1.
    c1["p_holm_with_H1"] = holm([boot_p(b_text, 0.5), c1["p_boot"]])[1]
    c1["success"] = bool(c1["ci95"][0] > 0 and c1["p_holm_with_H1"] < 0.05)
    # The text of "note" is stored in control_popularity.json, so it is kept as written.
    write_json(OUT / "phase6" / "control_popularity.json", {
        "note": "desvio da secao 13: popularidade no retrato (posterior a T0) no controle",
        "evaluated_at_local": datetime.now().astimezone().isoformat(),
        "point": {v: clf_metrics(yh, p) for v, p in probs.items()},
        "view_ci95": {v: ci(b) for v, b in boot.items()}, "C1_popularity": c1})
    pd.read_parquet(OUT / "phase6" / "holdout_predictions.parquet").assign(
        **{f"prob_{v.replace('+', '_')}": p for v, p in probs.items()}).to_parquet(
        OUT / "phase6" / "holdout_predictions_control.parquet", index=False)
    print(json.dumps({"dev": dev_test, "C1_popularity": c1}, indent=2, default=str))


def main_control() -> None:
    run_control()


# ---------------------------------------------------------------------------
# Phase 7: sensitivity and robustness (section 8)
# ---------------------------------------------------------------------------

N_BOOT_SENS = 500  # bootstrap resamples in the robustness checks (2,000 in the main result)


def _auc_ci(y: np.ndarray, p: np.ndarray, owners: np.ndarray) -> dict[str, object]:
    """AUC with a 95% interval by author bootstrap. If ``y`` has a single class the AUC
    does not exist and is returned as ``None``."""
    if len(np.unique(y)) < 2:
        return {"rows": len(y), "prevalence": float(np.mean(y)), "auc": None}
    b = owner_bootstrap(owners, lambda w: roc_auc_score(y, p, sample_weight=w), n_boot=N_BOOT_SENS)
    return {"rows": len(y), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "ci95": ci(b)}


def _retrain_eval(train: pd.DataFrame, test: pd.DataFrame, fz: dict, rules: dict) -> dict:
    """Train the text model (same algorithm and hyperparameters as the final one) on
    ``train`` and report its AUC on ``test``."""
    fp = json.loads((OUT / "phase5" / "final_params.json").read_text())
    pipe, cols = pipeline(fp["algo"], "text", rules, fp["views"]["text"])
    pipe.fit(train[cols], train["replicated"].to_numpy())
    p = pipe.predict_proba(test[cols])[:, 1]
    return {"train_rows": len(train), **_auc_ci(test["replicated"].to_numpy(), p,
                                                test["owner"].astype(str).to_numpy())}


def run_sensitivity() -> None:
    """Robustness checks (command ``skillreuse sensitivity``), done after the single evaluation.

    Does the result depend on our own choices? Seven checks, all on the holdout:

    * aggregator limit of 500 skills, and no exclusion of aggregators (target and T0 are
      recomputed and the model is trained again);
    * windows of 30 and 90 days instead of 60 (model trained again);
    * only skills whose copies could all be dated, since an undated copy could hide the
      true origin or a copy inside the window (frozen model);
    * temporal split: train on development skills that appeared before a cutoff (the 70th
      percentile of T0) and test on holdout skills that appeared after it, which is the
      closest test of predicting the future (model trained again);
    * only skills whose front matter name never appears in the development set, to rule
      out that the model just recognizes variants of known skills (frozen model).

    Writes ``results/phase7/sensitivity.json``.
    """
    if not (OUT / "phase6" / "evaluation_lock.json").exists():
        raise SystemExit("The robustness checks come after the single evaluation (phase 6).")
    fz, rules, w = frozen(), load_rules(), window()
    res: dict[str, object] = {"window_main": w}
    for variant in ("_agg500", "_aggnone"):  # needs the aggregator copies (cohort --variants)
        res[f"aggregators{variant}"] = _retrain_eval(
            load_frame("dev", w, variant), load_frame("holdout", w, variant), fz, rules) \
            if VARIANTS_TABLE.is_file() else "pendente: coleta dos agregadores"
    for alt in [x for x in WINDOWS if x != w]:
        res[f"window_{alt}"] = _retrain_eval(load_frame("dev", alt), load_frame("holdout", alt),
                                             fz, rules)
    hold, dev = load_frame("holdout"), load_frame("dev")
    p = predict(fz, "text", hold)
    y, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    full = hold["all_copies_dated"].to_numpy(dtype=bool)
    res["all_copies_dated"] = _auc_ci(y[full], p[full], owners[full])
    c = dev["t0_used"].quantile(0.70)  # cutoff of the temporal split
    res["temporal"] = {"cutoff": str(c), **_retrain_eval(dev[dev["t0_used"] < c],
                                                         hold[hold["t0_used"] >= c], fz, rules)}
    names = set(dev.loc[dev["name_norm"] != "", "name_norm"])
    unseen = ~(hold["name_norm"].ne("") & hold["name_norm"].isin(names)).to_numpy()
    res["name_unseen"] = {"share": float(unseen.mean()),
                          **_auc_ci(y[unseen], p[unseen], owners[unseen])}
    write_json(OUT / "phase7" / "sensitivity.json", res)
    print(json.dumps(res, indent=2, default=str))

# ---------------------------------------------------------------------------
# Phase 7: interpretation of the text model, if H1 is confirmed (section 9)
# ---------------------------------------------------------------------------

# Rows of the interpretation sample, its seed, permutations per attribute, bootstrap resamples.
INTERP_ROWS, INTERP_SEED, N_PERM, N_BOOT_PERM = 20_000, "20261123", 10, 200


def interp_sample(hold: pd.DataFrame) -> pd.DataFrame:
    """Sample of the holdout used to interpret the model: about 20,000 rows made of whole
    authors, chosen in the order of ``md5(author + ':' + seed)``. The whole holdout is used
    if it is not larger than that."""
    if len(hold) <= INTERP_ROWS:
        return hold.reset_index(drop=True)
    keys = hold["owner"].astype(str).map(
        lambda o: hashlib.md5(f"{o}:{INTERP_SEED}".encode()).hexdigest())
    order = hold.assign(_k=keys).sort_values(["_k", "content_hash"])
    sizes = order.groupby("_k", sort=False).size()
    # Keep every author whose first skill starts before the row limit is reached.
    keep = sizes.index[(sizes.cumsum() - sizes) < INTERP_ROWS]
    return order[order["_k"].isin(keep)].drop(columns="_k").reset_index(drop=True)


def permutation_table(pipe, cols: list[str], x: pd.DataFrame, y: np.ndarray, owners: np.ndarray,
                      seed: int, n_boot: int, families_only: bool = False) -> pd.DataFrame:
    """Permutation importance: how much the AUC drops when an attribute is shuffled.

    The values of an attribute (or of a whole family of attributes at once, since many are
    correlated) are shuffled among the skills, which breaks their link with the target. The
    larger the drop in AUC, the more the model relies on that attribute. Each shuffle is
    repeated ``N_PERM`` times and averaged. The 95% interval resamples authors, using the
    same resamples for all attributes. Returns one row per family (``family:<name>``) and,
    unless ``families_only``, per attribute, sorted from the largest drop.
    """
    rng = np.random.default_rng(seed)
    base = pipe.predict_proba(x[cols])[:, 1]
    targets = {f"family:{k}": v for k, v in FAMILIES.items()}
    if not families_only:
        targets.update({c: [c] for c in TEXT})
    # Bootstrap weights per skill (the number of times its author is drawn), computed once.
    codes, inv = np.unique(owners, return_inverse=True)
    brng = np.random.default_rng(seed + 1)
    weights = [np.bincount(brng.integers(0, len(codes), len(codes)),
                           minlength=len(codes))[inv].astype(float) for _ in range(n_boot)]
    base_auc = roc_auc_score(y, base)
    base_boot = np.array([roc_auc_score(y, base, sample_weight=w) for w in weights])
    rows = []
    for name, group in targets.items():
        group = [c for c in group if c in cols]
        if not group:
            continue
        drops, boots = [], []
        for _ in range(N_PERM):
            xp = x[cols].copy()
            # The same permutation moves all columns of the group together.
            xp[group] = xp[group].iloc[rng.permutation(len(xp))].to_numpy()
            pp = pipe.predict_proba(xp)[:, 1]
            drops.append(base_auc - roc_auc_score(y, pp))
            boots.append(base_boot - np.array([roc_auc_score(y, pp, sample_weight=w)
                                               for w in weights]))
        b = np.mean(boots, axis=0)
        rows.append({"feature": name, "auc_drop": float(np.mean(drops)),
                     "ci95_low": float(np.percentile(b, 2.5)),
                     "ci95_high": float(np.percentile(b, 97.5))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def run_interpret() -> None:
    """Interpret the text model (command ``skillreuse interpret``), answering RQ2.

    Only runs if H1 was confirmed; otherwise it records that RQ2 was not answered. Computes
    the permutation importance of families and attributes on the interpretation sample of
    the holdout (``permutation_holdout.csv``), then checks that the ranking of the families
    is stable by repeating the computation on the validation part of each of the 9
    development folds (``family_stability.csv``). The SHAP values are computed in
    ``report.py``. The status is written to ``interpretation_status.json``; its "rq2"
    values are in Portuguese because they are stored in the file.
    """
    out = OUT / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "phase6" / "holdout_results.json").read_text())
    if not res["H1"]["success"]:
        write_json(out / "interpretation_status.json", {"rq2": "nao respondida"})
        return
    fz = frozen()
    m = joblib.load(fz["models"]["text"]["file"])
    pipe, cols = m["pipeline"], m["columns"]
    hold = interp_sample(load_frame("holdout"))
    y, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    perm = permutation_table(pipe, cols, hold, y, owners, SEED, N_BOOT_PERM)
    perm.to_csv(out / "permutation_holdout.csv", index=False)
    top5 = [f for f in perm["feature"] if not f.startswith("family:")][:5]
    dev, rules = load_frame("dev"), load_rules()
    fp = json.loads((OUT / "phase5" / "final_params.json").read_text())
    yd, ranks = dev["replicated"].to_numpy(), []
    for r, f, tr, va in folds(dev):
        p, c = pipeline(fp["algo"], "text", rules, fp["views"]["text"])
        p.fit(dev.iloc[tr][c], yd[tr])
        t = permutation_table(p, c, dev.iloc[va].reset_index(drop=True), yd[va],
                              dev.iloc[va]["owner"].astype(str).to_numpy(), SEED + 10 * r + f,
                              n_boot=1, families_only=True)
        ranks.append(t.set_index("feature")["auc_drop"].rename(f"r{r}f{f}"))
        print(f"[interp] stability r{r} f{f}", flush=True)
    fam = pd.concat(ranks, axis=1)
    fam["holdout"] = perm.set_index("feature").loc[fam.index, "auc_drop"]
    fam.to_csv(out / "family_stability.csv")
    write_json(out / "interpretation_status.json", {
        "rq2": "respondida", "rows": len(hold), "authors": int(len(set(owners))),
        "top5": top5, "family_rank_spearman_vs_holdout":
        fam.rank().corr("spearman")["holdout"].drop("holdout").to_dict()})

# ---------------------------------------------------------------------------
# Reproduction and command line
# ---------------------------------------------------------------------------


def run_reproduce(out: Path = OUT / "reproduce") -> dict[str, object]:
    """Quick check (command ``skillreuse reproduce``): apply the frozen models to the
    holdout and compare each AUC with the one saved by the single evaluation (tolerance
    1e-9). This is what ``make verify`` runs first; no model is trained."""
    fz = frozen()
    saved = json.loads((OUT / "phase6" / "holdout_results.json").read_text())
    hold = load_frame("holdout")
    y = hold["replicated"].to_numpy()
    res: dict[str, object] = {"rows": len(hold), "point": {}, "matches": {}}
    for view in fz["models"]:
        m = clf_metrics(y, predict(fz, view, hold))
        res["point"][view] = m
        res["matches"][view] = abs(m["roc_auc"] - saved["point"][view]["roc_auc"]) < 1e-9
    write_json(out / "holdout_check.json", res)
    print(json.dumps(res["matches"]))
    return res


# One ``main_*`` function per command of ``cli.py``.


def main_eda() -> None:
    run_eda()


def main_spot() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--output-dir", type=Path, default=OUT / "phase4")
    a = p.parse_args()
    run_spot(a.repeats, a.output_dir)


def main_tune() -> None:
    run_tune()


def main_final() -> None:
    run_final()


def main_freeze() -> None:
    run_freeze()


def main_holdout() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--allow-reevaluation", action="store_true")
    run_holdout(p.parse_args().allow_reevaluation)


def main_sensitivity() -> None:
    run_sensitivity()


def main_interpret() -> None:
    run_interpret()


def main_reproduce() -> None:
    run_reproduce()
