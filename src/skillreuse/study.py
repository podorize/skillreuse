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
from skillreuse.config import CONFIG  # noqa: E402
from skillreuse.stats import N_BOOT, boot_p, holm, nb_test, owner_bootstrap  # noqa: E402

# Some models do not always converge in the iterations given; this is not an error here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = Path(CONFIG["paths"]["results"])  # one folder per phase: results/phase3, ...
MODELS_DIR = Path(CONFIG["paths"]["models"])  # frozen models (phase 5)
SEED = CONFIG["models"]["seed"]  # seed of the models, the searches and the bootstrap
REPEATS = CONFIG["models"]["repeats"]  # repetitions of the cross-validation
FOLDS = CONFIG["models"]["folds"]  # folds of each repetition
WINDOWS = CONFIG["models"]["windows"]  # candidate windows W, in days

# ---------------------------------------------------------------------------
# Attributes (section 3; the lists are in config/config.yaml, "attributes")
# ---------------------------------------------------------------------------

# The 27 attributes of the text, by family. The families are also used to measure the
# importance of a whole group of attributes (phase 7).
FAMILIES = CONFIG["attributes"]["families"]
TEXT = [c for fam in FAMILIES.values() for c in fam]
# The 14 attributes of the origin repository, used only by the control model.
ORIGIN = CONFIG["attributes"]["origin"]
CATEGORICAL = set(CONFIG["attributes"]["categorical"])
# Deviation of section 13: stars and forks at the snapshot replace the stars at T0.
POPULARITY = CONFIG["attributes"]["popularity"]
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
    the robustness variants. Two columns are computed here (section 2.3):

    * ``eligible``: T0 is at least ``w`` days before the snapshot, so the whole window was
      observed;
    * ``replicated``: the target, 1 if the first copy by another developer arrived within
      ``w`` days of T0, and 0 otherwise (also when no other developer copied the skill).

    Only eligible skills are returned, unless ``eligible_only`` is false.
    """
    frame = pd.read_parquet(TABLE)
    if POPULARITY_TABLE.is_file():
        frame = frame.merge(pd.read_parquet(POPULARITY_TABLE), on="content_hash", how="left")
    if variant:
        frame = frame.merge(pd.read_parquet(VARIANTS_TABLE), on="content_hash", how="left")
    if split == "all":
        frame = frame.copy()
    else:
        frame = frame[frame["split"] == split].copy()

    t0 = pd.to_datetime(frame[f"t0{variant}"], utc=True)
    days_to_first_copy = frame[f"first_copy_lag_days{variant}"]
    if w is None and eligible_only:
        w = window()
    if w is not None:
        frame["eligible"] = t0.notna() & (t0 <= SNAPSHOT - pd.Timedelta(days=w))
        frame["replicated"] = (days_to_first_copy <= w).fillna(False).astype(int)
        if eligible_only:
            frame = frame[frame["eligible"]]
    frame["t0_used"] = t0
    return frame.sort_values("content_hash").reset_index(drop=True)


def folds(frame: pd.DataFrame, repeats: int = REPEATS, n_folds: int = FOLDS, seed: int = SEED):
    """Cross-validation partitions, as ``(repetition, fold, train_idx, valid_idx)``.

    Each repetition shuffles the authors with its own seed (seed, seed + 1, ...). The folds
    are grouped by author, so an author is never in training and validation at once, and
    stratified by the target, so each fold has about the same share of copied skills. All
    algorithms are trained on these same partitions, so their results can be compared.
    """
    authors = frame["owner"].astype(str).to_numpy()
    y = frame["replicated"].to_numpy()
    for repetition in range(1, repeats + 1):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed + repetition - 1)
        partitions = splitter.split(frame, y=y, groups=authors)
        for fold, (train_idx, valid_idx) in enumerate(partitions, 1):
            if set(authors[train_idx]) & set(authors[valid_idx]):
                raise RuntimeError("an author is shared between training and validation")
            yield repetition, fold, train_idx, valid_idx


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

    Among the development skills observed for at least 90 days that were copied within
    90 days, W is the smallest of 30, 60 and 90 days that covers at least 80% of their
    times to first copy. Recent skills are left out because they have not had time to show
    a late first copy.
    """
    t0 = pd.to_datetime(dev["t0"], utc=True)
    observed_90_days = dev[t0 <= SNAPSHOT - pd.Timedelta(days=90)]
    lags = observed_90_days["first_copy_lag_days"]
    lags = lags[lags <= 90]
    coverage = {w: float((lags <= w).mean()) for w in WINDOWS}
    chosen = next((w for w in WINDOWS if coverage[w] >= 0.80), 90)
    # the text of "rule" is stored in window.json, so it is kept as it was written
    return {"W": chosen, "coverage": coverage, "skills_with_90d": len(observed_90_days),
            "copied_within_90d": len(lags),
            "lag_quantiles": lags.quantile([.25, .5, .75, .8, .9]).round(1).to_dict(),
            "rule": "menor W em {30,60,90} com >= 80% dos tempos"}


def eda_rules(dev: pd.DataFrame) -> dict[str, object]:
    """Derive the preprocessing rules from the development data (section 5).

    * ``constant``: columns with a single value, which carry no information;
    * ``skewed_log1p_nontree``: numeric columns with no negative values, more than two
      values and skewness above 2; they get a log transform in models that are sensitive
      to scale, so that a few extreme skills do not dominate them;
    * ``collinear_pairs``: pairs of columns with absolute Spearman correlation above 0.95;
    * ``drop_collinear_nontree``: columns removed from scale-sensitive models. Going through
      the columns in order, a column is kept unless it correlates above 0.95 with a column
      already kept.

    Tree models ignore the last two rules: they do not mind skewness or correlation.
    """
    columns = TEXT + ORIGIN
    numeric = [c for c in columns if c not in CATEGORICAL]
    constant = [c for c in columns if dev[c].nunique(dropna=True) <= 1]
    candidates = [c for c in numeric if c not in constant]
    skewed = [c for c in candidates if dev[c].dropna().min() >= 0 and dev[c].nunique() > 2
              and dev[c].dropna().astype(float).skew() > 2]

    correlation = dev[candidates].astype(float).corr("spearman").abs()
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if correlation.loc[a, b] > 0.95:
                pairs.append({"a": a, "b": b, "abs_spearman": float(correlation.loc[a, b])})
    kept, dropped = [], []
    for c in candidates:
        if any(correlation.loc[c, k] > 0.95 for k in kept):
            dropped.append(c)
        else:
            kept.append(c)
    return {"constant": constant, "skewed_log1p_nontree": skewed, "collinear_pairs": pairs,
            "drop_collinear_nontree": dropped,
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
    # a negative age means T0 is before the creation of the repository: impossible, so missing
    dev.loc[dev["repo_age_days_at_t0"] < 0, "repo_age_days_at_t0"] = np.nan
    write_json(out / "eda_rules.json", eda_rules(dev))

    # outcome of the dating of the development copies outside aggregators
    occ = pd.read_parquet(OCCURRENCES,
                          columns=["content_hash", "repo_full_name", "path", "aggregator"])
    split = pd.read_parquet(TABLE, columns=["content_hash", "split"])
    copies = pd.read_parquet(DATES).merge(occ, on=["content_hash", "repo_full_name", "path"])
    copies = copies.merge(split, on="content_hash")
    copies = copies[(copies["split"] == "dev") & ~copies["aggregator"].astype(bool)]
    status_shares = copies["status"].value_counts(normalize=True).to_dict()

    # profile of each attribute: missing values, distribution and skewness
    profile = []
    for c in TEXT + ORIGIN:
        column = dev[c]
        row = {"feature": c, "missing_share": float(column.isna().mean()),
               "n_unique": int(column.nunique())}
        if c not in CATEGORICAL:
            values = column.dropna().astype(float)
            row.update({"min": float(values.min()), "median": float(values.median()),
                        "p99": float(values.quantile(.99)), "max": float(values.max()),
                        "skew": float(values.skew())})
        profile.append(row)
    pd.DataFrame(profile).to_csv(out / "feature_profile.csv", index=False)

    # share of copied skills in each quartile of each text attribute; ranking with "first"
    # breaks ties by order, so zero-heavy attributes are split arbitrarily between quartiles
    rates = []
    for c in [c for c in TEXT if dev[c].nunique() > 4]:
        quartile = pd.qcut(dev[c].rank(method="first"), 4, labels=False)
        by_quartile = dev.groupby(quartile)["replicated"].mean()
        rates.append({"feature": c, **{f"q{int(q) + 1}": float(v) for q, v in by_quartile.items()}})
    pd.DataFrame(rates).to_csv(out / "univariate_rates.csv", index=False)

    t0_by_month = dev["t0_used"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    summary = {"dev_eligible_rows": len(dev), "dev_authors": int(dev["owner"].nunique()),
               "dev_all_rows": len(dev_all), "replicated_rate": float(dev["replicated"].mean()),
               "collection_status_dev": status_shares, "t0_by_month": t0_by_month,
               "all_copies_dated_share": float(dev["all_copies_dated"].mean())}
    write_json(out / "eda_summary.json", summary)
    print(json.dumps({"window": win, "dev_eligible_rows": summary["dev_eligible_rows"],
                      "replicated_rate": summary["replicated_rate"]}, indent=2))


# ---------------------------------------------------------------------------
# Preprocessing and algorithms (sections 5 and 6)
# ---------------------------------------------------------------------------

# Tree models need no scaling, no log and no removal of correlated columns.
TREE_ALGOS = set(CONFIG["models"]["tree_algorithms"])
ALGOS = CONFIG["models"]["algorithms"]


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
    if algo == "logistic":
        return LogisticRegression(max_iter=2000)
    if algo == "naive_bayes":
        return GaussianNB()
    if algo == "knn":
        return KNeighborsClassifier(n_jobs=threads)
    if algo == "decision_tree":
        return DecisionTreeClassifier(random_state=SEED)
    if algo == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=threads)
    if algo == "hist_gb":
        return HistGradientBoostingClassifier(random_state=SEED)
    if algo == "mlp":
        return MLPClassifier(random_state=SEED)
    raise KeyError(algo)


def search_space(values: object) -> object:
    """One search space of config/config.yaml: a list of values to choose from, or a
    distribution written as {loguniform: [low, high]} or {randint: [low, high]}."""
    if isinstance(values, dict):
        (name, (low, high)), = values.items()
        return {"loguniform": loguniform, "randint": randint}[name](low, high)
    # a list inside the list is a tuple, such as the layer sizes (64, 32) of the MLP
    return [tuple(v) if isinstance(v, list) else v for v in values]


# Hyperparameters searched in phase 4 (10 random configurations per algorithm).
# ``loguniform`` draws on a log scale, ``randint`` an integer, a list one of its values.
SPACES: dict[str, dict[str, object]] = {
    algo: {name: search_space(values) for name, values in space.items()}
    for algo, space in CONFIG["models"]["search_spaces"].items()
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


def validation_ratio(rows: pd.DataFrame) -> float:
    """Mean ratio (validation size / training size) of the folds, used by ``nb_test``."""
    return float((rows["validation_rows"] / rows["train_rows"]).mean())


def compare_to_best(folds_df: pd.DataFrame, key: str = "config") -> pd.DataFrame:
    """Compare the configuration with the highest mean AUC with each of the others.

    Uses the paired Nadeau-Bengio test on the AUC of each fold, and adjusts the p-values
    for the number of comparisons with Holm's method.
    """
    auc = folds_df.pivot_table(index=["repeat", "fold"], columns=key, values="roc_auc")
    best = auc.mean().idxmax()
    ratio = validation_ratio(folds_df)
    rows = [{"best": best, "other": other, **nb_test((auc[best] - auc[other]).to_numpy(), ratio)}
            for other in auc.columns if other != best]
    tests = pd.DataFrame(rows)
    if len(tests):
        tests["p_holm"] = holm(tests["p_nb"].tolist())
    return tests


# ---------------------------------------------------------------------------
# Phase 4: spot-check and nested search (sections 6.1 and 6.2)
# ---------------------------------------------------------------------------


def run_spot(repeats: int = REPEATS, out: Path = OUT / "phase4") -> None:
    """Spot-check (command ``skillreuse spot``): the 7 algorithms with default settings.

    Each algorithm is trained and validated on the same 9 partitions of the development
    set, with the text attributes. The 3 with the highest mean AUC go on to the nested
    search. Writes ``spot_folds.csv``, ``spot_summary.csv``, ``spot_tests.csv`` and
    ``selection.json``.
    """
    out.mkdir(parents=True, exist_ok=True)
    dev, rules = load_frame("dev"), load_rules()
    y = dev["replicated"].to_numpy()
    rows = []
    for repetition, fold, train_idx, valid_idx in folds(dev, repeats=repeats):
        for algo in ALGOS:
            start = time.monotonic()
            model, cols = pipeline(algo, "text", rules)
            model.fit(dev.iloc[train_idx][cols], y[train_idx])
            p = model.predict_proba(dev.iloc[valid_idx][cols])[:, 1]
            rows.append({"repeat": repetition, "fold": fold, "config": algo,
                         "train_rows": len(train_idx), "validation_rows": len(valid_idx),
                         "seconds": time.monotonic() - start, **clf_metrics(y[valid_idx], p)})
        print(f"[spot] r{repetition} f{fold}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "spot_folds.csv", index=False)
    metrics = ["roc_auc", "pr_auc", "brier", "seconds"]
    summary = results.groupby("config")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(out / "spot_summary.csv")
    compare_to_best(results).to_csv(out / "spot_tests.csv", index=False)
    write_json(out / "selection.json", {"ranking": summary.index.tolist(),
                                        "selected_top3": summary.index[:3].tolist()})
    print(summary.round(4).to_string())


N_ITER = CONFIG["models"]["search_iterations"]  # configurations tried
INNER_FOLDS = CONFIG["models"]["inner_folds"]
N_JOBS = CONFIG["models"]["parallel_jobs"]


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

    In each of 6 outer folds (the first 2 repetitions x 3 folds), the search chooses its
    best configuration using only the training part, and that configuration is measured on
    the outer fold, which the search never saw. The default version of each algorithm is
    measured on the same folds. The winner follows the rule of the protocol: the highest
    mean AUC, unless it is not significantly better than the others (Nadeau-Bengio with
    Holm); among the tied configurations, the one that trains fastest. Writes
    ``nested_folds.csv``, ``nested_summary.csv``, ``nested_tests.csv`` and ``winner.json``.
    """
    out = OUT / "phase4"
    dev, rules = load_frame("dev"), load_rules()
    top3 = json.loads((out / "selection.json").read_text())["selected_top3"]
    y = dev["replicated"].to_numpy()
    rows = []
    for repetition, fold, train_idx, valid_idx in folds(dev, repeats=2):
        train, valid = dev.iloc[train_idx], dev.iloc[valid_idx]
        fold_info = {"repeat": repetition, "fold": fold, "train_rows": len(train_idx),
                     "validation_rows": len(valid_idx)}
        for algo in top3:
            default_model, cols = pipeline(algo, "text", rules)
            start = time.monotonic()
            default_model.fit(train[cols], y[train_idx])
            seconds = time.monotonic() - start
            p_default = default_model.predict_proba(valid[cols])[:, 1]
            rows.append({**fold_info, "config": f"{algo}|default", "fit_seconds": seconds,
                         **clf_metrics(y[valid_idx], p_default)})

            tuned = search(algo, "text", train, rules, seed=SEED + repetition - 1)
            p_tuned = tuned.best_estimator_.predict_proba(valid[cols])[:, 1]
            best_params = {k.removeprefix("model__"): v for k, v in tuned.best_params_.items()}
            rows.append({**fold_info, "config": f"{algo}|tuned", "fit_seconds": tuned.refit_time_,
                         **clf_metrics(y[valid_idx], p_tuned),
                         "inner_auc": float(tuned.best_score_),
                         "best_params": json.dumps(best_params, default=str)})
            print(f"[tune] r{repetition} f{fold} {algo}: default {rows[-2]['roc_auc']:.4f} "
                  f"tuned {rows[-1]['roc_auc']:.4f}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "nested_folds.csv", index=False)
    summary = results.groupby("config").agg(mean=("roc_auc", "mean"), std=("roc_auc", "std"),
                                            fit_seconds=("fit_seconds", "mean"))
    summary = summary.sort_values("mean", ascending=False)
    summary.to_csv(out / "nested_summary.csv")
    tests = compare_to_best(results)
    tests.to_csv(out / "nested_tests.csv", index=False)

    # tied: the best configuration and every one that is not significantly worse
    best = summary.index[0]
    tied = [best] + tests.loc[tests["p_holm"] >= 0.05, "other"].tolist()
    winner = summary.loc[tied, "fit_seconds"].idxmin()  # the fastest of the tied ones
    algo, variant = winner.split("|")
    # the text of "rule" is stored in winner.json, so it is kept as it was written
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

    The winning algorithm is trained with four sets of columns: ``text`` (the model of the
    research question), ``origin`` (repository only), ``origin+text`` and ``size`` (only
    the size of the text). Each is validated on the 9 partitions. Writes the parameters,
    the results per fold and per view, the out-of-fold probabilities of the text model
    (used to choose the threshold) and the tests of H1, C1 and text against size.
    """
    out = OUT / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    dev, rules = load_frame("dev"), load_rules()
    winner = json.loads((OUT / "phase4" / "winner.json").read_text())
    algo, y = winner["algo"], dev["replicated"].to_numpy()
    params = {view: final_params(view, dev, rules, winner) for view in MAIN_VIEWS}
    params["size"] = params["text"]  # the reference uses the hyperparameters of the text model
    write_json(out / "final_params.json", {"algo": algo, "variant": winner["variant"],
                                           "views": params})
    rows, out_of_fold = [], []
    for repetition, fold, train_idx, valid_idx in folds(dev):
        for view in [*MAIN_VIEWS, "size"]:
            model, cols = pipeline(algo, view, rules, params[view])
            model.fit(dev.iloc[train_idx][cols], y[train_idx])
            p = model.predict_proba(dev.iloc[valid_idx][cols])[:, 1]
            rows.append({"repeat": repetition, "fold": fold, "config": view,
                         "train_rows": len(train_idx), "validation_rows": len(valid_idx),
                         **clf_metrics(y[valid_idx], p)})
            if view == "text":
                out_of_fold.append(pd.DataFrame({"repeat": repetition, "fold": fold,
                                                 "y": y[valid_idx], "prob": p}))
        print(f"[final] r{repetition} f{fold}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "dev_views_folds.csv", index=False)
    pd.concat(out_of_fold).to_csv(out / "dev_oof_text.csv", index=False)
    summary = results.groupby("config")[["roc_auc", "pr_auc", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "dev_views_summary.csv")
    auc = results.pivot_table(index=["repeat", "fold"], columns="config", values="roc_auc")
    ratio = validation_ratio(results)
    write_json(out / "dev_tests.json", {
        "H1_text_vs_0.5": nb_test((auc["text"] - 0.5).to_numpy(), ratio),
        "C1_origin+text_vs_origin": nb_test((auc["origin+text"] - auc["origin"]).to_numpy(), ratio),
        "text_vs_size": nb_test((auc["text"] - auc["size"]).to_numpy(), ratio)})
    print(summary.round(4).to_string())


def run_freeze() -> None:
    """Freeze the models and the threshold (command ``skillreuse freeze``), before the holdout.

    Each model is trained on the whole development set and saved in ``models/`` with its
    list of columns. The decision threshold is the value between 0.01 and 0.99 that
    maximizes the balanced accuracy of the out-of-fold probabilities of the text model.
    ``freeze.json`` records a SHA-256 of every model and input, with ``holdout_read`` set
    to false.
    """
    dev, rules = load_frame("dev"), load_rules()
    final = json.loads((OUT / "phase5" / "final_params.json").read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y, entries = dev["replicated"].to_numpy(), {}
    for view in [*MAIN_VIEWS, "size"]:
        model, cols = pipeline(final["algo"], view, rules, final["views"][view])
        model.fit(dev[cols], y)
        path = MODELS_DIR / f"{view.replace('+', '_')}.joblib"
        joblib.dump({"pipeline": model, "columns": cols, "view": view}, path, compress=3)
        entries[view] = {"file": str(path), "sha256": sha256(path), "params": final["views"][view]}

    out_of_fold = pd.read_csv(OUT / "phase5" / "dev_oof_text.csv")
    y_oof, p_oof = out_of_fold["y"].to_numpy(), out_of_fold["prob"].to_numpy()
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    threshold = float(grid[np.argmax([balanced_accuracy(y_oof, p_oof, t) for t in grid])])

    inputs = [TABLE, OUT / "phase3" / "window.json", OUT / "phase3" / "eda_rules.json",
              OUT / "phase4" / "winner.json", OUT / "phase5" / "final_params.json",
              Path("paper/protocol.md")]
    write_json(OUT / "phase5" / "freeze.json", {
        "frozen_at_local": datetime.now().astimezone().isoformat(), "algo": final["algo"],
        "models": entries, "threshold": threshold,
        "threshold_dev": threshold_report(y_oof, p_oof, threshold),
        "inputs_sha256": {str(p): sha256(p) for p in inputs}, "holdout_read": False})
    print("frozen; threshold", threshold)


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


def predict(frozen_info: dict, view: str, frame: pd.DataFrame) -> np.ndarray:
    """Probability of being copied for each row of ``frame``, from the frozen model of a view."""
    saved = joblib.load(frozen_info["models"][view]["file"])
    return saved["pipeline"].predict_proba(frame[saved["columns"]])[:, 1]


def auc_bootstrap(y: np.ndarray, p: np.ndarray, owners: np.ndarray, n_boot: int = N_BOOT
                  ) -> np.ndarray:
    """AUC of the probabilities ``p`` in each bootstrap resample of authors."""
    return owner_bootstrap(owners, lambda weights: roc_auc_score(y, p, sample_weight=weights),
                           n_boot=n_boot)


def run_holdout(allow_reevaluation: bool = False) -> None:
    """The single evaluation (command ``skillreuse holdout``): the only look at the holdout.

    Applies the frozen models to the holdout, computes the metrics and 95% intervals from
    2,000 bootstrap resamples of authors, and tests H1 (the AUC of the text model is above
    0.5) and C1 (adding the text to the origin repository raises the AUC), with Holm's
    correction for the two. A lock file then stops the command from running again unless
    ``--allow-reevaluation`` is given (``make rerun`` uses it, to reproduce the numbers).
    """
    out = OUT / "phase6"
    lock = out / "evaluation_lock.json"
    if lock.exists() and not allow_reevaluation:
        raise SystemExit(f"Holdout already evaluated ({lock}). "
                         "Re-evaluating requires a declaration.")
    frozen_info = frozen()
    hold = load_frame("holdout")
    y, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    probs = {view: predict(frozen_info, view, hold) for view in frozen_info["models"]}
    point = {view: clf_metrics(y, p) for view, p in probs.items()}
    boot = {view: auc_bootstrap(y, p, owners) for view, p in probs.items()}

    gain = boot["origin+text"] - boot["origin"]
    h1 = {"auc": point["text"]["roc_auc"], "ci95": ci(boot["text"]),
          "p_boot": boot_p(boot["text"], 0.5)}
    c1 = {"diff": point["origin+text"]["roc_auc"] - point["origin"]["roc_auc"],
          "ci95": ci(gain), "p_boot": boot_p(gain, 0.0)}
    h1["p_holm"], c1["p_holm"] = holm([h1["p_boot"], c1["p_boot"]])
    h1["success"] = bool(h1["ci95"][0] > 0.5 and h1["p_holm"] < 0.05)
    c1["success"] = bool(c1["ci95"][0] > 0 and c1["p_holm"] < 0.05)
    size = {"diff": point["text"]["roc_auc"] - point["size"]["roc_auc"],
            "ci95": ci(boot["text"] - boot["size"])}

    result = {"evaluated_at_local": datetime.now().astimezone().isoformat(), "window": window(),
              "holdout_rows": len(hold), "holdout_authors": int(len(set(owners))),
              "prevalence": float(y.mean()), "point": point,
              "view_ci95": {view: ci(b) for view, b in boot.items()}, "H1": h1, "C1": c1,
              "text_minus_size": size,
              "threshold": threshold_report(y, probs["text"], frozen_info["threshold"]),
              "n_boot": N_BOOT}
    write_json(out / "holdout_results.json", result)
    predictions = {f"prob_{view.replace('+', '_')}": p for view, p in probs.items()}
    pd.DataFrame({"content_hash": hold["content_hash"], "y": y, **predictions}).to_parquet(
        out / "holdout_predictions.parquet", index=False)
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
    available. The control is trained again with the stars and forks of the origin
    repository at the snapshot, which favors the repository, and only C1 is evaluated
    again on the holdout; the text model and H1 do not change (deviation of section 13).
    """
    dev = load_frame("dev")
    base = eda_rules(dev.assign(**{c: dev[c] for c in POPULARITY}))
    extra_rules = {rule: [c for c in base[rule] if c in POPULARITY]
                   for rule in ("constant", "skewed_log1p_nontree", "drop_collinear_nontree")}
    write_json(OUT / "phase5" / "control_rules.json", extra_rules)
    rules = load_rules()
    final = json.loads((OUT / "phase5" / "final_params.json").read_text())
    winner = json.loads((OUT / "phase4" / "winner.json").read_text())
    y, views = dev["replicated"].to_numpy(), ["origin_pop", "origin_pop+text"]
    params = {view: final_params(view, dev, rules, winner) for view in views}

    rows = []
    for repetition, fold, train_idx, valid_idx in folds(dev):
        for view in views:
            model, cols = pipeline(final["algo"], view, rules, params[view])
            model.fit(dev.iloc[train_idx][cols], y[train_idx])
            p = model.predict_proba(dev.iloc[valid_idx][cols])[:, 1]
            rows.append({"repeat": repetition, "fold": fold, "config": view,
                         "train_rows": len(train_idx), "validation_rows": len(valid_idx),
                         **clf_metrics(y[valid_idx], p)})
    results = pd.DataFrame(rows)
    auc = results.pivot_table(index=["repeat", "fold"], columns="config", values="roc_auc")
    dev_test = nb_test((auc["origin_pop+text"] - auc["origin_pop"]).to_numpy(),
                       validation_ratio(results))

    entries = {}
    for view in views:
        model, cols = pipeline(final["algo"], view, rules, params[view])
        model.fit(dev[cols], y)
        path = MODELS_DIR / f"{view.replace('+', '_')}.joblib"
        joblib.dump({"pipeline": model, "columns": cols, "view": view}, path, compress=3)
        entries[view] = {"file": str(path), "sha256": sha256(path), "params": params[view]}
    write_json(OUT / "phase5" / "freeze_control.json", {
        "frozen_at_local": datetime.now().astimezone().isoformat(), "models": entries,
        "dev_auc": results.groupby("config")["roc_auc"].agg(["mean", "std"]).to_dict("index"),
        "dev_C1": dev_test})

    hold = load_frame("holdout")
    y_hold, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    probs = {}
    for view, entry in entries.items():
        saved = joblib.load(entry["file"])
        probs[view] = saved["pipeline"].predict_proba(hold[saved["columns"]])[:, 1]
    p_text = predict(frozen(), "text", hold)
    boot = {view: auc_bootstrap(y_hold, p, owners) for view, p in probs.items()}
    boot_text = auc_bootstrap(y_hold, p_text, owners)
    gain = boot["origin_pop+text"] - boot["origin_pop"]
    c1 = {"diff": float(roc_auc_score(y_hold, probs["origin_pop+text"])
                        - roc_auc_score(y_hold, probs["origin_pop"])),
          "ci95": ci(gain), "p_boot": boot_p(gain, 0.0)}
    # Holm over the same two tests as before: H1 (text) and this C1
    c1["p_holm_with_H1"] = holm([boot_p(boot_text, 0.5), c1["p_boot"]])[1]
    c1["success"] = bool(c1["ci95"][0] > 0 and c1["p_holm_with_H1"] < 0.05)
    # the text of "note" is stored in control_popularity.json, so it is kept as written
    write_json(OUT / "phase6" / "control_popularity.json", {
        "note": "desvio da secao 13: popularidade no retrato (posterior a T0) no controle",
        "evaluated_at_local": datetime.now().astimezone().isoformat(),
        "point": {view: clf_metrics(y_hold, p) for view, p in probs.items()},
        "view_ci95": {view: ci(b) for view, b in boot.items()}, "C1_popularity": c1})
    predictions = {f"prob_{view.replace('+', '_')}": p for view, p in probs.items()}
    with_control = pd.read_parquet(OUT / "phase6" / "holdout_predictions.parquet")
    with_control = with_control.assign(**predictions)
    with_control.to_parquet(OUT / "phase6" / "holdout_predictions_control.parquet", index=False)
    print(json.dumps({"dev": dev_test, "C1_popularity": c1}, indent=2, default=str))


def main_control() -> None:
    run_control()


# ---------------------------------------------------------------------------
# Phase 7: sensitivity and robustness (section 8)
# ---------------------------------------------------------------------------

# bootstrap resamples in the robustness checks (2,000 in the main result)
N_BOOT_SENS = CONFIG["evaluation"]["n_boot_robustness"]


def auc_with_interval(y: np.ndarray, p: np.ndarray, owners: np.ndarray) -> dict[str, object]:
    """AUC with a 95% interval from 500 bootstrap resamples of authors. With a single class
    in ``y`` the AUC does not exist and is returned as ``None``."""
    if len(np.unique(y)) < 2:
        return {"rows": len(y), "prevalence": float(np.mean(y)), "auc": None}
    boot = auc_bootstrap(y, p, owners, n_boot=N_BOOT_SENS)
    return {"rows": len(y), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "ci95": ci(boot)}


def retrain_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, rules: dict) -> dict:
    """Train the text model (same algorithm and hyperparameters as the final one) on
    ``train`` and report its AUC with a 95% interval on ``test``."""
    final = json.loads((OUT / "phase5" / "final_params.json").read_text())
    model, cols = pipeline(final["algo"], "text", rules, final["views"]["text"])
    model.fit(train[cols], train["replicated"].to_numpy())
    p = model.predict_proba(test[cols])[:, 1]
    return {"train_rows": len(train), **auc_with_interval(test["replicated"].to_numpy(), p,
                                                          test["owner"].astype(str).to_numpy())}


def run_sensitivity() -> None:
    """Robustness checks (command ``skillreuse sensitivity``), after the single evaluation.

    Seven checks on the holdout: aggregator limits of 500 skills and of none, windows of
    30 and 90 days, and the temporal split (train on development skills that appeared
    before the 70th percentile of T0, test on holdout skills that appeared after it) retrain
    the text model; only skills whose copies were all dated, and only skills whose name
    never appears in the development set, use the frozen model.
    """
    if not (OUT / "phase6" / "evaluation_lock.json").exists():
        raise SystemExit("The robustness checks come after the single evaluation (phase 6).")
    frozen_info, rules, w = frozen(), load_rules(), window()
    res: dict[str, object] = {"window_main": w}
    for variant in ("_agg500", "_aggnone"):  # needs the aggregator copies (cohort --variants)
        if VARIANTS_TABLE.is_file():
            res[f"aggregators{variant}"] = retrain_and_evaluate(
                load_frame("dev", w, variant), load_frame("holdout", w, variant), rules)
        else:
            res[f"aggregators{variant}"] = "pendente: coleta dos agregadores"
    for other_w in [x for x in WINDOWS if x != w]:
        res[f"window_{other_w}"] = retrain_and_evaluate(
            load_frame("dev", other_w), load_frame("holdout", other_w), rules)

    hold, dev = load_frame("holdout"), load_frame("dev")
    p = predict(frozen_info, "text", hold)
    y, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    all_dated = hold["all_copies_dated"].to_numpy(dtype=bool)
    res["all_copies_dated"] = auc_with_interval(y[all_dated], p[all_dated], owners[all_dated])
    cutoff = dev["t0_used"].quantile(0.70)
    res["temporal"] = {"cutoff": str(cutoff), **retrain_and_evaluate(
        dev[dev["t0_used"] < cutoff], hold[hold["t0_used"] >= cutoff], rules)}
    names_in_dev = set(dev.loc[dev["name_norm"] != "", "name_norm"])
    unseen = ~(hold["name_norm"].ne("") & hold["name_norm"].isin(names_in_dev)).to_numpy()
    res["name_unseen"] = {"share": float(unseen.mean()),
                          **auc_with_interval(y[unseen], p[unseen], owners[unseen])}
    write_json(OUT / "phase7" / "sensitivity.json", res)
    print(json.dumps(res, indent=2, default=str))


# ---------------------------------------------------------------------------
# Phase 7: interpretation of the text model, if H1 is confirmed (section 9)
# ---------------------------------------------------------------------------

# Rows of the interpretation sample, its seed, permutations per attribute, bootstrap resamples.
INTERP_ROWS = CONFIG["interpretation"]["rows"]
INTERP_SEED = CONFIG["interpretation"]["seed"]
N_PERM = CONFIG["interpretation"]["permutations"]
N_BOOT_PERM = CONFIG["interpretation"]["n_boot"]


def interp_sample(hold: pd.DataFrame) -> pd.DataFrame:
    """Sample of the holdout used to interpret the model: about 20,000 rows made of whole
    authors, taken in the order of ``md5(author + ':' + seed)``. The whole holdout is used
    when it is not larger than that."""
    if len(hold) <= INTERP_ROWS:
        return hold.reset_index(drop=True)
    keys = hold["owner"].astype(str).map(
        lambda owner: hashlib.md5(f"{owner}:{INTERP_SEED}".encode()).hexdigest())
    ordered = hold.assign(_k=keys).sort_values(["_k", "content_hash"])
    skills_per_author = ordered.groupby("_k", sort=False).size()
    # keep every author whose first skill comes before the row limit
    keep = skills_per_author.index[(skills_per_author.cumsum() - skills_per_author) < INTERP_ROWS]
    return ordered[ordered["_k"].isin(keep)].drop(columns="_k").reset_index(drop=True)


def permutation_table(model, cols: list[str], x: pd.DataFrame, y: np.ndarray, owners: np.ndarray,
                      seed: int, n_boot: int, families_only: bool = False) -> pd.DataFrame:
    """Permutation importance: how much the AUC drops when an attribute is shuffled.

    The values of an attribute, or of a whole family at once, are shuffled among the skills,
    which breaks their link with the target. Each shuffle is repeated ``N_PERM`` times and
    averaged. The 95% interval comes from resampling authors, with the same resamples for
    every attribute. Returns one row per family (``family:<name>``) and, unless
    ``families_only``, per attribute, from the largest drop to the smallest.
    """
    rng = np.random.default_rng(seed)
    p_base = model.predict_proba(x[cols])[:, 1]
    targets = {f"family:{name}": columns for name, columns in FAMILIES.items()}
    if not families_only:
        targets.update({c: [c] for c in TEXT})

    # bootstrap weights of each row (the times its author is drawn), computed once
    authors, author_of_row = np.unique(owners, return_inverse=True)
    boot_rng = np.random.default_rng(seed + 1)
    weights = [np.bincount(boot_rng.integers(0, len(authors), len(authors)),
                           minlength=len(authors))[author_of_row].astype(float)
               for _ in range(n_boot)]
    auc_base = roc_auc_score(y, p_base)
    auc_base_boot = np.array([roc_auc_score(y, p_base, sample_weight=w) for w in weights])

    rows = []
    for name, group in targets.items():
        group = [c for c in group if c in cols]
        if not group:
            continue
        drops, boot_drops = [], []
        for _ in range(N_PERM):
            shuffled = x[cols].copy()
            # the same permutation moves all columns of the group together
            shuffled[group] = shuffled[group].iloc[rng.permutation(len(shuffled))].to_numpy()
            p = model.predict_proba(shuffled)[:, 1]
            drops.append(auc_base - roc_auc_score(y, p))
            boot_drops.append(auc_base_boot - np.array([roc_auc_score(y, p, sample_weight=w)
                                                        for w in weights]))
        mean_boot_drop = np.mean(boot_drops, axis=0)
        rows.append({"feature": name, "auc_drop": float(np.mean(drops)),
                     "ci95_low": float(np.percentile(mean_boot_drop, 2.5)),
                     "ci95_high": float(np.percentile(mean_boot_drop, 97.5))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def run_interpret() -> None:
    """Interpret the text model (command ``skillreuse interpret``), answering RQ2.

    Runs only if H1 was confirmed. Computes the permutation importance of families and
    attributes on the interpretation sample of the holdout, and checks that the ranking of
    the families is stable by repeating it on the validation part of each of the 9
    development folds. The SHAP values are computed in ``report.py``. The "rq2" values of
    ``interpretation_status.json`` are stored in Portuguese.
    """
    out = OUT / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    holdout_results = json.loads((OUT / "phase6" / "holdout_results.json").read_text())
    if not holdout_results["H1"]["success"]:
        write_json(out / "interpretation_status.json", {"rq2": "nao respondida"})
        return
    saved = joblib.load(frozen()["models"]["text"]["file"])
    model, cols = saved["pipeline"], saved["columns"]
    hold = interp_sample(load_frame("holdout"))
    y, owners = hold["replicated"].to_numpy(), hold["owner"].astype(str).to_numpy()
    importance = permutation_table(model, cols, hold, y, owners, SEED, N_BOOT_PERM)
    importance.to_csv(out / "permutation_holdout.csv", index=False)
    top5 = [f for f in importance["feature"] if not f.startswith("family:")][:5]

    dev, rules = load_frame("dev"), load_rules()
    final = json.loads((OUT / "phase5" / "final_params.json").read_text())
    y_dev, by_fold = dev["replicated"].to_numpy(), []
    for repetition, fold, train_idx, valid_idx in folds(dev):
        fold_model, fold_cols = pipeline(final["algo"], "text", rules, final["views"]["text"])
        fold_model.fit(dev.iloc[train_idx][fold_cols], y_dev[train_idx])
        valid = dev.iloc[valid_idx].reset_index(drop=True)
        drops = permutation_table(fold_model, fold_cols, valid, y_dev[valid_idx],
                                  valid["owner"].astype(str).to_numpy(),
                                  SEED + 10 * repetition + fold, n_boot=1, families_only=True)
        by_fold.append(drops.set_index("feature")["auc_drop"].rename(f"r{repetition}f{fold}"))
        print(f"[interp] stability r{repetition} f{fold}", flush=True)
    stability = pd.concat(by_fold, axis=1)
    stability["holdout"] = importance.set_index("feature").loc[stability.index, "auc_drop"]
    stability.to_csv(out / "family_stability.csv")
    write_json(out / "interpretation_status.json", {
        "rq2": "respondida", "rows": len(hold), "authors": int(len(set(owners))),
        "top5": top5, "family_rank_spearman_vs_holdout":
        stability.rank().corr("spearman")["holdout"].drop("holdout").to_dict()})


# ---------------------------------------------------------------------------
# Reproduction and command line
# ---------------------------------------------------------------------------


def run_reproduce(out: Path = OUT / "reproduce") -> dict[str, object]:
    """Quick check (command ``skillreuse reproduce``): apply the frozen models to the
    holdout and compare each AUC with the one saved by the single evaluation (tolerance
    1e-9). This is what ``make verify`` runs first; no model is trained."""
    frozen_info = frozen()
    saved = json.loads((OUT / "phase6" / "holdout_results.json").read_text())
    hold = load_frame("holdout")
    y = hold["replicated"].to_numpy()
    res: dict[str, object] = {"rows": len(hold), "point": {}, "matches": {}}
    for view in frozen_info["models"]:
        metrics = clf_metrics(y, predict(frozen_info, view, hold))
        res["point"][view] = metrics
        res["matches"][view] = abs(metrics["roc_auc"] - saved["point"][view]["roc_auc"]) < 1e-9
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
