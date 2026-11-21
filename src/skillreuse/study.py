"""Fases 3 a 7 do protocolo (paper/protocol.md): EDA e janela, selecao de modelos, modelos
finais, avaliacao unica, sensibilidades e interpretacao.

Todos os experimentos de desenvolvimento usam as mesmas particoes: 3 repeticoes x 3 dobras de
``StratifiedGroupKFold`` agrupadas por autor e estratificadas pelo alvo (sementes 2027-2029).
O holdout so e lido em ``skillreuse holdout`` (fase 6) e depois dela (fase 7).
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
import yaml
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

from skillreuse.stats import boot_p, holm, nb_test, owner_bootstrap  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATES = Path(config["dates"])
OCCURRENCES = Path(config["occurrences"])
SNAPSHOT = pd.Timestamp(config["snapshot"], tz="UTC")
TABLE = Path(config["table"])
VARIANTS_TABLE = Path(config["variants_table"])
POPULARITY_TABLE = Path(config["popularity_table"])

N_BOOT = config["n_boot"]

OUT = Path(config["out"])
MODELS_DIR = Path(config["models_dir"])
SEED = config["seed"]
REPEATS = config["repeats"]
FOLDS = config["folds"]
WINDOWS = config["windows"]

# ---------------------------------------------------------------------------
# Atributos (secao 3)
# ---------------------------------------------------------------------------

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
ORIGIN = ["log_stars_at_t0", "repo_age_days_at_t0", "repo_language", "repo_license",
          "repo_is_fork", "path_depth", "path_is_root", "p_template", "p_plugin", "p_vendor",
          "p_docs", "p_pkg", "p_tool_dir", "p_claude_dir"]
CATEGORICAL = {"repo_language", "repo_license"}
# desvio da secao 13: estrelas e forks no retrato substituem as estrelas em T0
POPULARITY = ["log_stars_snapshot", "log_forks_snapshot"]
ORIGIN_POP = [c for c in ORIGIN if c != "log_stars_at_t0"] + POPULARITY
VIEWS = {"text": TEXT, "origin": ORIGIN, "origin+text": ORIGIN + TEXT, "size": FAMILIES["Size"],
         "origin_pop": ORIGIN_POP, "origin_pop+text": ORIGIN_POP + TEXT}
MAIN_VIEWS = ["text", "origin", "origin+text"]
assert len(TEXT) == 27

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------


def window() -> int:
    return int(json.loads((OUT / "phase3" / "window.json").read_text())["W"])


def load_frame(split: str, w: int | None = None, variant: str = "",
               eligible_only: bool = True) -> pd.DataFrame:
    """Skills de um lado da divisao, com alvo e elegibilidade da janela W (secao 2.3).

    ``variant`` escolhe a definicao de agregador ("" = 100 skills, "_agg500", "_aggnone").
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
    groups = frame["owner"].astype(str).to_numpy()
    y = frame["replicated"].to_numpy()
    for r in range(1, repeats + 1):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r - 1)
        for f, (tr, va) in enumerate(splitter.split(frame, y=y, groups=groups), 1):
            if set(groups[tr]) & set(groups[va]):
                raise RuntimeError("autor compartilhado entre treino e validacao")
            yield r, f, tr, va


def clf_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
            "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
            "prevalence": float(np.average(y, weights=w))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
                    + "\n", encoding="utf-8")


def ci(samples: np.ndarray) -> list[float]:
    return np.percentile(samples, [2.5, 97.5]).tolist()

# ---------------------------------------------------------------------------
# Fase 3: EDA, janela e regras de pre-processamento (secao 5)
# ---------------------------------------------------------------------------


def choose_window(dev: pd.DataFrame) -> dict[str, object]:
    """Regra da janela: skills com 90 dias de observacao copiadas em ate 90 dias."""
    t0 = pd.to_datetime(dev["t0"], utc=True)
    obs = dev[t0 <= SNAPSHOT - pd.Timedelta(days=90)]
    lags = obs.loc[obs["first_copy_lag_days"] <= 90, "first_copy_lag_days"]
    cover = {w: float((lags <= w).mean()) for w in WINDOWS}
    w = next((w for w in WINDOWS if cover[w] >= 0.80), 90)
    return {"W": w, "coverage": cover, "skills_with_90d": len(obs),
            "copied_within_90d": len(lags), "lag_quantiles": lags.quantile([.25, .5, .75, .8, .9])
            .round(1).to_dict(), "rule": "menor W em {30,60,90} com >= 80% dos tempos"}


def eda_rules(dev: pd.DataFrame) -> dict[str, object]:
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
    out = OUT / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    dev_all = load_frame("dev", eligible_only=False)
    win = choose_window(dev_all)
    write_json(out / "window.json", win)
    dev = load_frame("dev", w=win["W"])
    dev.loc[dev["repo_age_days_at_t0"] < 0, "repo_age_days_at_t0"] = np.nan
    rules = eda_rules(dev)
    write_json(out / "eda_rules.json", rules)
    occ = pd.read_parquet(OCCURRENCES,
                          columns=["content_hash", "repo_full_name", "path", "aggregator"])
    status = pd.read_parquet(DATES).merge(occ, on=["content_hash", "repo_full_name", "path"]) \
        .merge(pd.read_parquet(TABLE, columns=["content_hash", "split"]), on="content_hash")
    status = status[(status["split"] == "dev") & ~status["aggregator"].astype(bool)]
    status = status["status"].value_counts(normalize=True).to_dict()
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
# Pre-processamento e algoritmos (secoes 5 e 6)
# ---------------------------------------------------------------------------

TREE_ALGOS = set(config["tree_algos"])
ALGOS = config["algos"]

def _log1p_clip(x: object) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def _dense(x: object) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def load_rules() -> dict[str, object]:
    rules = json.loads((OUT / "phase3" / "eda_rules.json").read_text())
    extra = OUT / "phase5" / "control_rules.json"  # regras das colunas de popularidade (desvio)
    if extra.is_file():
        for k, v in json.loads(extra.read_text()).items():
            if isinstance(v, list):
                rules[k] = rules.get(k, []) + [x for x in v if x not in rules.get(k, [])]
    return rules


def view_columns(view: str, algo: str, rules: dict[str, object]) -> list[str]:
    cols = [c for c in VIEWS[view] if c not in rules["constant"]]
    if algo not in TREE_ALGOS:
        cols = [c for c in cols if c not in rules["drop_collinear_nontree"]]
    return cols


def preprocessor(columns: list[str], algo: str, rules: dict[str, object]) -> ColumnTransformer:
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
    return {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "naive_bayes": lambda: GaussianNB(),
        "knn": lambda: KNeighborsClassifier(n_jobs=threads),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=threads),
        "hist_gb": lambda: HistGradientBoostingClassifier(random_state=SEED),
        "mlp": lambda: MLPClassifier(random_state=SEED),
    }[algo]()


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
    return float((rows["validation_rows"] / rows["train_rows"]).mean())


def compare_to_best(folds_df: pd.DataFrame, key: str = "config") -> pd.DataFrame:
    """Testes pareados (Nadeau-Bengio) do melhor contra os demais, com Holm."""
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
# Fase 4: spot-checking e busca aninhada (secoes 6.1 e 6.2)
# ---------------------------------------------------------------------------


def run_spot(repeats: int = REPEATS, out: Path = OUT / "phase4") -> None:
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


N_ITER, INNER_FOLDS, N_JOBS = 10, 3, 4


def inner_cv(train: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(train, y=train["replicated"], groups=train["owner"].astype(str)))


def search(algo: str, view: str, train: pd.DataFrame, rules: dict, seed: int
           ) -> RandomizedSearchCV:
    pipe, cols = pipeline(algo, view, rules, threads=1)
    rs = RandomizedSearchCV(pipe, {f"model__{k}": v for k, v in SPACES[algo].items()},
                            n_iter=N_ITER, cv=inner_cv(train, seed), scoring="roc_auc",
                            n_jobs=N_JOBS, random_state=SEED, refit=True, error_score="raise")
    return rs.fit(train[cols], train["replicated"].to_numpy())


def run_tune() -> None:
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
    tied = [best] + tests.loc[tests["p_holm"] >= 0.05, "other"].tolist()
    winner = summary.loc[tied, "fit_seconds"].idxmin()  # desempate da secao 6.2
    algo, variant = winner.split("|")
    write_json(out / "winner.json", {"best_auc": best, "tied": tied, "winner": winner,
                                     "algo": algo, "variant": variant,
                                     "rule": "maior AUC externa; entre empatados, menor tempo"})
    print(summary.round(4).to_string(), "\nvencedor:", winner)

# ---------------------------------------------------------------------------
# Fase 5: modelos finais, referencia, limiar e congelamento (secao 6.3-6.6)
# ---------------------------------------------------------------------------


def final_params(view: str, dev: pd.DataFrame, rules: dict, winner: dict) -> dict:
    if winner["variant"] == "default":
        return {}
    rs = search(winner["algo"], view, dev, rules, seed=SEED)
    return {k.removeprefix("model__"): v for k, v in rs.best_params_.items()}


def balanced_accuracy(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    return float((np.mean(pred[y == 1]) + np.mean(~pred[y == 0])) / 2)


def threshold_report(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = p >= t
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": t, "balanced_accuracy": balanced_accuracy(y, p, t), "precision": prec,
            "recall": rec, "true_positive": tp, "false_positive": fp, "false_negative": fn}


def run_final() -> None:
    out = OUT / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    dev, rules = load_frame("dev"), load_rules()
    winner = json.loads((OUT / "phase4" / "winner.json").read_text())
    algo, y = winner["algo"], dev["replicated"].to_numpy()
    params = {v: final_params(v, dev, rules, winner) for v in MAIN_VIEWS}
    params["size"] = params["text"]  # referencia: mesmo algoritmo e hiperparametros do texto
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
    print("congelado; limiar", thr)

# ---------------------------------------------------------------------------
# Fase 6: avaliacao unica no holdout (secao 7)
# ---------------------------------------------------------------------------


def frozen() -> dict:
    fz = json.loads((OUT / "phase5" / "freeze.json").read_text())
    for view, e in fz["models"].items():
        if sha256(Path(e["file"])) != e["sha256"]:
            raise RuntimeError(f"modelo congelado alterado: {view}")
    return fz


def predict(fz: dict, view: str, frame: pd.DataFrame) -> np.ndarray:
    m = joblib.load(fz["models"][view]["file"])
    return m["pipeline"].predict_proba(frame[m["columns"]])[:, 1]


def run_holdout(allow_reevaluation: bool = False) -> None:
    out = OUT / "phase6"
    lock = out / "evaluation_lock.json"
    if lock.exists() and not allow_reevaluation:
        raise SystemExit(f"Holdout ja avaliado ({lock}). Reavaliar exige declaracao.")
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
# Desvio da secao 13: controle com a popularidade do repositorio de origem no retrato
# ---------------------------------------------------------------------------


def run_control() -> None:
    """Retreina o controle com estrelas e forks no retrato e reavalia so o C1 no holdout.

    O modelo de texto e a H1 nao mudam. Os resultados originais (phase6) sao preservados.
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
    c1["p_holm_with_H1"] = holm([boot_p(b_text, 0.5), c1["p_boot"]])[1]
    c1["success"] = bool(c1["ci95"][0] > 0 and c1["p_holm_with_H1"] < 0.05)
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
# Fase 7: sensibilidade e robustez (secao 8)
# ---------------------------------------------------------------------------

N_BOOT_SENS = 500


def _auc_ci(y: np.ndarray, p: np.ndarray, owners: np.ndarray) -> dict[str, object]:
    if len(np.unique(y)) < 2:
        return {"rows": len(y), "prevalence": float(np.mean(y)), "auc": None}
    b = owner_bootstrap(owners, lambda w: roc_auc_score(y, p, sample_weight=w), n_boot=N_BOOT_SENS)
    return {"rows": len(y), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "ci95": ci(b)}


def _retrain_eval(train: pd.DataFrame, test: pd.DataFrame, fz: dict, rules: dict) -> dict:
    fp = json.loads((OUT / "phase5" / "final_params.json").read_text())
    pipe, cols = pipeline(fp["algo"], "text", rules, fp["views"]["text"])
    pipe.fit(train[cols], train["replicated"].to_numpy())
    p = pipe.predict_proba(test[cols])[:, 1]
    return {"train_rows": len(train), **_auc_ci(test["replicated"].to_numpy(), p,
                                                test["owner"].astype(str).to_numpy())}


def run_sensitivity() -> None:
    if not (OUT / "phase6" / "evaluation_lock.json").exists():
        raise SystemExit("As sensibilidades vem depois da avaliacao unica (fase 6).")
    fz, rules, w = frozen(), load_rules(), window()
    res: dict[str, object] = {"window_main": w}
    for variant in ("_agg500", "_aggnone"):  # exige a coleta dos agregadores (cohort --variants)
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
    c = dev["t0_used"].quantile(0.70)
    res["temporal"] = {"cutoff": str(c), **_retrain_eval(dev[dev["t0_used"] < c],
                                                         hold[hold["t0_used"] >= c], fz, rules)}
    names = set(dev.loc[dev["name_norm"] != "", "name_norm"])
    unseen = ~(hold["name_norm"].ne("") & hold["name_norm"].isin(names)).to_numpy()
    res["name_unseen"] = {"share": float(unseen.mean()),
                          **_auc_ci(y[unseen], p[unseen], owners[unseen])}
    write_json(OUT / "phase7" / "sensitivity.json", res)
    print(json.dumps(res, indent=2, default=str))

# ---------------------------------------------------------------------------
# Fase 7: interpretacao do modelo de texto, se H1 confirmada (secao 9)
# ---------------------------------------------------------------------------

INTERP_ROWS, INTERP_SEED, N_PERM, N_BOOT_PERM = 20_000, "20261123", 10, 200


def interp_sample(hold: pd.DataFrame) -> pd.DataFrame:
    if len(hold) <= INTERP_ROWS:
        return hold.reset_index(drop=True)
    keys = hold["owner"].astype(str).map(
        lambda o: hashlib.md5(f"{o}:{INTERP_SEED}".encode()).hexdigest())
    order = hold.assign(_k=keys).sort_values(["_k", "content_hash"])
    sizes = order.groupby("_k", sort=False).size()
    keep = sizes.index[(sizes.cumsum() - sizes) < INTERP_ROWS]
    return order[order["_k"].isin(keep)].drop(columns="_k").reset_index(drop=True)


def permutation_table(pipe, cols: list[str], x: pd.DataFrame, y: np.ndarray, owners: np.ndarray,
                      seed: int, n_boot: int, families_only: bool = False) -> pd.DataFrame:
    """Queda de AUC ao permutar familias e colunas; IC por reamostragem de autores."""
    rng = np.random.default_rng(seed)
    base = pipe.predict_proba(x[cols])[:, 1]
    targets = {f"family:{k}": v for k, v in FAMILIES.items()}
    if not families_only:
        targets.update({c: [c] for c in TEXT})
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
        print(f"[interp] estabilidade r{r} f{f}", flush=True)
    fam = pd.concat(ranks, axis=1)
    fam["holdout"] = perm.set_index("feature").loc[fam.index, "auc_drop"]
    fam.to_csv(out / "family_stability.csv")
    write_json(out / "interpretation_status.json", {
        "rq2": "respondida", "rows": len(hold), "authors": int(len(set(owners))),
        "top5": top5, "family_rank_spearman_vs_holdout":
        fam.rank().corr("spearman")["holdout"].drop("holdout").to_dict()})

# ---------------------------------------------------------------------------
# Reproducao e CLI
# ---------------------------------------------------------------------------


def run_reproduce(out: Path = OUT / "reproduce") -> dict[str, object]:
    """Reaplica os modelos congelados ao holdout e confere com a avaliacao unica."""
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
