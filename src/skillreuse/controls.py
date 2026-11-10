"""Controles de atalho de coleta: historico, localizacao e texto da description.

Tres testes sobre o conjunto de desenvolvimento (teste externo nao lido), com as mesmas
particoes agrupadas repetidas (seeds ``seed + r``) usadas nos demais experimentos:

1. ``shortcut``: a visao de repositorio sem ``history_fetched``/``commit_count``, o historico
   sozinho e a localizacao sozinha (``location_class``);
2. ``by_location``: as mesmas previsoes avaliadas dentro de cada ``location_class``;
3. ``text``: TF-IDF da ``description`` (front matter) contra o texto completo, com e sem
   a localizacao.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skillreuse.benchmark import (
    _metrics,
    default_regressor,
    paired_test_table,
    subset_frame,
)
from skillreuse.models import (
    DEFAULT_SAMPLE,
    NUMERIC_FEATURES,
    REPOSITORY_CATEGORICAL_FEATURES,
    REPOSITORY_NUMERIC_FEATURES,
    STRUCTURAL_CATEGORICAL_FEATURES,
    STRUCTURAL_NUMERIC_FEATURES,
    load_count_frame,
)

DEFAULT_OUTPUT_DIR = Path("results/controls")
HISTORY = ["history_fetched", "commit_count"]
REPO = REPOSITORY_NUMERIC_FEATURES + REPOSITORY_CATEGORICAL_FEATURES
REPO_NO_HISTORY = [c for c in REPO if c not in HISTORY]
STRUCT_NO_LOCATION = STRUCTURAL_NUMERIC_FEATURES + ["filename"]
STRUCT_ALL = STRUCTURAL_NUMERIC_FEATURES + STRUCTURAL_CATEGORICAL_FEATURES

SHORTCUT_VIEWS: dict[str, list[str]] = {
    "repo_full": REPO,
    "repo_no_history": REPO_NO_HISTORY,
    "history_only": HISTORY,
    "location_only": ["location_class"],
    "repo_no_history+location": REPO_NO_HISTORY + ["location_class"],
    "skill_struct_no_location": STRUCT_NO_LOCATION,
    "skill_struct_full": STRUCT_ALL,
}
MODELS = ("ridge_log", "random_forest_log")


def _preprocessor(columns: list[str]) -> ColumnTransformer:
    """Mesmo tratamento de ``build_preprocessor`` para um subconjunto arbitrario de colunas."""
    numeric = [c for c in columns if c in NUMERIC_FEATURES]
    categorical = [c for c in columns if c not in numeric]
    blocks: list[tuple[str, object, list[str]]] = []
    if numeric:
        blocks.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        blocks.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(blocks, remainder="drop")


def _dense(matrix: object) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _fit_predict_log(
    model_name: str, columns: list[str], train: pd.DataFrame, valid: pd.DataFrame,
    y_log: np.ndarray,
) -> np.ndarray:
    pre = _preprocessor(columns)
    x_train = pre.fit_transform(train[columns])
    x_valid = pre.transform(valid[columns])
    if model_name != "ridge_log":
        x_train, x_valid = _dense(x_train), _dense(x_valid)
    return default_regressor(model_name).fit(x_train, y_log).predict(x_valid)


def _score(y_counts: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    return _metrics(y_counts, np.maximum(np.expm1(log_pred), 0.0))


def _outer_folds(frame: pd.DataFrame, repeats: int, folds: int, seed: int):
    strata = frame["split_stratum"].astype(str).to_numpy()
    groups = frame["representative_repo_id"].astype(str).to_numpy()
    for repeat in range(1, repeats + 1):
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed + repeat - 1
        )
        for fold, (tr, va) in enumerate(splitter.split(frame, y=strata, groups=groups), 1):
            train = frame.iloc[tr].reset_index(drop=True)
            valid = frame.iloc[va].reset_index(drop=True)
            if set(train["representative_repo_id"]) & set(valid["representative_repo_id"]):
                raise RuntimeError(f"Repeticao {repeat}, dobra {fold}: grupos compartilhados")
            yield repeat, fold, train, valid


def _records(
    base: dict[str, object], config: str, valid: pd.DataFrame, log_pred: np.ndarray,
    overall: list, by_loc: list,
) -> None:
    y = valid["count_target"].to_numpy(dtype=float)
    overall.append({**base, "config": config, **_score(y, log_pred)})
    for loc, idx in valid.groupby("location_class").indices.items():
        by_loc.append(
            {**base, "config": config, "location_class": loc, "validation_rows": len(idx),
             **_score(y[idx], log_pred[idx])}
        )


def run_shortcut(
    frame: pd.DataFrame, *, repeats: int, folds: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall: list[dict[str, object]] = []
    by_loc: list[dict[str, object]] = []
    for repeat, fold, train, valid in _outer_folds(frame, repeats, folds, seed):
        base = {"repeat": repeat, "fold": fold, "train_rows": len(train),
                "validation_rows": len(valid)}
        y_log = train["log_count_target"].to_numpy(dtype=float)
        for name, strategy in (("dummy_mean_log", "mean"), ("dummy_median_log", "median")):
            pred = DummyRegressor(strategy=strategy).fit(train[["filename"]], y_log).predict(
                valid[["filename"]]
            )
            _records(base, name, valid, pred, overall, by_loc)
        for view, columns in SHORTCUT_VIEWS.items():
            for model in MODELS:
                pred = _fit_predict_log(model, columns, train, valid, y_log)
                _records(base, f"{view}|{model}", valid, pred, overall, by_loc)
        print(f"[shortcut] repeticao {repeat} dobra {fold} concluida", flush=True)
    return pd.DataFrame(overall), pd.DataFrame(by_loc)


def _load_text_frame(max_rows: int | None) -> pd.DataFrame:
    frame = load_count_frame(need_text=True, split_name="train")
    description = pq.read_table(DEFAULT_SAMPLE, columns=["content_hash", "description"])
    frame = frame.merge(description.to_pandas(), on="content_hash", how="left", validate="1:1")
    frame["description"] = frame["description"].fillna("")
    return subset_frame(frame, max_rows)


def _tfidf(train_text: pd.Series, valid_text: pd.Series, y_log: np.ndarray, k: int) -> tuple:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.95, max_features=10_000, sublinear_tf=True,
        strip_accents="unicode", dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_text)
    x_valid = vectorizer.transform(valid_text)
    k = min(k, x_train.shape[1])
    selector = SelectKBest(f_regression, k=k).fit(x_train, y_log)
    return selector.transform(x_train), selector.transform(x_valid)


def run_text(
    frame: pd.DataFrame, *, repeats: int, folds: int, seed: int, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall: list[dict[str, object]] = []
    by_loc: list[dict[str, object]] = []
    for repeat, fold, train, valid in _outer_folds(frame, repeats, folds, seed):
        base = {"repeat": repeat, "fold": fold, "train_rows": len(train),
                "validation_rows": len(valid)}
        y_log = train["log_count_target"].to_numpy(dtype=float)
        start = time.monotonic()
        desc_tr, desc_va = _tfidf(train["description"], valid["description"], y_log, top_k)
        full_tr, full_va = _tfidf(train["content"], valid["content"], y_log, top_k)
        loc_pre = _preprocessor(["location_class"])
        loc_tr = csr_matrix(loc_pre.fit_transform(train[["location_class"]]))
        loc_va = csr_matrix(loc_pre.transform(valid[["location_class"]]))
        blocks = {
            "dummy_mean_log": None,
            "location_only": (loc_tr, loc_va),
            "description": (desc_tr, desc_va),
            "content": (full_tr, full_va),
            "description+location": (hstack([desc_tr, loc_tr]).tocsr(),
                                     hstack([desc_va, loc_va]).tocsr()),
            "content+location": (hstack([full_tr, loc_tr]).tocsr(),
                                 hstack([full_va, loc_va]).tocsr()),
        }
        for name, matrices in blocks.items():
            if matrices is None:
                pred = np.full(len(valid), float(y_log.mean()))
                _records(base, name, valid, pred, overall, by_loc)
                continue
            pred = Ridge(alpha=10.0).fit(matrices[0], y_log).predict(matrices[1])
            _records(base, f"{name}|ridge_log", valid, pred, overall, by_loc)
        print(f"[text] repeticao {repeat} dobra {fold} concluida "
              f"({time.monotonic() - start:.0f}s)", flush=True)
    return pd.DataFrame(overall), pd.DataFrame(by_loc)


def _summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = ["rmsle", "mae", "mae_log1p", "spearman"]
    out = df.groupby(keys, as_index=False)[cols].agg(["mean", "std"])
    out.columns = ["_".join(p for p in c if p).rstrip("_") for c in out.columns]
    return out.sort_values(keys[:-1] + ["rmsle_mean"] if len(keys) > 1 else ["rmsle_mean"])


def _pairs(configs: set[str], wanted: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in wanted if a in configs and b in configs]


SHORTCUT_PAIRS = [
    (f"{a}|{m}", f"{b}|{m}")
    for m in MODELS
    for a, b in [
        ("repo_no_history", "repo_full"),
        ("history_only", "location_only"),
        ("repo_no_history+location", "repo_no_history"),
        ("repo_no_history+location", "location_only"),
        ("repo_full", "location_only"),
        ("skill_struct_full", "skill_struct_no_location"),
        ("skill_struct_no_location", "location_only"),
        ("skill_struct_full", "location_only"),
    ]
] + [
    (f"{v}|{m}", "dummy_mean_log")
    for m in MODELS
    for v in ("repo_full", "repo_no_history", "history_only", "location_only",
              "skill_struct_no_location")
]
TEXT_PAIRS = [
    ("description|ridge_log", "dummy_mean_log"),
    ("content|ridge_log", "dummy_mean_log"),
    ("description|ridge_log", "content|ridge_log"),
    ("description+location|ridge_log", "location_only|ridge_log"),
    ("content+location|ridge_log", "location_only|ridge_log"),
    ("description+location|ridge_log", "content+location|ridge_log"),
    ("location_only|ridge_log", "dummy_mean_log"),
]


def _paired_by_location(by_loc: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    tables = []
    for loc, part in by_loc.groupby("location_class"):
        part = part.rename(columns={"config": "config"})
        table = paired_test_table(part, _pairs(set(part["config"]), pairs), label="config")
        table.insert(0, "location_class", loc)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _save(
    out: Path, name: str, overall: pd.DataFrame, by_loc: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out / f"{name}_folds.csv", index=False)
    by_loc.to_csv(out / f"{name}_by_location_folds.csv", index=False)
    _summary(overall, ["config"]).to_csv(out / f"{name}_summary.csv", index=False)
    _summary(by_loc, ["location_class", "config"]).to_csv(
        out / f"{name}_by_location_summary.csv", index=False
    )
    paired_test_table(overall, _pairs(set(overall["config"]), pairs), label="config").to_csv(
        out / f"{name}_paired.csv", index=False
    )
    _paired_by_location(by_loc, pairs).to_csv(out / f"{name}_by_location_paired.csv", index=False)


def controls_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("shortcut", "text"), default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def controls_main() -> None:
    args = controls_parse_args()
    started = time.monotonic()
    if args.only in (None, "shortcut"):
        frame = subset_frame(load_count_frame(need_text=False, split_name="train"), args.max_rows)
        overall, by_loc = run_shortcut(
            frame, repeats=args.repeats, folds=args.folds, seed=args.seed
        )
        _save(args.output_dir, "shortcut", overall, by_loc, SHORTCUT_PAIRS)
        print(_summary(overall, ["config"]).round(4).to_string(index=False))
    if args.only in (None, "text"):
        frame = _load_text_frame(args.max_rows)
        overall, by_loc = run_text(
            frame, repeats=args.repeats, folds=args.folds, seed=args.seed, top_k=args.top_k
        )
        _save(args.output_dir, "text", overall, by_loc, TEXT_PAIRS)
        print(_summary(overall, ["config"]).round(4).to_string(index=False))
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "repeated StratifiedGroupKFold by representative_repo_id; external test "
        "not read; Ridge(alpha=10) and default RandomForest on log1p(count)",
        "repeats": args.repeats,
        "folds": args.folds,
        "seeds": [args.seed + r for r in range(args.repeats)],
        "text_top_k": args.top_k,
        "views": SHORTCUT_VIEWS,
        "diagnostic_subset": args.max_rows is not None,
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
