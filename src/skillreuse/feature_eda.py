"""EDA dos atributos do protocolo principal: ausentes, assimetria, colinearidade, equilibrio.

Calculada sobre o conjunto de desenvolvimento com limite de 50 por dono (o mesmo dos modelos
principais); nenhuma informacao do holdout e usada.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from skillreuse.owner_models import CATEGORICAL, CLEAN_FULL, _prepare, _preprocessor  # noqa: E402

OUTPUT_DIR = Path("results/owner/feature_eda")
SKEW_THRESHOLD = 2.0
CORR_THRESHOLD = 0.95


def skewed_columns(frame: pd.DataFrame, columns: list[str] = CLEAN_FULL) -> list[str]:
    """Numericas nao negativas com assimetria acima do limiar (propriedade marginal, sem alvo)."""
    out = []
    for col in columns:
        if col in CATEGORICAL:
            continue
        s = frame[col].dropna()
        if len(s) and s.min() >= 0 and s.nunique() > 2 and s.skew() > SKEW_THRESHOLD:
            out.append(col)
    return out


def feature_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CLEAN_FULL:
        s = frame[col]
        row = {"feature": col, "kind": "categorical" if col in CATEGORICAL else "numeric",
               "missing_share": float(s.isna().mean()), "n_unique": int(s.nunique())}
        if col not in CATEGORICAL:
            v = s.dropna()
            row.update({
                "skew": float(v.skew()),
                "skew_after_log1p": float(np.log1p(v.clip(lower=0)).skew()) if v.min() >= 0
                else float("nan"),
                "p99_over_median": float(v.quantile(0.99) / v.median()) if v.median() > 0
                else float("nan"),
                "constant": bool(v.nunique() <= 1),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def collinear_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in CLEAN_FULL if c not in CATEGORICAL and frame[c].nunique() > 1]
    corr = frame[numeric].corr(method="spearman").abs()
    pairs = [{"a": a, "b": b, "abs_spearman": float(corr.loc[a, b])}
             for i, a in enumerate(numeric) for b in numeric[i + 1:]
             if corr.loc[a, b] > CORR_THRESHOLD]
    return pd.DataFrame(pairs).sort_values("abs_spearman", ascending=False)


def _figure(frame: pd.DataFrame, profile: pd.DataFrame, out: Path) -> None:
    numeric = [c for c in CLEAN_FULL if c not in CATEGORICAL and frame[c].nunique() > 1]
    corr = frame[numeric].corr(method="spearman").abs().fillna(0).to_numpy()
    order = leaves_list(linkage(squareform(1 - corr, checks=False), "average"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    top = profile.dropna(subset=["skew"]).sort_values("skew", ascending=False).head(8)
    y = np.arange(len(top))
    ax[0].barh(y + 0.2, top["skew"], 0.4, label="raw", color="#d62728")
    ax[0].barh(y - 0.2, top["skew_after_log1p"].fillna(0), 0.4, label="after log1p",
               color="#1f77b4")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(top["feature"], fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("skewness")
    ax[0].set_title("Most skewed numeric attributes", fontsize=10)
    ax[0].legend(fontsize=8)
    im = ax[1].imshow(corr[np.ix_(order, order)], vmin=0, vmax=1, cmap="viridis")
    ax[1].set_title("|Spearman| among numeric attributes", fontsize=10)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "feature_eda.png", dpi=160)
    plt.close(fig)


def feature_eda_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-per-owner", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def feature_eda_main() -> None:
    args = feature_eda_parse_args()
    ns = argparse.Namespace(tier=None, exclude_owners_above=None,
                            cap_per_owner=args.cap_per_owner, max_rows=None, seed=2027)
    frame = _prepare(ns)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = feature_profile(frame)
    pairs = collinear_pairs(frame)
    profile.to_csv(args.output_dir / "feature_profile.csv", index=False)
    pairs.to_csv(args.output_dir / "collinear_pairs.csv", index=False)
    matrix = _preprocessor(CLEAN_FULL).fit_transform(frame[CLEAN_FULL])
    skewed = skewed_columns(frame)
    missing = profile[profile["missing_share"] > 0]
    summary = {
        "rows": int(len(frame)), "owners": int(frame["repo_owner"].nunique()),
        "numeric_columns": int((profile["kind"] == "numeric").sum()),
        "categorical_columns": int((profile["kind"] == "categorical").sum()),
        "matrix_columns_after_preprocessing": int(matrix.shape[1]),
        "columns_with_missing": int(len(missing)),
        "max_missing_share": float(profile["missing_share"].max()),
        "constant_columns": profile.loc[profile.get("constant", False) == True,  # noqa: E712
                                        "feature"].tolist(),
        "collinear_pairs_gt_0.95": int(len(pairs)),
        "skewed_columns_gt_2": skewed,
        "cross_owner_prevalence": float(frame["cross_owner"].mean()),
        "owner_count_equal_1_share": float((frame["owner_count"] == 1).mean()),
    }
    (args.output_dir / "feature_eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _figure(frame, profile, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
