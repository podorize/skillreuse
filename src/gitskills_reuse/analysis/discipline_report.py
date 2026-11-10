"""Gera tabelas e figuras auditáveis para o relatório da disciplina."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold

from gitskills_reuse.modeling.count_benchmark import _metrics
from gitskills_reuse.modeling.count_targets import load_count_frame
from gitskills_reuse.modeling.preprocess import build_preprocessor, predictor_view

OUTPUT = Path("reports/modeling/discipline_benchmark/report_audit")
PREDICTIONS = Path(
    "reports/modeling/discipline_benchmark/final_repository_only/"
    "random_forest_test_predictions.csv"
)
PERMUTED_COLUMNS = (
    "repo_stars", "repo_forks", "commit_count", "history_fetched",
    "repo_language", "repo_license", "repo_is_fork", "repo_metadata_fetched",
)


def _model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=2027,
    )


def _predict(model: RandomForestRegressor, matrix: object) -> np.ndarray:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    return np.maximum(np.expm1(model.predict(dense)), 0.0)


def _summarize_tail(frame: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    boundaries = [-np.inf, *cuts, np.inf]
    labels = ["≤P50", "P50–P75", "P75–P90", "P90–P95", ">P95"]
    frame = frame.copy()
    frame["tail_bin"] = pd.cut(
        frame["count_target"], boundaries, labels=labels, include_lowest=True
    )
    rows = []
    for label, group in frame.groupby("tail_bin", observed=True):
        actual = group["count_target"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        rows.append({
            "tail_bin": str(label), "rows": len(group),
            "median_true": float(np.median(actual)),
            "median_predicted": float(np.median(predicted)),
            "median_signed_error": float(np.median(predicted - actual)),
            "underprediction_rate": float(np.mean(predicted < actual)),
            **_metrics(actual, predicted),
        })
    return pd.DataFrame(rows)


def run(output_dir: Path = OUTPUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_count_frame(need_text=False, split_name="train")
    actual_train = train["count_target"].to_numpy(dtype=float)
    cuts = [float(np.quantile(actual_train, q)) for q in (0.5, 0.75, 0.9, 0.95)]
    pd.DataFrame({"quantile": ["P50", "P75", "P90", "P95", "P99"],
                  "train_count": [float(np.quantile(actual_train, q)) for q in
                                  (0.5, 0.75, 0.9, 0.95, 0.99)]}).to_csv(
        output_dir / "target_quantiles.csv", index=False
    )
    predicted_test = pd.read_csv(PREDICTIONS)
    actual_test = predicted_test["count_target"].to_numpy(dtype=float)
    baseline = np.full(len(actual_test), np.median(actual_train))
    pd.DataFrame([
        {"model": "train_median_baseline", **_metrics(actual_test, baseline)},
        {"model": "random_forest", **_metrics(actual_test,
                                               predicted_test["prediction"].to_numpy())},
    ]).to_csv(output_dir / "baseline_comparison.csv", index=False)
    tail = _summarize_tail(predicted_test, cuts)
    tail.to_csv(output_dir / "random_forest_tail.csv", index=False)

    missing = []
    for column in PERMUTED_COLUMNS:
        missing.append({"feature": column, "train_missing_rate": float(train[column].isna().mean()),
                        "train_rows": len(train)})
    pd.DataFrame(missing).to_csv(output_dir / "repository_missingness.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(actual_train, bins=np.logspace(np.log10(2), np.log10(actual_train.max()), 35),
            color="#38679a", edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de repositórios (treino, escala log)")
    ax.set_ylabel("Número de skills (escala log)")
    ax.set_title("Distribuição da intensidade entre skills reutilizadas")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hexbin(actual_test, predicted_test["prediction"], xscale="log", yscale="log",
              gridsize=42, mincnt=1, cmap="viridis")
    ax.plot([2, max(actual_test)], [2, max(actual_test)], "--", color="#cc4339", lw=1)
    ax.set_xlim(1.8, max(actual_test) * 1.1)
    ax.set_ylim(1, max(actual_test) * 1.1)
    ax.set_xlabel("Contagem observada no teste")
    ax.set_ylabel("Previsão do Random Forest")
    ax.set_title("Valores observados e previstos no teste")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(tail["tail_bin"], tail["mae"], color="#38679a")
    ax.set_yscale("log")
    ax.set_ylabel("MAE em repositórios (escala log)")
    ax.set_xlabel("Faixa definida pelos quantis do treino")
    ax.set_title("Erro do Random Forest por intensidade no teste")
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_intensity.png", dpi=180)
    plt.close(fig)

    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2027)
    records = []
    for fold, (tr_idx, val_idx) in enumerate(folds.split(
        train, y=train["split_stratum"], groups=train["representative_repo_id"]
    ), start=1):
        tr, val = train.iloc[tr_idx], train.iloc[val_idx]
        config = {"use_text": False, "use_structural": False, "use_repository": True}
        x_tr = predictor_view(tr, **config)
        x_val = predictor_view(val, **config)
        preprocessor = build_preprocessor(x_tr, **config)
        fit_tr = preprocessor.fit_transform(x_tr)
        fit_val = preprocessor.transform(x_val)
        model = _model()
        model.fit(fit_tr.toarray() if hasattr(fit_tr, "toarray") else fit_tr,
                  tr["log_count_target"].to_numpy(dtype=float))
        y_val = val["count_target"].to_numpy(dtype=float)
        base = _metrics(y_val, _predict(model, fit_val))["rmsle"]
        for position, column in enumerate(PERMUTED_COLUMNS):
            shuffled = x_val.copy()
            shuffled[column] = np.random.default_rng(2027 + 100 * fold + position).permutation(
                shuffled[column].to_numpy()
            )
            score = _metrics(y_val, _predict(model, preprocessor.transform(shuffled)))["rmsle"]
            records.append({"fold": fold, "feature": column, "rmsle_increase": score - base})
        if fold == 1:
            sample = x_val.sample(n=min(1000, len(x_val)), random_state=2027)
            effects = []
            for stars in (0, 1, 3, 10, 100, 1000):
                varied = sample.copy()
                varied["repo_stars"] = stars
                predictions = _predict(model, preprocessor.transform(varied))
                effects.append({"repo_stars_set_to": stars,
                                "mean_predicted_count": float(np.mean(predictions))})
            effects_frame = pd.DataFrame(effects)
            effects_frame.to_csv(output_dir / "stars_scenario_fold1.csv", index=False)
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.plot(effects_frame["repo_stars_set_to"], effects_frame["mean_predicted_count"],
                    marker="o", color="#38679a")
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("Estrelas do repositório fixadas no cenário")
            ax.set_ylabel("Previsão média de repositórios")
            ax.set_title("Resposta do modelo a cenários de estrelas (dobra 1)")
            fig.tight_layout()
            fig.savefig(output_dir / "stars_scenario_fold1.png", dpi=180)
            plt.close(fig)
    perm = pd.DataFrame(records)
    perm.to_csv(output_dir / "permutation_by_fold.csv", index=False)
    perm_summary = (perm.groupby("feature")["rmsle_increase"]
                    .agg(["mean", "std"]).sort_values("mean", ascending=False))
    perm_summary.to_csv(output_dir / "permutation_summary.csv", index_label="feature")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.barh(perm_summary.index.astype(str), perm_summary["mean"], color="#38679a")
    ax.invert_yaxis()
    ax.set_xlabel("Aumento do RMSLE após permutação (validação)")
    ax.set_title("Importância agrupada dos atributos do repositório")
    fig.tight_layout()
    fig.savefig(output_dir / "permutation_importance.png", dpi=180)
    plt.close(fig)
    (output_dir / "audit_manifest.json").write_text(json.dumps({
        "source_predictions": str(PREDICTIONS),
        "source_split": "external test; no model refitting for tail or baseline",
        "tail_cutoffs_from": "development train",
        "tail_cutoffs": cuts,
        "permutation_split": "development only, 3 StratifiedGroupKFold folds",
        "group": "representative_repo_id", "seed": 2027,
        "permutations_per_feature_per_fold": 1,
        "model": "random_forest_d12_n100",
    }, indent=2) + "\n", encoding="utf-8")
    print("Artefatos gerados em", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
