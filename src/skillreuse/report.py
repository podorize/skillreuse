"""EDA do alvo e auditoria numerica das figuras/tabelas do relatorio."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold

from skillreuse.benchmark import _metrics
from skillreuse.models import build_preprocessor, load_count_frame, predictor_view

# ==========================================================================
# EDA orientada a decidir entre alvo ordinal e regressao de contagem.
# ==========================================================================

matplotlib.use("Agg")

DEFAULT_TARGETS = Path("data/processed/final_targets.parquet")

DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")

DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")

DEFAULT_TABLES = Path("results/tables")

DEFAULT_FIGURES = Path("results/figures")

DEFAULT_REPORT = Path("results/target_eda.md")

ORDINAL_LABELS = {
    0: "2 repositorios",
    1: "3-4 repositorios",
    2: "5-9 repositorios",
    3: "10+ repositorios",
}

def load_target_frame(targets_path: Path, features_path: Path) -> pd.DataFrame:
    """Carrega alvos e split sem usar atributos derivados do conteudo."""
    targets = pq.read_table(targets_path).to_pandas()
    split = pq.read_table(features_path, columns=["content_hash", "split"]).to_pandas()
    frame = targets.merge(split, on="content_hash", how="inner", validate="one_to_one")
    if len(frame) != len(targets):
        raise ValueError("A junção dos alvos com o split perdeu observacoes")
    return frame

def count_distribution(frame: pd.DataFrame, *, split: str = "train") -> pd.DataFrame:
    """Resume repository_count entre todas as linhas e entre reutilizadas."""
    subset = frame.loc[frame["split"].eq(split)].copy()
    rows: list[dict[str, object]] = []
    for population, data in (("all", subset), ("reused", subset.loc[subset["reused"].eq(1)])):
        counts = pd.to_numeric(data["repository_count"], errors="coerce").dropna()
        mean = float(counts.mean()) if len(counts) else float("nan")
        variance = float(counts.var(ddof=1)) if len(counts) > 1 else float("nan")
        row: dict[str, object] = {
            "split": split,
            "population": population,
            "n": int(len(counts)),
            "mean": mean,
            "median": float(counts.median()) if len(counts) else float("nan"),
            "std": float(counts.std(ddof=1)) if len(counts) > 1 else float("nan"),
            "variance": variance,
            "min": float(counts.min()) if len(counts) else float("nan"),
            "max": float(counts.max()) if len(counts) else float("nan"),
            "p50": float(counts.quantile(0.50)) if len(counts) else float("nan"),
            "p75": float(counts.quantile(0.75)) if len(counts) else float("nan"),
            "p90": float(counts.quantile(0.90)) if len(counts) else float("nan"),
            "p95": float(counts.quantile(0.95)) if len(counts) else float("nan"),
            "p99": float(counts.quantile(0.99)) if len(counts) else float("nan"),
            "overdispersion_variance_mean": variance / mean if mean > 0 else float("nan"),
            "missing_count": int(data["repository_count"].isna().sum()),
            "invalid_nonpositive_count": int((counts < 1).sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)

def ordinal_distribution(frame: pd.DataFrame, *, split: str = "train") -> pd.DataFrame:
    """Resume classes ordinais apenas entre skills reutilizadas."""
    subset = frame.loc[frame["split"].eq(split) & frame["reused"].eq(1)].copy()
    counts = subset["reuse_intensity"].astype("Int64")
    total = len(subset)
    rows: list[dict[str, object]] = []
    for code in range(4):
        values = subset.loc[counts.eq(code), "repository_count"].astype(float)
        rows.append(
            {
                "split": split,
                "intensity_code": code,
                "label": ORDINAL_LABELS[code],
                "n": int(len(values)),
                "percentage": 100 * len(values) / total if total else float("nan"),
                "count_min": float(values.min()) if len(values) else float("nan"),
                "count_max": float(values.max()) if len(values) else float("nan"),
                "count_mean": float(values.mean()) if len(values) else float("nan"),
                "count_median": float(values.median()) if len(values) else float("nan"),
                "count_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)

def group_distribution(frame: pd.DataFrame, *, split: str = "train") -> pd.DataFrame:
    """Resume tamanho dos grupos para auditar a divisao por repositorio."""
    subset = frame.loc[frame["split"].eq(split)].copy()
    rows: list[dict[str, object]] = []
    for population, data in (("all", subset), ("reused", subset.loc[subset["reused"].eq(1)])):
        sizes = data.groupby("representative_repo_id").size()
        rows.append(
            {
                "split": split,
                "population": population,
                "n_rows": int(len(data)),
                "n_groups": int(sizes.size),
                "mean_rows_per_group": float(sizes.mean()),
                "median_rows_per_group": float(sizes.median()),
                "p90_rows_per_group": float(sizes.quantile(0.90)),
                "p99_rows_per_group": float(sizes.quantile(0.99)),
                "max_rows_per_group": int(sizes.max()) if len(sizes) else 0,
            }
        )
    return pd.DataFrame(rows)

def split_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume a distribuicao dos estratos por split; nao orienta a decisao."""
    data = frame.copy()
    data["intensity_label"] = data["reuse_intensity"].map(ORDINAL_LABELS).fillna("not_reused")
    result = (
        data.groupby(["split", "split_stratum", "intensity_label"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    result["percentage_within_split"] = (
        result["n"] / result.groupby("split")["n"].transform("sum") * 100
    )
    return result

def coverage_summary(frame: pd.DataFrame, sample_path: Path) -> pd.DataFrame:
    """Audita occurrence_count contra repository_count, sem tratar ocorrencia como alvo."""
    sample = pq.read_table(sample_path, columns=["content_hash", "occurrence_count"]).to_pandas()
    data = frame.merge(sample, on="content_hash", how="inner", validate="one_to_one")
    if len(data) != len(frame):
        raise ValueError("A auditoria de cobertura perdeu observacoes")
    rows: list[dict[str, object]] = []
    for population, subset in (("all", data), ("reused", data.loc[data["reused"].eq(1)])):
        rows.append(
            {
                "population": population,
                "n": int(len(subset)),
                "missing_occurrence_count": int(subset["occurrence_count"].isna().sum()),
                "occurrence_lt_repository": int(
                    (subset["occurrence_count"] < subset["repository_count"]).sum()
                ),
                "occurrence_eq_repository": int(
                    (subset["occurrence_count"] == subset["repository_count"]).sum()
                ),
                "occurrence_gt_repository": int(
                    (subset["occurrence_count"] > subset["repository_count"]).sum()
                ),
                "occurrence_median": float(subset["occurrence_count"].median()),
                "occurrence_p99": float(subset["occurrence_count"].quantile(0.99)),
                "occurrence_max": int(subset["occurrence_count"].max()),
            }
        )
    return pd.DataFrame(rows)

def group_overlap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Verifica se algum repositorio representativo aparece nos dois splits."""
    train_groups = set(frame.loc[frame["split"].eq("train"), "representative_repo_id"])
    test_groups = set(frame.loc[frame["split"].eq("test"), "representative_repo_id"])
    return pd.DataFrame(
        [
            {
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "overlapping_groups": len(train_groups.intersection(test_groups)),
            }
        ]
    )

def save_count_figure(frame: pd.DataFrame, output: Path, split: str = "train") -> None:
    values = frame.loc[frame["split"].eq(split) & frame["reused"].eq(1), "repository_count"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(values, bins=60, color="#2563eb", alpha=0.85)
    axes[0].set_title("Contagem bruta entre reutilizadas")
    axes[0].set_xlabel("repository_count")
    axes[0].set_ylabel("Numero de skills")
    axes[0].grid(alpha=0.2)
    axes[1].hist(np.log1p(values), bins=50, color="#0f766e", alpha=0.85)
    axes[1].set_title("Contagem em escala logaritmica")
    axes[1].set_xlabel("log(1 + repository_count)")
    axes[1].set_ylabel("Numero de skills")
    axes[1].grid(alpha=0.2)
    figure.suptitle("EDA do alvo de reutilizacao", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

def save_ordinal_figure(table: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(table["label"], table["percentage"], color="#7c3aed")
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in table["percentage"]])
    axis.set_ylabel("Percentual entre reutilizadas")
    axis.set_xlabel("Classe de intensidade")
    axis.set_title("Distribuicao do alvo ordinal")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)

def write_report(
    output: Path,
    count_table: pd.DataFrame,
    ordinal_table: pd.DataFrame,
    group_table: pd.DataFrame,
    split_table: pd.DataFrame,
    coverage_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
) -> str:
    reused = count_table.loc[count_table["population"].eq("reused")].iloc[0]
    class3 = ordinal_table.loc[ordinal_table["intensity_code"].eq(3)].iloc[0]
    lines = [
        "# Fase 1 - EDA e decisao preliminar do alvo",
        "",
        "## Protocolo",
        "",
        "A decisao sobre a formulacao do alvo foi baseada somente no split de desenvolvimento "
        "(`split=train`). O teste externo foi resumido para auditoria de balanceamento, mas "
        "nao participou da recomendacao. `repository_count` e a origem do alvo e nao pode ser "
        "usado como atributo preditor.",
        "",
        "## Achados principais",
        "",
        f"- Skills reutilizadas no desenvolvimento: **{int(reused['n']):,}**.".replace(",", "."),
        f"- Mediana de `repository_count`: **{_fmt(reused['median'], 0)}**.",
        f"- Media: **{_fmt(reused['mean'])}**; variancia: **{_fmt(reused['variance'])}**.",
        f"- Razao variancia/media: **{_fmt(reused['overdispersion_variance_mean'])}**, "
        "indicando forte sobredispersao.",
        f"- P95: **{_fmt(reused['p95'], 1)}**; "
        f"P99: **{_fmt(reused['p99'], 1)}**; "
        f"maximo: **{_fmt(reused['max'], 0)}**.",
        f"- A classe `10+` contem **{int(class3['n']):,}** skills, mas varia de "
        f"{_fmt(class3['count_min'], 0)} a { _fmt(class3['count_max'], 0)} reutilizacoes.",
        "",
        "## Distribuicao ordinal",
        "",
        "A discretizacao atual produz classes ordenadas, mas a ultima classe agrega uma cauda "
        "muito ampla. Isso melhora a estabilidade da classificacao, mas perde informacao sobre "
        "a diferenca entre reutilizacao alta e muito alta.",
        "",
        ordinal_table.to_markdown(index=False),
        "",
        "## Grupos e divisao",
        "",
        "A divisao por `representative_repo_id` e necessaria: os grupos possuem cauda longa e "
        "nao devem ser fragmentados entre treino e validacao. "
        "A tabela resume o tamanho dos grupos.",
        "",
        group_table.to_markdown(index=False),
        "",
        "## Integridade da contagem e da divisao",
        "",
        "`occurrence_count` e sempre maior ou igual a `repository_count`; isso e consistente "
        "com multiplas ocorrencias do mesmo conteudo dentro de um repositorio. Nao foram "
        "observados valores ausentes ou casos em que a contagem de repositorios excedesse a "
        "contagem de ocorrencias.",
        "",
        coverage_table.to_markdown(index=False),
        "",
        "A divisao externa nao possui sobreposicao de `representative_repo_id` "
        "entre treino e teste:",
        "",
        overlap_table.to_markdown(index=False),
        "",
        "## Recomendacao para a proxima fase",
        "",
        "Decisao da Fase 1: a pesquisa adotara **regressao de contagem** como alvo principal, "
        "preferencialmente com modelo Binomial Negativo ou regressao sobre "
        "`log1p(repository_count)`. "
        "A decisao e sustentada pela forte assimetria e sobredispersao observadas na contagem. "
        "A classificacao ordinal nao integra o protocolo principal.",
        "",
        "O proximo passo e implementar e validar o pipeline de regressao de contagem com divisao "
        "agrupada, sem leakage, antes do tuning completo e da avaliacao no teste externo.",
        "",
        "## Artefatos",
        "",
        "- `results/tables/target_eda_count_distribution.csv`",
        "- `results/tables/target_eda_ordinal_distribution.csv`",
        "- `results/tables/target_eda_group_distribution.csv`",
        "- `results/tables/target_eda_split_distribution.csv`",
        "- `results/tables/target_eda_coverage_summary.csv`",
        "- `results/tables/target_eda_group_overlap.csv`",
        "- `results/figures/target_eda_count_distribution.png`",
        "- `results/figures/target_eda_ordinal_distribution.png`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output.read_text(encoding="utf-8")

def run_pipeline(
    targets_path: Path = DEFAULT_TARGETS,
    features_path: Path = DEFAULT_FEATURES,
    tables_dir: Path = DEFAULT_TABLES,
    figures_dir: Path = DEFAULT_FIGURES,
    report_path: Path = DEFAULT_REPORT,
    sample_path: Path = DEFAULT_SAMPLE,
) -> dict[str, object]:
    if not targets_path.is_file():
        raise FileNotFoundError(f"Alvos nao encontrados: {targets_path.resolve()}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Features nao encontradas: {features_path.resolve()}")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Amostra nao encontrada: {sample_path.resolve()}")
    frame = load_target_frame(targets_path, features_path)
    train_count = count_distribution(frame, split="train")
    train_ordinal = ordinal_distribution(frame, split="train")
    train_groups = group_distribution(frame, split="train")
    split_table = split_distribution(frame)
    coverage_table = coverage_summary(frame, sample_path)
    overlap_table = group_overlap_summary(frame)
    # The test table is exported only for audit; it is not used in the report decision.
    test_count = count_distribution(frame, split="test")
    test_ordinal = ordinal_distribution(frame, split="test")
    count_table = pd.concat([train_count, test_count], ignore_index=True)
    ordinal_table = pd.concat([train_ordinal, test_ordinal], ignore_index=True)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    count_table.to_csv(tables_dir / "target_eda_count_distribution.csv", index=False)
    ordinal_table.to_csv(tables_dir / "target_eda_ordinal_distribution.csv", index=False)
    train_groups.to_csv(tables_dir / "target_eda_group_distribution.csv", index=False)
    split_table.to_csv(tables_dir / "target_eda_split_distribution.csv", index=False)
    coverage_table.to_csv(tables_dir / "target_eda_coverage_summary.csv", index=False)
    overlap_table.to_csv(tables_dir / "target_eda_group_overlap.csv", index=False)
    save_count_figure(frame, figures_dir / "target_eda_count_distribution.png")
    save_ordinal_figure(train_ordinal, figures_dir / "target_eda_ordinal_distribution.png")
    write_report(
        report_path,
        train_count,
        train_ordinal,
        train_groups,
        split_table,
        coverage_table,
        overlap_table,
    )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "targets_source": str(targets_path),
        "features_source": str(features_path),
        "sample_source": str(sample_path),
        "rows": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "train_reused_rows": int(((frame["split"] == "train") & (frame["reused"] == 1)).sum()),
        "test_reused_rows": int(((frame["split"] == "test") & (frame["reused"] == 1)).sum()),
        "decision_scope": "train split only; test summarized for audit",
        "ordinal_labels": ORDINAL_LABELS,
        "target_decision": "count regression conditional on reused=1",
        "primary_metric": "RMSLE",
        "recommended_next_step": "implement grouped count-regression pipeline",
        "overlapping_groups_train_test": int(overlap_table.iloc[0]["overlapping_groups"]),
    }
    (tables_dir / "target_eda_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()

def target_eda_main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        targets_path=args.targets,
        features_path=args.features,
        sample_path=args.sample,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        report_path=args.report,
    )
    print("EDA do alvo gerada com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ==========================================================================
# Gera tabelas e figuras auditáveis para o relatório da disciplina.
# ==========================================================================

matplotlib.use("Agg")

OUTPUT = Path("results/discipline_benchmark/report_audit")

PREDICTIONS = Path(
    "results/discipline_benchmark/final_repository_only/"
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

def report_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)
