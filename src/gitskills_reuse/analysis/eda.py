"""Executa a analise exploratoria da populacao e da amostra final do SkillReuse."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from pandas.api.types import is_object_dtype, is_string_dtype

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_FEATURES = Path("data/processed/final_structured_features.parquet")
DEFAULT_POPULATION_QUALITY = Path("reports/tables/full_data_quality.csv")
DEFAULT_TABLES = Path("reports/tables")
DEFAULT_FIGURES = Path("reports/figures")
DEFAULT_REPORT = Path("reports/data_analysis.md")

NUMERIC_COLUMNS = [
    "text_char_count",
    "text_line_count",
    "text_nonempty_line_count",
    "text_word_count",
    "text_unique_word_count",
    "text_lexical_diversity",
    "text_average_word_length",
    "text_sentence_count",
    "text_average_sentence_words",
    "markdown_heading_count",
    "markdown_unordered_list_items",
    "markdown_ordered_list_items",
    "markdown_fenced_code_blocks",
    "markdown_inline_code_count",
    "markdown_link_count",
    "markdown_url_count",
    "markdown_blockquote_count",
    "markdown_checkbox_count",
    "markdown_table_row_count",
    "frontmatter_name_char_count",
    "frontmatter_description_char_count",
    "body_chars",
    "sibling_count",
    "sibling_bytes",
    "commit_count",
    "repo_stars",
    "repo_forks",
]

CATEGORICAL_COLUMNS = [
    "location_class",
    "frontmatter_valid",
    "has_scripts",
    "has_references",
    "composition_fetched",
    "composition_truncated",
    "history_fetched",
    "repo_language",
    "repo_license",
]

CORRELATION_COLUMNS = [
    "text_word_count",
    "text_unique_word_count",
    "text_lexical_diversity",
    "text_sentence_count",
    "markdown_heading_count",
    "markdown_unordered_list_items",
    "markdown_ordered_list_items",
    "markdown_fenced_code_blocks",
    "markdown_inline_code_count",
    "markdown_link_count",
    "markdown_table_row_count",
    "sibling_count",
    "repo_stars",
    "repo_forks",
]

PLOT_NUMERIC_COLUMNS = [
    "text_word_count",
    "markdown_fenced_code_blocks",
    "sibling_count",
    "repo_stars",
]


def attribute_catalog() -> pd.DataFrame:
    """Classifica atributos conforme tipo e escala discutidos no capitulo-base."""
    rows = [
        ("content_hash", "qualitativo", "nominal", "identificador; nao e feature"),
        (
            "representative_repo_id",
            "qualitativo",
            "nominal",
            "identificador de grupo; nao e feature",
        ),
        ("reused", "qualitativo binario", "nominal", "atributo-alvo"),
        ("split", "qualitativo", "nominal", "controle experimental"),
        ("filename", "qualitativo", "nominal", "nome do arquivo"),
        ("location_class", "qualitativo", "nominal", "localizacao no repositorio"),
        ("repo_language", "qualitativo", "nominal", "linguagem do repositorio"),
        ("repo_license", "qualitativo", "nominal", "licenca do repositorio"),
        ("first_commit_at", "quantitativo temporal", "intervalar", "data/hora"),
        ("last_commit_at", "quantitativo temporal", "intervalar", "data/hora"),
        ("repo_created_at", "quantitativo temporal", "intervalar", "data/hora"),
        ("repo_pushed_at", "quantitativo temporal", "intervalar", "data/hora"),
    ]
    binary = [
        "frontmatter_valid",
        "frontmatter_name_missing",
        "frontmatter_description_missing",
        "has_scripts",
        "has_references",
        "composition_fetched",
        "composition_truncated",
        "history_fetched",
        "repo_is_fork",
        "repo_metadata_fetched",
    ]
    rows.extend((name, "qualitativo binario", "nominal", "indicador") for name in binary)
    discrete = [
        name
        for name in NUMERIC_COLUMNS
        if name
        not in {
            "text_lexical_diversity",
            "text_average_word_length",
            "text_average_sentence_words",
        }
    ]
    rows.extend(
        (name, "quantitativo discreto", "racional", "contagem") for name in discrete
    )
    continuous = [
        "text_lexical_diversity",
        "text_average_word_length",
        "text_average_sentence_words",
    ]
    rows.extend(
        (name, "quantitativo continuo", "racional", "medida derivada")
        for name in continuous
    )
    return pd.DataFrame(rows, columns=["attribute", "type", "scale", "role"])


def numeric_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calcula localidade, espalhamento, formato e outliers pelo criterio 1,5 IQR."""
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "feature": column,
                "valid_count": len(values),
                "missing_count": int(frame[column].isna().sum()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "q1": q1,
                "q3": q3,
                "maximum": float(values.max()),
                "iqr": iqr,
                "skewness": float(values.skew()),
                "excess_kurtosis": float(values.kurt()),
                "outlier_count_1_5_iqr": outliers,
                "outlier_percentage_1_5_iqr": 100 * outliers / len(values),
            }
        )
    return pd.DataFrame(rows)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = len(frame)
    missing = frame.isna().sum().rename("missing_count").reset_index()
    missing = missing.rename(columns={"index": "feature"})
    missing["missing_percentage"] = 100 * missing["missing_count"] / rows
    return missing.sort_values(["missing_percentage", "feature"], ascending=[False, True])


def normalize_empty_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Converte strings vazias ou compostas por espacos em valores ausentes."""
    result = frame.copy()
    for column in result.columns:
        if is_string_dtype(result[column].dtype) or is_object_dtype(result[column].dtype):
            values = result[column].astype("string")
            result[column] = values.mask(values.str.strip().eq(""), pd.NA)
    return result


def constant_feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Lista atributos sem variacao observada, que nao ajudam a classificacao."""
    rows = []
    excluded = {"content_hash", "representative_repo_id", "reused", "split"}
    for column in frame.columns:
        if column in excluded:
            continue
        distinct = frame[column].nunique(dropna=True)
        if distinct <= 1:
            observed = frame[column].dropna()
            rows.append(
                {
                    "feature": column,
                    "distinct_non_missing_values": int(distinct),
                    "only_value": observed.iloc[0] if len(observed) else "<ausente>",
                    "missing_count": int(frame[column].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def categorical_frequencies(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume frequencias; limita atributos de alta cardinalidade aos 20 mais comuns."""
    rows: list[dict[str, object]] = []
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype("string").fillna("<ausente>")
        counts = values.value_counts(dropna=False)
        limit = len(counts) if len(counts) <= 20 else 20
        for value, count in counts.iloc[:limit].items():
            rows.append(
                {
                    "feature": column,
                    "value": value,
                    "count": int(count),
                    "percentage": 100 * int(count) / len(frame),
                    "is_top_20_truncated": len(counts) > 20,
                }
            )
    return pd.DataFrame(rows)


def numeric_target_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Compara classes usando medias, medianas, SMD e correlacao ponto-bisserial."""
    rows = []
    for column in NUMERIC_COLUMNS:
        data = frame[[column, "reused"]].dropna()
        negative = pd.to_numeric(data.loc[data["reused"] == 0, column])
        positive = pd.to_numeric(data.loc[data["reused"] == 1, column])
        pooled_variance = (
            (len(negative) - 1) * negative.var(ddof=1)
            + (len(positive) - 1) * positive.var(ddof=1)
        ) / max(len(negative) + len(positive) - 2, 1)
        pooled_std = float(np.sqrt(pooled_variance))
        smd = (
            float((positive.mean() - negative.mean()) / pooled_std)
            if pooled_std > 0
            else 0.0
        )
        correlation = float(data[column].corr(data["reused"]))
        rows.append(
            {
                "feature": column,
                "negative_count": len(negative),
                "positive_count": len(positive),
                "negative_mean": float(negative.mean()),
                "positive_mean": float(positive.mean()),
                "negative_median": float(negative.median()),
                "positive_median": float(positive.median()),
                "standardized_mean_difference": smd,
                "point_biserial_correlation": correlation,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        "point_biserial_correlation", key=lambda values: values.abs(), ascending=False
    )


def categorical_target_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CATEGORICAL_COLUMNS:
        data = frame[[column, "reused"]].copy()
        data[column] = data[column].astype("string").fillna("<ausente>")
        grouped = (
            data.groupby(column)["reused"]
            .agg(["size", "sum", "mean"])
            .reset_index()
            .rename(
                columns={
                    column: "value",
                    "size": "count",
                    "sum": "positive_count",
                    "mean": "positive_rate",
                }
            )
            .sort_values("count", ascending=False)
            .head(20)
        )
        grouped.insert(0, "feature", column)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def correlation_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, first in enumerate(matrix.columns):
        for second in matrix.columns[index + 1 :]:
            rows.append(
                {"feature_1": first, "feature_2": second, "correlation": matrix.loc[first, second]}
            )
    result = pd.DataFrame(rows)
    return result.sort_values("correlation", key=lambda values: values.abs(), ascending=False)


def save_missingness_figure(summary: pd.DataFrame, output: Path) -> None:
    visible = summary.loc[summary["missing_count"] > 0].head(15).sort_values(
        "missing_percentage"
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(visible["feature"], visible["missing_percentage"], color="#d97706")
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in visible["missing_percentage"]])
    axis.set_xlabel("Valores ausentes (%)")
    axis.set_title("Atributos com valores ausentes na amostra final")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_distribution_figure(train: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = {0: "Nao reutilizada", 1: "Reutilizada"}
    colors = {0: "#64748b", 1: "#2563eb"}
    for axis, column in zip(axes.flat, PLOT_NUMERIC_COLUMNS, strict=True):
        for target in (0, 1):
            values = pd.to_numeric(
                train.loc[train["reused"] == target, column], errors="coerce"
            ).dropna()
            axis.hist(
                np.log1p(values),
                bins=50,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=labels[target],
                color=colors[target],
            )
        axis.set_title(column)
        axis.set_xlabel("log(1 + valor)")
        axis.set_ylabel("Densidade")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("Distribuicoes no treino, separadas pelo alvo", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_boxplot_figure(train: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, column in zip(axes.flat, PLOT_NUMERIC_COLUMNS, strict=True):
        groups = [
            np.log1p(pd.to_numeric(train.loc[train["reused"] == target, column]).dropna())
            for target in (0, 1)
        ]
        axis.boxplot(groups, tick_labels=["Nao reutilizada", "Reutilizada"], showfliers=False)
        axis.set_title(column)
        axis.set_ylabel("log(1 + valor)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Boxplots modificados por classe (sem exibir outliers)", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_correlation_figure(matrix: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        ax=axis,
    )
    axis.set_title("Correlacao de Spearman entre atributos numericos do treino")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_categorical_rate_figure(rates: pd.DataFrame, output: Path) -> None:
    columns = ["location_class", "frontmatter_valid", "has_scripts", "has_references"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, column in zip(axes.flat, columns, strict=True):
        data = rates.loc[rates["feature"] == column].head(10).copy()
        data["value"] = data["value"].astype(str)
        bars = axis.bar(data["value"], 100 * data["positive_rate"], color="#2563eb")
        axis.set_title(column)
        axis.set_ylabel("Reutilizadas (%)")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        for bar, count in zip(bars, data["count"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={count:,}".replace(",", "."),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    figure.suptitle("Taxa de reutilizacao por atributo categorico no treino", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_scatter_figure(train: pd.DataFrame, output: Path, random_seed: int) -> None:
    sample = train.sample(n=min(8_000, len(train)), random_state=random_seed)
    figure, axis = plt.subplots(figsize=(9, 6))
    for target, label, color in (
        (0, "Nao reutilizada", "#64748b"),
        (1, "Reutilizada", "#2563eb"),
    ):
        subset = sample.loc[sample["reused"] == target]
        axis.scatter(
            np.log1p(subset["text_word_count"]),
            np.log1p(subset["markdown_heading_count"]),
            s=10,
            alpha=0.25,
            label=label,
            color=color,
        )
    axis.set_xlabel("log(1 + numero de palavras)")
    axis.set_ylabel("log(1 + numero de titulos Markdown)")
    axis.set_title("Relacao entre volume textual e estrutura Markdown")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 10) -> str:
    visible = frame.loc[:, columns].head(rows).copy()
    for column in visible.select_dtypes(include=["float"]).columns:
        visible[column] = visible[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in visible.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_report(
    output: Path,
    population: dict[str, float],
    frame: pd.DataFrame,
    train: pd.DataFrame,
    numeric: pd.DataFrame,
    missing: pd.DataFrame,
    associations: pd.DataFrame,
    correlation: pd.DataFrame,
    categorical_rates: pd.DataFrame,
    constants: pd.DataFrame,
) -> None:
    top_missing = missing.loc[missing["missing_count"] > 0]
    top_outliers = numeric.sort_values("outlier_percentage_1_5_iqr", ascending=False)
    location_rates = categorical_rates.loc[
        categorical_rates["feature"] == "location_class"
    ]
    lines = [
        "# Analise exploratoria dos dados do SkillReuse",
        "",
        "## Escopo e protocolo",
        "",
        "A analise segue o roteiro do Capitulo 2 - Analise fornecido para o projeto: "
        "caracterizacao dos dados, classificacao dos atributos por tipo e escala, estatistica "
        "descritiva univariada e analise multivariada por correlacoes e visualizacoes.",
        "",
        "As estatisticas da populacao e o balanceamento da amostra usam todos os dados. "
        "Comparacoes entre atributos preditivos e o alvo usam somente o treino congelado, "
        "preservando o teste para a avaliacao final.",
        "",
        "## Caracterizacao",
        "",
        f"- Populacao: {int(population['population_contents']):,} conteudos distintos e "
        f"{int(population['artifact_occurrences']):,} ocorrencias.".replace(",", "."),
        (
            f"- Amostra de modelagem: {len(frame):,} conteudos; treino: {len(train):,}."
        ).replace(",", "."),
        f"- Alvo positivo na amostra: {100 * frame['reused'].mean():.2f}%.",
        "- Estrutura: dados mistos; texto nao estruturado transformado em atributos estruturados, "
        "categorias, contagens e metadados.",
        "- Tempo: snapshot estatico da coleta; a ordem temporal interna dos textos e relevante, "
        "mas a tarefa atual nao e uma previsao temporal.",
        "- Dependencia: conteudos podem compartilhar repositorios; por isso a divisao e agrupada "
        "por repositorio representativo.",
        "- Volume: a populacao e processada fora da memoria; a amostra de 200 mil conteudos e "
        "processada em lotes.",
        "",
        "## Qualidade e valores ausentes",
        "",
        markdown_table(
            top_missing,
            ["feature", "missing_count", "missing_percentage"],
            rows=10,
        ),
        "",
        "O historico de commits tem a maior ausencia. Essa ausencia representa principalmente "
        "indisponibilidade de coleta, nao o valor zero; deve ser modelada com indicador de "
        "ausencia e avaliada por ablacao.",
        "",
        "Atributos sem variacao observada devem ser removidos do conjunto preditivo:",
        "",
        markdown_table(
            constants,
            ["feature", "distinct_non_missing_values", "only_value", "missing_count"],
            rows=10,
        ),
        "",
        "## Estatistica univariada",
        "",
        "Os atributos de contagem apresentam forte assimetria positiva e caudas longas. Por "
        "isso, mediana e IQR sao mais informativos que media e desvio-padrao isoladamente, e "
        "transformacoes como log(1+x) sao candidatas naturais para modelos lineares.",
        "",
        markdown_table(
            top_outliers,
            [
                "feature",
                "median",
                "q1",
                "q3",
                "skewness",
                "outlier_percentage_1_5_iqr",
            ],
            rows=10,
        ),
        "",
        "## Relacoes com o alvo no treino",
        "",
        "A tabela abaixo ordena os atributos numericos pela magnitude da correlacao "
        "ponto-bisserial. Esses valores descrevem associacao marginal, nao causalidade e nem "
        "desempenho preditivo fora da amostra.",
        "",
        markdown_table(
            associations,
            [
                "feature",
                "negative_median",
                "positive_median",
                "standardized_mean_difference",
                "point_biserial_correlation",
            ],
            rows=12,
        ),
        "",
        "Taxa de reutilizacao por localizacao do arquivo:",
        "",
        markdown_table(location_rates, ["value", "count", "positive_rate"], rows=10),
        "",
        "Taxas de categorias com poucas observacoes sao instaveis e nao devem ser interpretadas "
        "isoladamente. A associacao de `commit_count` tambem exige cautela, pois o historico foi "
        "coletado para apenas cerca de um quarto dos conteudos.",
        "",
        "## Dependencia entre atributos",
        "",
        "As maiores correlacoes absolutas de Spearman indicam grupos redundantes de medidas "
        "de tamanho textual, estrutura Markdown e popularidade do repositorio. Isso recomenda "
        "regularizacao nos modelos lineares e analises de ablacao por grupos de atributos.",
        "",
        markdown_table(correlation, ["feature_1", "feature_2", "correlation"], rows=12),
        "",
        "## Implicacoes para a modelagem",
        "",
        "1. Preservar a prevalencia natural no teste e comparar pesos de classe apenas no treino.",
        "2. Aplicar transformacoes e imputacao dentro de cada dobra de validacao cruzada.",
        "3. Usar log(1+x) ou escalonamento robusto para contagens muito assimetricas.",
        "4. Representar ausencias com indicadores quando significarem falha ou "
        "indisponibilidade de coleta.",
        "5. Separar experimentos textuais, estruturais e contextuais para medir o ganho "
        "de cada grupo.",
        "6. Nao interpretar reutilizacao como qualidade, origem historica ou relacao causal.",
        "",
        "## Figuras produzidas",
        "",
        "- `analysis_missingness.png`: cobertura dos atributos.",
        "- `analysis_numeric_distributions.png`: histogramas por classe no treino.",
        "- `analysis_boxplots.png`: quartis e dispersao por classe.",
        "- `analysis_correlation_heatmap.png`: correlacao multivariada.",
        "- `analysis_categorical_reuse_rates.png`: frequencias condicionais categoricas.",
        "- `analysis_scatter_text_structure.png`: relacao bivariada entre texto e estrutura.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def metric_lookup(path: Path) -> dict[str, float]:
    quality = pd.read_csv(path)
    return dict(zip(quality["metric"], quality["value"], strict=True))


def run_pipeline(
    features_path: Path,
    population_quality_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    report_path: Path,
    random_seed: int,
) -> dict[str, int | float | str]:
    """Executa a EDA e salva resultados tabulares, visuais e narrativos."""
    if not features_path.is_file():
        raise FileNotFoundError(f"Features finais nao encontradas: {features_path.resolve()}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = normalize_empty_strings(pq.read_table(features_path).to_pandas())
    train = frame.loc[frame["split"] == "train"].copy()
    if len(train) == 0:
        raise RuntimeError("O subconjunto de treino esta vazio")

    catalog = attribute_catalog()
    constants = constant_feature_summary(frame)
    missing = missingness_summary(frame.drop(columns=["content_hash", "representative_repo_id"]))
    numeric = numeric_summary(train, NUMERIC_COLUMNS)
    frequencies = categorical_frequencies(frame)
    associations = numeric_target_associations(train)
    categorical_rates = categorical_target_rates(train)
    pearson = train[CORRELATION_COLUMNS].corr(method="pearson")
    spearman = train[CORRELATION_COLUMNS].corr(method="spearman")
    pairs = correlation_pairs(spearman)

    catalog.to_csv(tables_dir / "analysis_attribute_types.csv", index=False)
    constants.to_csv(tables_dir / "analysis_constant_features.csv", index=False)
    missing.to_csv(tables_dir / "analysis_missingness.csv", index=False)
    numeric.to_csv(tables_dir / "analysis_numeric_summary_train.csv", index=False)
    frequencies.to_csv(tables_dir / "analysis_categorical_frequencies.csv", index=False)
    associations.to_csv(tables_dir / "analysis_numeric_target_associations_train.csv", index=False)
    categorical_rates.to_csv(
        tables_dir / "analysis_categorical_target_rates_train.csv", index=False
    )
    pearson.to_csv(tables_dir / "analysis_pearson_correlation_train.csv")
    spearman.to_csv(tables_dir / "analysis_spearman_correlation_train.csv")
    pairs.to_csv(tables_dir / "analysis_spearman_pairs_train.csv", index=False)

    save_missingness_figure(missing, figures_dir / "analysis_missingness.png")
    save_distribution_figure(train, figures_dir / "analysis_numeric_distributions.png")
    save_boxplot_figure(train, figures_dir / "analysis_boxplots.png")
    save_correlation_figure(spearman, figures_dir / "analysis_correlation_heatmap.png")
    save_categorical_rate_figure(
        categorical_rates, figures_dir / "analysis_categorical_reuse_rates.png"
    )
    save_scatter_figure(
        train, figures_dir / "analysis_scatter_text_structure.png", random_seed
    )

    population = metric_lookup(population_quality_path)
    write_report(
        report_path,
        population,
        frame,
        train,
        numeric,
        missing,
        associations,
        pairs,
        categorical_rates,
        constants,
    )

    manifest: dict[str, int | float | str] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "features_source": str(features_path),
        "population_quality_source": str(population_quality_path),
        "sample_rows": len(frame),
        "train_rows_used_for_target_conditioned_analysis": len(train),
        "numeric_features_analyzed": len(NUMERIC_COLUMNS),
        "categorical_features_analyzed": len(CATEGORICAL_COLUMNS),
        "constant_features_found": len(constants),
        "figures_generated": 6,
        "random_seed": random_seed,
        "test_features_used_for_target_conditioned_analysis": 0,
    }
    (tables_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--population-quality", type=Path, default=DEFAULT_POPULATION_QUALITY
    )
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--random-seed", type=int, default=20261009)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(
        features_path=args.features,
        population_quality_path=args.population_quality,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        report_path=args.report,
        random_seed=args.random_seed,
    )
    print("Analise exploratoria gerada com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
