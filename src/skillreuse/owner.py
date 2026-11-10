"""Alvo por donos, atributos de contexto, auditoria do representante e EDA do novo alvo.

* ``owner_count``: numero de donos (contas/organizacoes) distintos que contem o mesmo
  conteudo; ``cross_owner = owner_count >= 2`` (reuso entre donos). Isso remove a autocopia
  entre repositorios do mesmo dono, que inflava ``repository_count``.
* Atributos de contexto do repositorio representativo (idade, recencia do push, ``log1p`` das
  variaveis de popularidade), tamanho do catalogo e indicadores de caminho.
* Auditoria da regra de escolha do representante (efeito da localizacao e do historico).
* Tudo calculado a partir das ocorrencias brutas; o teste externo nao e lido.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from skillreuse.models import DEFAULT_SAMPLE, load_count_frame  # noqa: E402

RAW_ARTIFACTS = "data/raw/gitskills-full/artifacts/*.parquet"
OWNER_TARGETS = Path("data/processed/owner_targets.parquet")
CONTEXT_FEATURES = Path("data/processed/context_features.parquet")
OUTPUT_DIR = Path("results/owner")
# data da ultima descoberta do GitSkills (coleta de julho de 2026)
REFERENCE_DATE = pd.Timestamp("2026-07-20", tz="UTC")

TOOL_DIRS = r"(?:^|/)\.(?:codex|agents|cursor|gemini|github|opencode|windsurf|kiro|roo)(?:/|$)"
PATH_PATTERNS = {
    "p_template": (
        r"(?:^|/)(?:templates?|examples?|samples?|demos?|fixtures?|tests?|testdata)(?:/|$)"
    ),
    "p_plugin": r"(?:^|/)(?:plugins?|extensions?|marketplace|packs?)(?:/|$)",
    "p_vendor": r"(?:^|/)(?:vendor|node_modules|third_party|site-packages|\.venv)(?:/|$)",
    "p_docs": r"(?:^|/)(?:docs?|documentation)(?:/|$)",
    "p_pkg": r"(?:^|/)(?:packages|apps|src|lib)(?:/|$)",
    "p_tool_dir": TOOL_DIRS,
    "p_claude_dir": r"(?:^|/)\.claude(?:/|$)",
}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("pragma threads=6; pragma memory_limit='5GB'")
    con.sql(
        "create table o as select file_sha, repo_full_name, "
        "split_part(repo_full_name,'/',1) as owner, path, location_class, dedup_primary "
        f"from read_parquet('{RAW_ARTIFACTS}')"
    )
    return con


def build_owner_targets(output: Path = OWNER_TARGETS) -> pd.DataFrame:
    """Contagens por conteudo a partir de todas as ocorrencias (1,88 milhao de conteudos)."""
    con = _connect()
    frame = con.sql(
        """
        select file_sha as content_hash,
               count(distinct repo_full_name) as repository_count,
               count(distinct owner) as owner_count,
               count(*) as occurrence_count,
               avg((location_class = 'canonical')::int) as canonical_share,
               max((location_class = 'canonical')::int) as any_canonical,
               max(case when dedup_primary = 1 then
                   case location_class when 'canonical' then 0
                        when 'skills-dir' then 1 else 2 end end) as rep_class_rank
        from o group by file_sha
        """
    ).df()
    frame["cross_owner"] = (frame["owner_count"] >= 2).astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def build_context_features(output: Path = CONTEXT_FEATURES) -> pd.DataFrame:
    """Atributos de contexto do repositorio representativo e do caminho (200 mil amostrados)."""
    sample = pq.read_table(
        DEFAULT_SAMPLE,
        columns=["content_hash", "representative_repo_id", "representative_path", "repo_owner",
                 "repo_stars", "repo_forks", "repo_created_at", "repo_pushed_at", "name"],
    ).to_pandas()
    con = _connect()
    catalog = con.sql(
        "select repo_full_name as representative_repo_id, count(distinct path) as skills_in_repo "
        "from o group by 1"
    ).df()
    frame = sample.merge(catalog, on="representative_repo_id", how="left")
    created = pd.to_datetime(frame["repo_created_at"], utc=True, errors="coerce")
    pushed = pd.to_datetime(frame["repo_pushed_at"], utc=True, errors="coerce")
    out = pd.DataFrame({"content_hash": frame["content_hash"], "repo_owner": frame["repo_owner"]})
    out["repo_age_days"] = (REFERENCE_DATE - created).dt.days.astype(float)
    out["push_recency_days"] = (REFERENCE_DATE - pushed).dt.days.astype(float)
    out["repo_active_days"] = (pushed - created).dt.days.astype(float)
    out["log_stars"] = np.log1p(frame["repo_stars"].astype(float))
    out["log_forks"] = np.log1p(frame["repo_forks"].astype(float))
    out["log_skills_in_repo"] = np.log1p(frame["skills_in_repo"].astype(float))
    out["catalog_ge_10"] = (frame["skills_in_repo"] >= 10).astype(float)
    path = frame["representative_path"].fillna("")
    out["path_depth"] = path.str.count("/").astype(float)
    out["path_is_root"] = path.str.fullmatch(r"(?i)skill\.md").astype(float)
    for name, pattern in PATH_PATTERNS.items():
        out[name] = path.str.contains(pattern, case=False, regex=True).astype(float)
    out["name_norm"] = frame["name"].fillna("").str.strip().str.lower()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    return out


def load_owner_frame(split_name: str = "train", *, need_text: bool = False) -> pd.DataFrame:
    """Quadro de modelagem: alvo de contagem existente + alvo por donos + contexto."""
    frame = load_count_frame(need_text=need_text, split_name=split_name)
    targets = pd.read_parquet(OWNER_TARGETS, columns=["content_hash", "owner_count", "cross_owner",
                                                       "canonical_share", "any_canonical"])
    context = pd.read_parquet(CONTEXT_FEATURES)
    frame = frame.merge(targets, on="content_hash", how="left", validate="1:1")
    frame = frame.merge(context, on="content_hash", how="left", validate="1:1")
    frame["log_owner_count"] = np.log1p(frame["owner_count"].astype(float))
    return frame


# ---------------------------------------------------------------------------
# Auditoria da regra de escolha do representante
# ---------------------------------------------------------------------------


def selection_audit() -> dict[str, pd.DataFrame | float]:
    """Mostra que a classe do representante cresce com o numero de copias (artefato)."""
    targets = pd.read_parquet(OWNER_TARGETS)
    targets["rep_canonical"] = (targets["rep_class_rank"] == 0).astype(float)
    edges = [0, 1, 2, 3, 5, 10, 50, np.inf]
    labels = ["1", "2", "3", "4-5", "6-10", "11-50", ">50"]

    def table(column: str) -> pd.DataFrame:
        bins = pd.cut(targets[column], edges, labels=labels)
        grouped = targets.groupby(bins, observed=True)
        return grouped.agg(
            n=("content_hash", "size"),
            p_representative_canonical=("rep_canonical", "mean"),
            p_any_copy_canonical=("any_canonical", "mean"),
            mean_share_of_copies_canonical=("canonical_share", "mean"),
        ).reset_index(names="bin")

    # regra do representante: melhor classe, depois (repo_full_name, path) minimos
    con = _connect()
    rule = con.sql(
        """
        with m as (select file_sha from o group by 1 having count(distinct repo_full_name) >= 2),
        r as (select o.*, case location_class when 'canonical' then 0
                           when 'skills-dir' then 1 else 2 end pr
              from o join m using (file_sha)),
        best as (select file_sha, min(pr) bp from r group by 1),
        cand as (select r.* from r join best b on r.file_sha = b.file_sha and r.pr = b.bp),
        pred as (select file_sha, arg_min(repo_full_name, repo_full_name || '|' || path) rep
                 from cand group by 1),
        act as (select file_sha, any_value(repo_full_name) rep, any_value(pr) pr
                from r where dedup_primary = 1 group by 1)
        select avg((act.rep = pred.rep)::int) as rule_match,
               avg((act.pr = best.bp)::int) as best_class_match
        from act join pred using (file_sha) join best using (file_sha)
        """
    ).df().iloc[0]
    return {
        "by_repository_count": table("repository_count"),
        "by_owner_count": table("owner_count"),
        "rule_match": float(rule["rule_match"]),
        "best_class_match": float(rule["best_class_match"]),
    }


# ---------------------------------------------------------------------------
# EDA do alvo por donos
# ---------------------------------------------------------------------------


def _quantiles(series: pd.Series) -> dict[str, float]:
    qs = series.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    return {f"P{int(k * 100)}": float(v) for k, v in qs.items()} | {
        "mean": float(series.mean()), "variance": float(series.var()),
        "variance_to_mean": float(series.var() / series.mean()), "max": float(series.max()),
    }


def run_eda(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = load_owner_frame("train")
    stats = pd.DataFrame(
        {"repository_count": _quantiles(dev["count_target"]),
         "owner_count": _quantiles(dev["owner_count"])}
    )
    stats.to_csv(output_dir / "eda_target_quantiles.csv")
    dist = (
        pd.crosstab(dev["owner_count"].clip(upper=11).rename("owner_count_capped_at_11"),
                    dev["count_target"].clip(upper=11).rename("repository_count_capped_at_11"))
    )
    dist.to_csv(output_dir / "eda_owner_vs_repository_count.csv")
    same_owner = float((dev["owner_count"] == 1).mean())
    owners = dev["repo_owner"].value_counts()
    conc = pd.DataFrame(
        {"rows": owners.head(10), "share": (owners.head(10) / len(dev)).round(4),
         "mean_repository_count": dev.groupby("repo_owner")["count_target"].mean()
         .reindex(owners.head(10).index).round(2),
         "mean_owner_count": dev.groupby("repo_owner")["owner_count"].mean()
         .reindex(owners.head(10).index).round(2)}
    )
    conc.to_csv(output_dir / "eda_owner_concentration.csv")
    names = dev["name_norm"]
    seen_by_owner = float(names.map(names.value_counts()).gt(1).mean())
    missing = dev[["repo_age_days", "push_recency_days", "log_stars", "log_skills_in_repo"]]
    missing.isna().mean().rename("missing_share").to_csv(output_dir / "eda_new_feature_missing.csv")
    audit = selection_audit()
    audit["by_repository_count"].to_csv(output_dir / "audit_by_repository_count.csv", index=False)
    audit["by_owner_count"].to_csv(output_dir / "audit_by_owner_count.csv", index=False)

    split = pq.read_table("data/processed/final_structured_features.parquet",
                          columns=["split", "reused"]).to_pandas()
    all_targets = pd.read_parquet(OWNER_TARGETS)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_date": str(REFERENCE_DATE.date()),
        "dev_rows": int(len(dev)),
        "dev_owners": int(dev["repo_owner"].nunique()),
        "share_same_owner_only": same_owner,
        "share_cross_owner": 1 - same_owner,
        "excess_same_owner_copies_share": float(
            1 - dev["owner_count"].sum() / dev["count_target"].sum()),
        "top1_owner_share": float(owners.iloc[0] / len(dev)),
        "top5_owner_share": float(owners.head(5).sum() / len(dev)),
        "name_empty_share": float((names == "").mean()),
        "name_shared_with_other_row_share": seen_by_owner,
        "population_contents": int(len(all_targets)),
        "population_reused_by_repo": int((all_targets["repository_count"] >= 2).sum()),
        "population_cross_owner": int((all_targets["owner_count"] >= 2).sum()),
        "development_contents_in_split": int((split["split"] == "train").sum()),
        "representative_rule_match": audit["rule_match"],
        "representative_best_class_match": audit["best_class_match"],
        "quantiles": stats.to_dict(),
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _figures(dev, audit, output_dir)
    return summary


def _figures(dev: pd.DataFrame, audit: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.arange(0.5, 21.5)
    ax[0].hist(dev["count_target"].clip(upper=20), bins=bins, alpha=0.6, label="repositórios")
    ax[0].hist(dev["owner_count"].clip(upper=20), bins=bins, alpha=0.6, label="donos")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("contagem (truncada em 20)")
    ax[0].set_ylabel("conteúdos")
    ax[0].legend()
    ax[0].set_title("Repositórios vs. donos")
    table = audit["by_repository_count"]
    ax[1].plot(table["bin"], table["p_representative_canonical"], "o-",
               label="representante é canonical")
    ax[1].plot(table["bin"], table["mean_share_of_copies_canonical"], "s--",
               label="fração das cópias em canonical")
    ax[1].set_xlabel("repositórios com o conteúdo")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=8)
    ax[1].set_title("Efeito da regra do representante")
    fig.tight_layout()
    fig.savefig(out / "eda_owner_and_selection.png", dpi=160)
    plt.close(fig)
    owners = dev["repo_owner"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(owners.index[::-1], owners.values[::-1])
    ax.set_xlabel("skills reutilizadas no desenvolvimento")
    ax.set_title("Concentração por dono")
    fig.tight_layout()
    fig.savefig(out / "eda_owner_concentration.png", dpi=160)
    plt.close(fig)


def owner_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def owner_main() -> None:
    args = owner_parse_args()
    if not args.skip_build:
        build_owner_targets()
        build_context_features()
    print(json.dumps(run_eda(args.output_dir), indent=2, ensure_ascii=False, default=str)[:3500])
