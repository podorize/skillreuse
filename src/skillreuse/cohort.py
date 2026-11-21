"""Fase 2 do protocolo (paper/protocol.md, secoes 2.3, 3 e 4): autores, divisao e atributos.

A partir das datas coletadas na fase 1, cada conteudo recebe T0 (primeira chegada entre as
copias datadas fora de agregadores), a copia de origem, o autor (dono do repositorio de origem)
e o tempo ate a primeira copia de outro dono. O alvo de cada janela W e calculado depois
(fase 3), porque W e escolhido pela regra da secao 5. O mesmo calculo e repetido com os limites
de agregador das sensibilidades (500 skills e sem exclusao).

Saida: ``data/derived/study_table.parquet`` (uma linha por conteudo datado).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from skillreuse.download import sql_path
from skillreuse.features import add_engineered_features

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATES = Path(config["dates"])
OCCURRENCES = Path(config["occurrences"])
SNAPSHOT = pd.Timestamp(config["snapshot"], tz="UTC")
STAR_CAP = config["star_cap"]
STARS = Path(config["stars"])

DEFAULT_ARTIFACTS = Path(config["default_artifacts"])
DEFAULT_REPOS = Path(config["default_repos"])

TOOL_DIRS = config["tool_dirs"]

PATH_PATTERNS = {
    **config["path_patterns"],
    "p_tool_dir": TOOL_DIRS,
}

SPLIT_SEED = config["split_seed"]
HOLDOUT_SHARE = config["holdout_share"]
CAP_PER_AUTHOR = config["cap_per_author"]
TABLE = Path(config["table"])
VARIANTS_TABLE = Path(config["variants_table"])
POPULARITY_TABLE = Path(config["popularity_table"])
MANIFEST = Path(config["manifest"])
VARIANTS = config["variants"]


def _band(n: pd.Series) -> pd.Series:
    return pd.cut(n, [0, 1, 3, 10, 50, np.inf], labels=["1", "2-3", "4-10", "11-50", "51+"])


def timing(dates: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """T0, origem, autor e tempo ate a primeira copia de outro dono, por conteudo."""
    d = dates if limit is None else dates[dates["repo_skills"] < limit]
    dated = d[d["arrival"].notna()].sort_values(["content_hash", "arrival", "repo_full_name",
                                                 "path"])
    first = dated.drop_duplicates("content_hash", keep="first")
    out = first[["content_hash", "arrival", "repo_full_name", "path", "owner",
                 "repo_created_at"]].rename(columns={
                     "arrival": "t0", "repo_full_name": "origin_repo", "path": "origin_path",
                     "owner": "author", "repo_created_at": "origin_repo_created_at"})
    other = dated.merge(out[["content_hash", "author", "t0"]], on="content_hash")
    other = other[other["owner"] != other["author"]]
    lag = (other.groupby("content_hash")["arrival"].min()
           - other.groupby("content_hash")["t0"].first()).dt.total_seconds() / 86400
    out["first_copy_lag_days"] = out["content_hash"].map(lag)
    complete = d.groupby("content_hash")["arrival"].apply(lambda s: bool(s.notna().all()))
    out["all_copies_dated"] = out["content_hash"].map(complete)
    return out


def text_features(hashes: pd.Series) -> pd.DataFrame:
    """Atributos de conteudo (secao 3), extraidos do texto da ocorrencia representativa.

    O texto e identico em todas as copias. Lido em 16 lotes pelo primeiro caractere do hash.
    """
    con = duckdb.connect()
    con.sql("pragma threads=4; pragma memory_limit='3GB'")
    con.register("sel", pd.DataFrame({"h": hashes}))
    parts = []
    for prefix in "0123456789abcdef":
        df = con.sql(f"""
            select file_sha as content_hash, content, name, description, frontmatter_valid,
                   body_chars
            from read_parquet('{sql_path(DEFAULT_ARTIFACTS)}')
            where dedup_primary = 1 and file_sha like '{prefix}%'
              and file_sha in (select h from sel where h like '{prefix}%')
        """).df()
        f = add_engineered_features(df)
        f["name_norm"] = f["name"].fillna("").str.strip().str.lower()
        parts.append(f.drop(columns=["content", "name", "description"]))
        print(f"[atributos] lote {prefix}: {len(df):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def origin_features(t: pd.DataFrame) -> pd.DataFrame:
    """Atributos do repositorio de origem fixos no tempo e estrelas em T0 (secao 3)."""
    repos = duckdb.sql(f"""select full_name as origin_repo, language as repo_language,
                           license as repo_license, is_fork as repo_is_fork
                           from read_parquet('{sql_path(DEFAULT_REPOS)}')""").df()
    out = t[["content_hash", "origin_repo", "origin_path", "t0",
             "origin_repo_created_at"]].merge(repos, on="origin_repo", how="left")
    out["repo_age_days_at_t0"] = (out["t0"] - out["origin_repo_created_at"]) \
        .dt.total_seconds() / 86400
    stars = pd.read_parquet(STARS)
    times = {r: np.sort(pd.to_datetime(pd.Series(s), utc=True).to_numpy())
             for r, s in zip(stars["repo_full_name"], stars["starred_at"], strict=True)}
    ok = set(stars.loc[stars["status"] == "ok", "repo_full_name"])

    def before(row: pd.Series) -> float:
        if row["origin_repo"] not in ok:
            return np.nan
        return float(min(np.searchsorted(times[row["origin_repo"]], row["t0"].to_datetime64()),
                         STAR_CAP))

    out["log_stars_at_t0"] = np.log1p(out.apply(before, axis=1))
    path = out["origin_path"].fillna("")
    out["path_depth"] = path.str.count("/").astype(float)
    out["path_is_root"] = path.str.fullmatch(r"(?i)skill\.md").astype(float)
    for name, pattern in PATH_PATTERNS.items():
        out[name] = path.str.contains(pattern, case=False, regex=True).astype(float)
    out["repo_is_fork"] = out["repo_is_fork"].astype(float)
    return out.drop(columns=["origin_repo", "origin_path", "t0", "origin_repo_created_at"])


def split_authors(t: pd.DataFrame) -> pd.Series:
    """Holdout = 30% dos autores de cada faixa de tamanho, na ordem do hash (secao 4)."""
    sizes = t.groupby("author").size()
    a = pd.DataFrame({"author": sizes.index, "n": sizes.to_numpy()})
    a["band"] = _band(a["n"]).astype(str)
    a["h"] = [hashlib.md5(f"{x}:{SPLIT_SEED}".encode()).hexdigest() for x in a["author"]]
    a = a.sort_values(["band", "h", "author"])
    a["rank"] = a.groupby("band").cumcount() + 1
    a["nb"] = a.groupby("band")["author"].transform("size")
    a["holdout"] = a["rank"] <= (HOLDOUT_SHARE * a["nb"]).round()
    return a.set_index("author")["holdout"]


def build() -> None:
    occ = pd.read_parquet(OCCURRENCES)
    dates = pd.read_parquet(DATES).merge(
        occ[["content_hash", "repo_full_name", "path", "owner", "repo_skills"]],
        on=["content_hash", "repo_full_name", "path"], how="right")
    main = timing(dates, 100)
    # cap de 50 por autor, na ordem do hash, antes da divisao
    main = main.sort_values(["author", "content_hash"])
    main = main[main.groupby("author").cumcount() < CAP_PER_AUTHOR].copy()
    holdout = split_authors(main)
    main["split"] = np.where(main["author"].map(holdout), "holdout", "dev")
    feats = text_features(main["content_hash"])
    table = main.merge(feats, on="content_hash", validate="1:1") \
        .merge(origin_features(main), on="content_hash", validate="1:1")
    table = table.rename(columns={"author": "owner"}).sort_values("content_hash")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE, index=False)
    # invariantes (secao 4)
    dev, hold = table[table["split"] == "dev"], table[table["split"] == "holdout"]
    sizes = main.groupby("author").size()
    share = pd.DataFrame({"band": _band(sizes).astype(str), "hold": holdout.reindex(sizes.index)})
    by_band = share.groupby("band")["hold"].mean().to_dict()
    inv = {"author_overlap": len(set(dev["owner"]) & set(hold["owner"])),
           "hash_overlap": len(set(dev["content_hash"]) & set(hold["content_hash"])),
           "max_per_author": int(table.groupby("owner").size().max()),
           "holdout_author_share_by_band": by_band}
    failed = [k for k in ("author_overlap", "hash_overlap") if inv[k]]
    if inv["max_per_author"] > CAP_PER_AUTHOR:
        failed.append("max_per_author")
    if not all(0.29 <= v <= 0.31 for b, v in by_band.items() if (share["band"] == b).sum() >= 20):
        failed.append("holdout_author_share_by_band")
    inv["failed"] = failed
    status = pd.read_parquet(DATES)["status"].value_counts().to_dict()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "phase": 2, "created_utc": datetime.now(UTC).isoformat(), "seed": SPLIT_SEED,
        "snapshot": str(SNAPSHOT.date()), "collection_status": status,
        "contents_dated": int(len(timing(dates, 100))), "contents_after_cap": len(table),
        "dev_rows": len(dev), "dev_authors": int(dev["owner"].nunique()),
        "holdout_rows": len(hold), "holdout_authors": int(hold["owner"].nunique()),
        "invariants": inv,
        "table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest()}, indent=2,
        default=str) + "\n")
    if failed:
        raise RuntimeError(f"Invariantes da fase 2 falharam: {failed}")
    print(json.dumps(inv, indent=2, default=str))


def build_variants() -> None:
    """T0 e tempo ate a primeira copia com os limites de agregador da sensibilidade (secao 8).

    Exige a coleta das copias em agregadores; nao altera a tabela principal.
    """
    occ = pd.read_parquet(OCCURRENCES)
    dates = pd.read_parquet(DATES).merge(
        occ[["content_hash", "repo_full_name", "path", "owner", "repo_skills"]],
        on=["content_hash", "repo_full_name", "path"], how="right")
    keep = pd.read_parquet(TABLE, columns=["content_hash"])
    out = keep
    for suffix, limit in VARIANTS.items():
        if suffix:
            v = timing(dates, limit)[["content_hash", "t0", "first_copy_lag_days"]]
            out = out.merge(v.rename(columns={"t0": f"t0{suffix}", "first_copy_lag_days":
                                              f"first_copy_lag_days{suffix}"}),
                            on="content_hash", how="left")
    out.to_parquet(VARIANTS_TABLE, index=False)
    print(out.notna().mean().round(3).to_dict())


def build_popularity() -> None:
    """Estrelas e forks do repositorio de origem no retrato do GitSkills (desvio, secao 13).

    A API do GitHub nao lista mais as estrelas com data, entao a popularidade em T0 nao pode ser
    reconstruida; a do retrato (posterior a T0) entra no controle como versao mais exigente.
    """
    t = pd.read_parquet(TABLE, columns=["content_hash", "origin_repo"])
    repos = duckdb.sql(f"""select full_name as origin_repo, stars, forks
                           from read_parquet('{sql_path(DEFAULT_REPOS)}')""").df()
    out = t.merge(repos, on="origin_repo", how="left")
    out["log_stars_snapshot"] = np.log1p(out["stars"].astype(float))
    out["log_forks_snapshot"] = np.log1p(out["forks"].astype(float))
    out[["content_hash", "log_stars_snapshot", "log_forks_snapshot"]].to_parquet(
        POPULARITY_TABLE, index=False)
    print(out[["stars", "forks"]].describe().round(1).to_string())


def cohort_main() -> None:
    import sys  # noqa: PLC0415

    if "--variants" in sys.argv:
        build_variants()
    elif "--popularity" in sys.argv:
        build_popularity()
    else:
        build()
