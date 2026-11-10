"""Holdout novo, disjunto por dono, para a confirmacao (paper/protocol.md, secao 8).

Populacao: conteudos fora da amostra de 200 mil, com ``repository_count >= 2``, cujo dono do
repositorio representativo nao aparece entre as skills reutilizadas da amostra (desenvolvimento
e teste original). Limite de 50 conteudos por dono; donos escolhidos por ``md5(dono || seed)``
ate atingir o tamanho alvo. O holdout so e avaliado depois de o modelo final ser congelado
(``skillreuse final-model``), uma unica vez (``skillreuse holdout-eval``).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from skillreuse.data import DEFAULT_ARTIFACTS, DEFAULT_REPOS, DEFAULT_SAMPLE, sql_path
from skillreuse.features import write_structured_features_batched
from skillreuse.owner import (
    OWNER_TARGETS,
    _connect,
    build_context_features,
    build_owner_targets,
)

HOLDOUT_SAMPLE = Path("data/processed/holdout_sample.parquet")
HOLDOUT_FEATURES = Path("data/processed/holdout_structured_features.parquet")
HOLDOUT_CONTEXT = Path("data/processed/holdout_context_features.parquet")
HOLDOUT_MANIFEST = Path("results/owner/holdout/holdout_manifest.json")
HOLDOUT_SEED = "20261110"


def select_holdout(target_rows: int, cap: int, seed: str = HOLDOUT_SEED) -> pd.DataFrame:
    if not OWNER_TARGETS.is_file():
        build_owner_targets()
    sample = pq.read_table(
        DEFAULT_SAMPLE, columns=["content_hash", "repository_count", "repo_owner"]
    ).to_pandas()
    excluded_owners = pd.DataFrame(
        {"owner": sorted(set(sample.loc[sample["repository_count"] >= 2, "repo_owner"].dropna()))}
    )
    sample_hashes = sample[["content_hash"]]
    con = _connect()
    con.register("excluded_owners", excluded_owners)
    con.register("sample_hashes", sample_hashes)
    con.sql(f"create table targets as select * from read_parquet('{sql_path(OWNER_TARGETS)}')")
    selected = con.sql(
        f"""
        with cand as (
            select o.file_sha as content_hash, o.owner
            from o join targets t on t.content_hash = o.file_sha
            where o.dedup_primary = 1 and t.repository_count >= 2
              and o.file_sha not in (select content_hash from sample_hashes)
              and o.owner not in (select owner from excluded_owners)
        ),
        capped as (
            select *, row_number() over (partition by owner order by content_hash) as rn
            from cand
        ),
        kept as (select content_hash, owner from capped where rn <= {int(cap)}),
        owners as (
            select owner, count(*) as n, md5(owner || ':{seed}') as h from kept group by owner
        ),
        ranked as (
            select owner, n, sum(n) over (order by h rows between unbounded preceding and
                   current row) as cum from owners
        )
        select kept.content_hash, kept.owner
        from kept join ranked using (owner)
        where ranked.cum - ranked.n < {int(target_rows)}
        order by kept.content_hash
        """
    ).df()
    return selected


def build_holdout_sample(selected: pd.DataFrame, output: Path = HOLDOUT_SAMPLE) -> None:
    con = _connect()
    con.register("hold", selected[["content_hash"]])
    con.sql(f"create table targets as select * from read_parquet('{sql_path(OWNER_TARGETS)}')")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT t.content_hash, t.occurrence_count, t.repository_count, 1 AS reused,
                   r.repo_full_name AS representative_repo_id, r.path AS representative_path,
                   r.filename, r.location_class, r.content, r.frontmatter_valid, r.name,
                   r.description, r.body_chars, r.sibling_count, r.sibling_bytes, r.has_scripts,
                   r.has_references, r.composition_fetched, r.composition_truncated,
                   r.history_fetched, r.first_commit_at, r.last_commit_at, r.commit_count,
                   repos.owner AS repo_owner, repos.stars AS repo_stars,
                   repos.forks AS repo_forks, repos.is_fork AS repo_is_fork,
                   repos.language AS repo_language, repos.license AS repo_license,
                   repos.created_at AS repo_created_at, repos.pushed_at AS repo_pushed_at,
                   repos.metadata_fetched AS repo_metadata_fetched
            FROM hold JOIN targets t ON t.content_hash = hold.content_hash
            JOIN read_parquet('{sql_path(DEFAULT_ARTIFACTS)}') r
              ON r.file_sha = t.content_hash AND r.dedup_primary = 1
            LEFT JOIN read_parquet('{sql_path(DEFAULT_REPOS)}') repos
              ON repos.full_name = r.repo_full_name
            ORDER BY t.content_hash
        ) TO '{sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    tmp.replace(output)


def holdout_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-rows", type=int, default=25_000)
    parser.add_argument("--cap-per-owner", type=int, default=50)
    parser.add_argument("--seed", default=HOLDOUT_SEED)
    return parser.parse_args()


def holdout_main() -> None:
    args = holdout_parse_args()
    selected = select_holdout(args.target_rows, args.cap_per_owner, args.seed)
    build_holdout_sample(selected)
    assignments = selected[["content_hash"]].assign(split="holdout")
    write_structured_features_batched(HOLDOUT_SAMPLE, HOLDOUT_FEATURES, assignments, 2000)
    build_context_features(HOLDOUT_CONTEXT, HOLDOUT_SAMPLE)

    sample = pq.read_table(DEFAULT_SAMPLE, columns=["content_hash", "repository_count",
                                                    "repo_owner"]).to_pandas()
    dev_owners = set(sample.loc[sample["repository_count"] >= 2, "repo_owner"].dropna())
    context = pd.read_parquet(HOLDOUT_CONTEXT, columns=["content_hash", "repo_owner"])
    targets = pd.read_parquet(OWNER_TARGETS, columns=["content_hash", "owner_count"])
    check = context.merge(targets, on="content_hash")
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "paper/protocol.md secao 8",
        "seed": args.seed,
        "target_rows": args.target_rows,
        "cap_per_owner": args.cap_per_owner,
        "rows": int(len(check)),
        "owners": int(check["repo_owner"].nunique()),
        "owner_overlap_with_sample_reused": len(set(check["repo_owner"]) & dev_owners),
        "hash_overlap_with_200k_sample": int(
            check["content_hash"].isin(sample["content_hash"]).sum()
        ),
        "max_rows_per_owner": int(check["repo_owner"].value_counts().max()),
        "sanity_cross_owner_prevalence": float((check["owner_count"] >= 2).mean()),
        "note": "prevalencia consultada apenas como verificacao de integridade; "
        "o modelo nao foi avaliado no holdout",
    }
    HOLDOUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    _ = pa  # pyarrow usado indiretamente pelo gravador de atributos
