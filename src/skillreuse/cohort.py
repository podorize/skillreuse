"""Phase 2 of the protocol (paper/protocol.md, sections 2.3, 3 and 4): authors, split, attributes.

From the dates collected in phase 1 (``collect.py``), this module builds the table of the
study, with one row per skill. For each skill it computes:

* T0: the first arrival among the dated copies outside aggregators (first appearance);
* origin: the copy that arrived at T0, and author: the owner of the origin repository;
* time to first copy: days from T0 until the first copy by another owner;
* the attributes: 27 of the text (``features.py``) and 14 of the origin repository;
* the split: the whole author goes either to the development set or to the holdout.

The target ("copied within W days") is *not* in the table. It is computed from the time to
first copy in ``study.load_frame``, because the window W is chosen later, in phase 3. The
same computation is repeated with the aggregator limits of the robustness analysis (500
skills, and no exclusion of aggregators).

Output: ``data/derived/study_table.parquet`` (one row per dated skill).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from skillreuse.collect import DATES, OCCURRENCES, SNAPSHOT, STAR_CAP, STARS
from skillreuse.config import CONFIG
from skillreuse.download import DEFAULT_ARTIFACTS, DEFAULT_REPOS, sql_path
from skillreuse.features import add_engineered_features

# Attributes of the origin repository taken from the path of the file. Each one is 1 when
# the path has a folder of that kind and 0 otherwise (config/config.yaml, "cohort").
PATH_PATTERNS = CONFIG["cohort"]["folder_patterns"]

# The split by author: its seed, the share of authors in the holdout and the cap of skills.
SPLIT_SEED = CONFIG["cohort"]["split_seed"]
HOLDOUT_SHARE = CONFIG["cohort"]["holdout_share"]
CAP_PER_AUTHOR = CONFIG["cohort"]["cap_per_author"]
# Aggregator limit of each variant: "" is the main analysis (100 skills), "_agg500" uses
# 500 skills and "_aggnone" does not exclude any aggregator (None = no limit).
VARIANTS = CONFIG["cohort"]["aggregator_variants"]

TEXT_ATTRIBUTES = Path(CONFIG["paths"]["text_attributes"])
REPOSITORIES = Path(CONFIG["paths"]["repositories"])
TABLE = Path(CONFIG["paths"]["study_table"])
VARIANTS_TABLE = Path(CONFIG["paths"]["aggregator_variants"])
POPULARITY_TABLE = Path(CONFIG["paths"]["origin_popularity"])
MANIFEST = Path(CONFIG["paths"]["table_manifest"])


def size_band(n: pd.Series) -> pd.Series:
    """Size band of an author, from the number of skills: 1, 2-3, 4-10, 11-50 or 51+."""
    return pd.cut(n, [0, 1, 3, 10, 50, np.inf], labels=["1", "2-3", "4-10", "11-50", "51+"])


def timing(dates: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """T0, origin, author and time to first copy of each skill.

    ``dates`` has one row per copy. Copies in repositories with ``limit`` or more skills
    (aggregators) are left out; ``None`` keeps all of them. For each skill:

    * the dated copy that arrived first is the origin; its arrival is T0 and its owner is the
      author. Ties are broken by repository name and then by path;
    * ``first_copy_lag_days`` is the number of days from T0 to the first copy by another
      owner (empty if there is none). Copies by the author do not count;
    * ``all_copies_dated`` is True if every copy could be dated.
    """
    copies = dates if limit is None else dates[dates["repo_skills"] < limit]
    dated = copies[copies["arrival"].notna()].sort_values(
        ["content_hash", "arrival", "repo_full_name", "path"])

    # the first dated copy of each skill is its origin
    origin = dated.drop_duplicates("content_hash", keep="first")
    result = origin[["content_hash", "arrival", "repo_full_name", "path", "owner",
                     "repo_created_at"]].rename(columns={
        "arrival": "t0", "repo_full_name": "origin_repo", "path": "origin_path",
        "owner": "author", "repo_created_at": "origin_repo_created_at"})

    # days from T0 to the first copy by an owner other than the author
    others = dated.merge(result[["content_hash", "author", "t0"]], on="content_hash")
    others = others[others["owner"] != others["author"]]
    first_other_copy = others.groupby("content_hash")["arrival"].min()
    t0 = others.groupby("content_hash")["t0"].first()
    lag_days = (first_other_copy - t0).dt.total_seconds() / 86400
    result["first_copy_lag_days"] = result["content_hash"].map(lag_days)

    every_copy_dated = copies.groupby("content_hash")["arrival"].apply(
        lambda arrivals: bool(arrivals.notna().all()))
    result["all_copies_dated"] = result["content_hash"].map(every_copy_dated)
    return result


def text_features(hashes: pd.Series) -> pd.DataFrame:
    """Attributes of the text (section 3) of the skills with the given hashes.

    The text is read from the representative occurrence of each skill (``dedup_primary``
    is 1). It is the same in all copies, so it does not matter which copy is read. The
    file is large, so it is read in 16 batches, by the first character of the hash.
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
        # Normalized name, used only by the "unseen names" robustness check.
        f["name_norm"] = f["name"].fillna("").str.strip().str.lower()
        parts.append(f.drop(columns=["content", "name", "description"]))
        print(f"[attributes] batch {prefix}: {len(df):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def export_gitskills() -> None:
    """Save what the study table needs from GitSkills (command ``skillreuse cohort --gitskills``).

    For the sampled skills only: the text attributes of each skill (``TEXT_ATTRIBUTES``) and
    the language, license, fork flag, stars and forks of each repository (``REPOSITORIES``).
    These two small files are published, so the table can be built without GitSkills.
    """
    occ = pd.read_parquet(OCCURRENCES, columns=["content_hash", "repo_full_name"])
    attributes = text_features(occ["content_hash"].drop_duplicates())
    attributes.sort_values("content_hash").to_parquet(TEXT_ATTRIBUTES, index=False)
    repos = duckdb.sql(f"""select full_name as repo_full_name, language, license, is_fork,
                           stars, forks from read_parquet('{sql_path(DEFAULT_REPOS)}')""").df()
    repos = repos[repos["repo_full_name"].isin(occ["repo_full_name"])]
    repos.sort_values("repo_full_name").to_parquet(REPOSITORIES, index=False)
    print(f"{len(attributes):,} skills and {len(repos):,} repositories saved")


def origin_features(t: pd.DataFrame) -> pd.DataFrame:
    """Attributes of the origin repository (section 3), used only by the control model.

    ``t`` is the table of ``timing``. The attributes are: age of the repository at T0,
    language, license, fork flag, stars at T0 (log scale) and the position of the file
    (depth, file at the root and the seven folder kinds of ``PATH_PATTERNS``).
    """
    repos = pd.read_parquet(REPOSITORIES, columns=["repo_full_name", "language", "license",
                                                    "is_fork"])
    repos.columns = ["origin_repo", "repo_language", "repo_license", "repo_is_fork"]
    out = t[["content_hash", "origin_repo", "origin_path", "t0",
             "origin_repo_created_at"]].merge(repos, on="origin_repo", how="left")
    # Negative when T0 is before the creation of the repository (inherited history).
    out["repo_age_days_at_t0"] = (out["t0"] - out["origin_repo_created_at"]) \
        .dt.total_seconds() / 86400
    stars = pd.read_parquet(STARS)
    times = {r: np.sort(pd.to_datetime(pd.Series(s), utc=True).to_numpy())
             for r, s in zip(stars["repo_full_name"], stars["starred_at"], strict=True)}
    ok = set(stars.loc[stars["status"] == "ok", "repo_full_name"])

    def before(row: pd.Series) -> float:
        """Number of stars given before T0 (at most ``STAR_CAP``), or missing if unknown."""
        if row["origin_repo"] not in ok:
            return np.nan
        return float(min(np.searchsorted(times[row["origin_repo"]], row["t0"].to_datetime64()),
                         STAR_CAP))

    # The API returned no dates for the stars, so this column is zero for every skill and
    # is removed later as a constant column (protocol, section 13, deviation 3).
    out["log_stars_at_t0"] = np.log1p(out.apply(before, axis=1))
    path = out["origin_path"].fillna("")
    out["path_depth"] = path.str.count("/").astype(float)
    out["path_is_root"] = path.str.fullmatch(r"(?i)skill\.md").astype(float)
    for name, pattern in PATH_PATTERNS.items():
        out[name] = path.str.contains(pattern, case=False, regex=True).astype(float)
    out["repo_is_fork"] = out["repo_is_fork"].astype(float)
    return out.drop(columns=["origin_repo", "origin_path", "t0", "origin_repo_created_at"])


def split_authors(t: pd.DataFrame) -> pd.Series:
    """Choose the authors of the holdout: 30% of the authors of each size band (section 4).

    The split is by author: all skills of an author go to the same side, so the model is
    tested on developers it has never seen. Inside each band (1, 2-3, 4-10, 11-50 skills),
    authors are ordered by ``md5(author + ':' + seed)`` and the first 30% go to the holdout,
    which gives both sides the same mix of small and large accounts.

    Returns a series indexed by author, True for the holdout authors.
    """
    skills_per_author = t.groupby("author").size()
    authors = pd.DataFrame({"author": skills_per_author.index, "n": skills_per_author.to_numpy()})
    authors["band"] = size_band(authors["n"]).astype(str)
    authors["hash"] = [hashlib.md5(f"{author}:{SPLIT_SEED}".encode()).hexdigest()
                       for author in authors["author"]]
    authors = authors.sort_values(["band", "hash", "author"])
    authors["position_in_band"] = authors.groupby("band").cumcount() + 1
    authors["authors_in_band"] = authors.groupby("band")["author"].transform("size")
    holdout_size = (HOLDOUT_SHARE * authors["authors_in_band"]).round()
    authors["holdout"] = authors["position_in_band"] <= holdout_size
    return authors.set_index("author")["holdout"]


def build() -> None:
    """Build the table of the study and check its invariants (command ``skillreuse cohort``).

    Steps: join the dates with the occurrences, compute ``timing`` with the main
    aggregator limit, keep at most 50 skills per author, split by author, add the text and
    origin attributes, write the table and the manifest. The run fails if an invariant of
    section 4 breaks: no author or content on both sides, at most 50 skills per author, and
    29% to 31% of the authors in the holdout in each size band.
    """
    occ = pd.read_parquet(OCCURRENCES)
    dates = pd.read_parquet(DATES).merge(
        occ[["content_hash", "repo_full_name", "path", "owner", "repo_skills"]],
        on=["content_hash", "repo_full_name", "path"], how="right")
    main = timing(dates, 100)
    # Cap of 50 skills per author (the first ones in hash order), before the split.
    main = main.sort_values(["author", "content_hash"])
    main = main[main.groupby("author").cumcount() < CAP_PER_AUTHOR].copy()
    holdout = split_authors(main)
    main["split"] = np.where(main["author"].map(holdout), "holdout", "dev")
    feats = pd.read_parquet(TEXT_ATTRIBUTES)
    feats = feats[feats["content_hash"].isin(main["content_hash"])]
    table = main.merge(feats, on="content_hash", validate="1:1") \
        .merge(origin_features(main), on="content_hash", validate="1:1")
    table = table.rename(columns={"author": "owner"}).sort_values("content_hash")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE, index=False)
    # Invariants of the split (section 4).
    dev, hold = table[table["split"] == "dev"], table[table["split"] == "holdout"]
    sizes = main.groupby("author").size()
    share = pd.DataFrame({"band": size_band(sizes).astype(str),
                          "hold": holdout.reindex(sizes.index)})
    by_band = share.groupby("band")["hold"].mean().to_dict()
    inv = {"author_overlap": len(set(dev["owner"]) & set(hold["owner"])),
           "hash_overlap": len(set(dev["content_hash"]) & set(hold["content_hash"])),
           "max_per_author": int(table.groupby("owner").size().max()),
           "holdout_author_share_by_band": by_band}
    failed = [k for k in ("author_overlap", "hash_overlap") if inv[k]]
    if inv["max_per_author"] > CAP_PER_AUTHOR:
        failed.append("max_per_author")
    # Bands with fewer than 20 authors are too small to check the share.
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
        raise RuntimeError(f"Phase 2 invariants failed: {failed}")
    print(json.dumps(inv, indent=2, default=str))


def build_variants() -> None:
    """T0 and time to first copy with the aggregator limits of the robustness analysis.

    Needs the copies in aggregators to be dated (``collect dates --scope aggregators``).
    Writes ``aggregator_variants.parquet`` and does not change the main table.
    """
    occ = pd.read_parquet(OCCURRENCES)
    dates = pd.read_parquet(DATES).merge(
        occ[["content_hash", "repo_full_name", "path", "owner", "repo_skills"]],
        on=["content_hash", "repo_full_name", "path"], how="right")
    keep = pd.read_parquet(TABLE, columns=["content_hash"])
    out = keep
    for suffix, limit in VARIANTS.items():
        if suffix:  # the main variant ("") is already in the study table
            v = timing(dates, limit)[["content_hash", "t0", "first_copy_lag_days"]]
            out = out.merge(v.rename(columns={"t0": f"t0{suffix}", "first_copy_lag_days":
                                              f"first_copy_lag_days{suffix}"}),
                            on="content_hash", how="left")
    out.to_parquet(VARIANTS_TABLE, index=False)
    print(out.notna().mean().round(3).to_dict())


def build_popularity() -> None:
    """Stars and forks of the origin repository at the snapshot (protocol deviation).

    The GitHub API no longer lists stars with their dates, so the popularity at T0 could
    not be rebuilt. The popularity at the snapshot, which is later than T0, is used
    instead in the control model. This makes the control harder to beat: the stars may
    even be partly caused by the copies. See protocol, section 13, deviation 3.
    """
    t = pd.read_parquet(TABLE, columns=["content_hash", "origin_repo"])
    repos = pd.read_parquet(REPOSITORIES, columns=["repo_full_name", "stars", "forks"])
    repos = repos.rename(columns={"repo_full_name": "origin_repo"})
    out = t.merge(repos, on="origin_repo", how="left")
    out["log_stars_snapshot"] = np.log1p(out["stars"].astype(float))
    out["log_forks_snapshot"] = np.log1p(out["forks"].astype(float))
    out[["content_hash", "log_stars_snapshot", "log_forks_snapshot"]].to_parquet(
        POPULARITY_TABLE, index=False)
    print(out[["stars", "forks"]].describe().round(1).to_string())


def cohort_main() -> None:
    """Command ``skillreuse cohort [--gitskills | --popularity | --variants]``.

    Without options it builds the table of the study. ``--gitskills`` first saves what the
    table needs from GitSkills; the other two options build the extra tables used by the
    control model and by the robustness analysis.
    """
    import sys  # noqa: PLC0415

    if "--gitskills" in sys.argv:
        export_gitskills()
    elif "--variants" in sys.argv:
        build_variants()
    elif "--popularity" in sys.argv:
        build_popularity()
    else:
        build()
