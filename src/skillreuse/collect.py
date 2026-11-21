"""Phase 1 of the protocol (paper/protocol.md, sections 2.1 and 2.2): sample and dates.

GitSkills is a snapshot of GitHub taken in July 2026: it says *where* each SKILL.md was,
but not *when* it got there. This module adds the dates. The steps, in order:

1. ``sample``: choose 150,000 skills and list all their copies ("occurrences").
2. ``dates``: ask the GitHub API when each copy arrived in its repository.
3. ``stars``: ask the API for the stars of the origin repositories (the API returned no
   dates, so this was later replaced by the popularity at the snapshot; see protocol,
   section 13).
4. ``merge``: join the partial results into one table of dates.

Vocabulary used here and in the rest of the package:

* skill: a distinct content, identified by the hash of the file (``file_sha``).
* occurrence (or copy): one file in one repository with the content of a skill.
* aggregator: a repository with 100 or more distinct skills. Registries, mirrors and
  marketplaces collect skills in bulk, so a copy there does not show that a developer
  chose the skill.
* population: skills with at least one occurrence outside aggregators.
* arrival: the date a copy appeared in its repository (defined in ``date_batch``
  and ``merge_dates``).

Outputs go to ``data/derived/collection/``. The progress is saved in parts, so the
collection can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from skillreuse.download import DEFAULT_ARTIFACTS, sql_path

# A repository with at least this many distinct skills is an aggregator.
AGGREGATOR_MIN_SKILLS = 100
# Seed of the sample: skills are sorted by md5(hash + ':' + seed) and the first are kept.
SAMPLE_SEED = "20261121"
SAMPLE_SIZE = 150_000
# Start of the GitSkills crawl. A skill is eligible only if its whole window ends before it.
SNAPSHOT = pd.Timestamp("2026-07-09", tz="UTC")
# End of the GitSkills crawl (20 July). A copy seen in the snapshot cannot have arrived later.
CRAWL_END = pd.Timestamp("2026-07-21", tz="UTC")
STAR_CAP = 10_000  # stars are counted up to this number
MAX_COMMITS = 500  # most recent commits read in the history of one file
HISTORY_PAGE = 20  # commits per GraphQL page

COLLECTION = Path("data/derived/collection")
OCCURRENCES = COLLECTION / "sample_occurrences.parquet"
DATES_DIR = COLLECTION / "dates_parts"
DATES = COLLECTION / "dates.parquet"
STARS = COLLECTION / "origin_stars.parquet"
MANIFEST = Path("results/phase1/manifest.json")
API = "https://api.github.com/graphql"


# ---------------------------------------------------------------------------
# Step 1: the sample
# ---------------------------------------------------------------------------

def build_sample() -> tuple[pd.DataFrame, dict[str, int]]:
    """Choose the sample and list every occurrence of its skills.

    Returns the occurrences (one row per file in a repository) and a dictionary with the
    counts written to the phase 1 manifest. The work is done in SQL (DuckDB) because the
    GitSkills table is too large for memory.
    """
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    con.sql("pragma threads=4; pragma memory_limit='3GB'")
    # o: every file of GitSkills (h = hash of the content, r = repository)
    con.sql(f"""create table o as select file_sha as h, repo_full_name as r,
                split_part(repo_full_name, '/', 1) as owner, path, filename, dedup_primary
                from read_parquet('{sql_path(DEFAULT_ARTIFACTS)}')""")
    # rs: number of distinct skills in each repository (>= 100 means aggregator)
    con.sql("create table rs as select r, count(distinct h) as repo_skills from o group by r")
    # pop: the population, skills found in at least one repository that is not an aggregator
    con.sql(f"""create table pop as select h from o join rs using (r)
                group by h having bool_or(repo_skills < {AGGREGATOR_MIN_SKILLS})""")
    # sample: the first SAMPLE_SIZE skills in a pseudo-random but reproducible order
    con.sql(f"""create table sample as select h from pop
                order by md5(h || ':{SAMPLE_SEED}'), h limit {SAMPLE_SIZE}""")
    # occurrences: every copy of the sampled skills, with a flag for aggregators
    occ = con.sql(f"""
        select o.h as content_hash, o.r as repo_full_name, o.owner, o.path, o.filename,
               o.dedup_primary, rs.repo_skills,
               rs.repo_skills >= {AGGREGATOR_MIN_SKILLS} as aggregator
        from o join rs using (r) where o.h in (select h from sample)
        order by o.h, o.r, o.path
    """).df()
    counts = {"population_contents": int(con.sql("select count(*) from pop").fetchone()[0]),
              "sample_contents": int(occ["content_hash"].nunique()),
              "occurrences": len(occ), "occurrences_outside_aggregators":
              int((~occ["aggregator"]).sum()), "occurrences_in_aggregators":
              int(occ["aggregator"].sum())}
    return occ, counts


# ---------------------------------------------------------------------------
# GitHub GraphQL client: retries, waiting for the rate limit and parallel calls
# ---------------------------------------------------------------------------

class Client:
    """Minimal client of the GitHub GraphQL API, safe to share between threads.

    The token comes from the GitHub CLI (``gh auth token``). Every query is tried up to
    4 times (1 attempt + 3 retries) with growing waits. When the remaining quota gets low,
    all threads pause until the quota is reset.
    """

    def __init__(self) -> None:
        self.token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                                    check=True).stdout.strip()
        self.lock = threading.Lock()
        self.cost = 0  # total cost of the queries so far, in GitHub rate-limit points
        self.pause_until = 0.0  # time (seconds since the epoch) until which all threads wait

    def query(self, q: str, variables: dict) -> dict:
        """Run one query. After the last failed attempt, return a ``COLLECTION_FAILED`` error."""
        for attempt in range(4):  # 1 attempt + 3 retries
            wait = self.pause_until - time.time()
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(API, data=json.dumps({"query": q, "variables": variables})
                                         .encode(), headers={
                                             "Authorization": f"bearer {self.token}",
                                             "Content-Type": "application/json",
                                             "User-Agent": "skillreuse-collection"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    res = json.loads(r.read())
            except urllib.error.HTTPError as e:
                retry = e.headers.get("retry-after")
                time.sleep(int(retry) if retry else 10 * 2 ** attempt)
                continue
            except (urllib.error.URLError, http.client.HTTPException, OSError,
                    json.JSONDecodeError):
                time.sleep(5 * 2 ** attempt)
                continue
            rl = (res.get("data") or {}).get("rateLimit") or {}
            with self.lock:
                self.cost += rl.get("cost", 0)
                if rl.get("remaining", 5000) < 300:  # wait for the quota to be reset
                    reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
                    self.pause_until = reset.timestamp() + 5
            if res.get("data") is None and res.get("errors"):
                time.sleep(5 * 2 ** attempt)
                continue
            return res
        return {"data": None, "errors": [{"type": "COLLECTION_FAILED"}]}


# Part of the query that reads the history of one file. {i} is the position of the
# occurrence in the batch, so one query can ask about many repositories at once. The
# doubled braces are literal braces, because the text goes through str.format. For each
# commit, ``file(path).oid`` is the git hash of the file at that commit, which is the same
# value as the ``file_sha`` of GitSkills, so a commit "has the skill" when the two match.
HISTORY = """createdAt defaultBranchRef {{ target {{ ... on Commit {{
  history(first: {page}, path: $p{i}, after: $c{i}) {{ totalCount
    pageInfo {{ hasNextPage endCursor }} nodes {{ committedDate file(path: $p{i}) {{ oid }} }} }}
}} }} }}"""


# ---------------------------------------------------------------------------
# Step 2: dates of the copies
# ---------------------------------------------------------------------------

def date_batch(client: Client, rows: list[dict]) -> list[dict]:
    """Date a batch of occurrences: when did each copy first have the content of the skill?

    For every occurrence, read the history of the file on the default branch (newest commit
    first, ``HISTORY_PAGE`` commits per query, up to ``MAX_COMMITS``) and keep the earliest
    commit whose file has exactly the hash of the skill. Each occurrence ends with a status:

    * ``datada``: dated (a commit with the hash was found);
    * ``repositorio_inacessivel``: the repository is gone or private;
    * ``arquivo_ausente``: the repository exists but the file has no history;
    * ``hash_nao_encontrado``: the file has history but no commit with this exact content;
    * ``falha_de_coleta``: the query failed after all retries.

    These Portuguese codes are stored in the data files, so they are kept as they are.
    """
    # Reading state of each occurrence of the batch.
    state = {i: {"nodes": [], "cursor": None, "status": None, "created": None, "commits": None}
             for i in range(len(rows))}
    pending = list(state)  # occurrences that still have more pages to read
    for _page in range(MAX_COMMITS // HISTORY_PAGE):
        if not pending:
            break
        # Build one query with an alias (f0, f1, ...) for each pending occurrence.
        decl, parts, var = [], [], {}
        for i in pending:
            owner, name = rows[i]["repo_full_name"].split("/", 1)
            var.update({f"o{i}": owner, f"n{i}": name, f"p{i}": rows[i]["path"],
                        f"c{i}": state[i]["cursor"]})
            decl += [f"$o{i}: String!", f"$n{i}: String!", f"$p{i}: String!", f"$c{i}: String"]
            parts.append(f"f{i}: repository(owner: $o{i}, name: $n{i}) {{ "
                         + HISTORY.format(page=HISTORY_PAGE, i=i) + " }")
        q = "query(" + ", ".join(decl) + ") { rateLimit { cost remaining resetAt } " \
            + " ".join(parts) + " }"
        res = client.query(q, var)
        data = res.get("data")
        nxt = []  # occurrences that need another page
        for i in pending:
            if data is None:
                state[i]["status"] = "falha_de_coleta"
                continue
            repo = data.get(f"f{i}")
            if repo is None:
                state[i]["status"] = "repositorio_inacessivel"
                continue
            state[i]["created"] = repo["createdAt"]
            hist = ((repo.get("defaultBranchRef") or {}).get("target") or {}).get("history")
            if not hist:
                state[i]["status"] = "arquivo_ausente"
                continue
            state[i]["commits"] = hist["totalCount"]
            state[i]["nodes"] += hist["nodes"]
            if hist["pageInfo"]["hasNextPage"]:
                state[i]["cursor"] = hist["pageInfo"]["endCursor"]
                nxt.append(i)
        pending = nxt
    # Decide the final status and the date of each occurrence.
    out = []
    for i, r in enumerate(rows):
        s = state[i]
        # Dates of the commits in which the file has exactly the hash of the skill.
        match = [n["committedDate"] for n in s["nodes"]
                 if (n.get("file") or {}).get("oid") == r["content_hash"]]
        status = s["status"] or ("datada" if match else
                                 "arquivo_ausente" if not s["nodes"] else "hash_nao_encontrado")
        out.append({"content_hash": r["content_hash"], "repo_full_name": r["repo_full_name"],
                    "path": r["path"], "status": status, "repo_created_at": s["created"],
                    # ISO 8601 dates sort correctly as text, so min() gives the earliest.
                    "first_hash_commit_at": min(match) if match else None,
                    "commits_on_path": s["commits"], "commits_read": len(s["nodes"])})
    return out


def collect_dates(scope: str, batch: int, workers: int, limit: int | None = None) -> None:
    """Date every occurrence of one scope, in parallel, saving the progress in parts.

    ``scope`` is ``main`` (copies outside aggregators) or ``aggregators`` (used only in the
    robustness analysis). Occurrences already present in ``dates_parts/`` are skipped, so
    the command can be stopped and run again. Results are written every 5,000 rows.
    """
    occ = pd.read_parquet(OCCURRENCES)
    occ = occ[occ["aggregator"] == (scope == "aggregators")]
    DATES_DIR.mkdir(parents=True, exist_ok=True)
    # An occurrence is identified by (repository, path, content hash).
    done = set()
    for part in DATES_DIR.glob("*.parquet"):
        d = pd.read_parquet(part, columns=["repo_full_name", "path", "content_hash"])
        done |= set(zip(d["repo_full_name"], d["path"], d["content_hash"], strict=True))
    todo = [r for r in occ.to_dict("records")
            if (r["repo_full_name"], r["path"], r["content_hash"]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[collect] {scope}: {len(todo):,} occurrences to date ({len(done):,} already done)",
          flush=True)
    client, t0, buf, n_done = Client(), time.time(), [], 0
    lock = threading.Lock()
    part_id = len(list(DATES_DIR.glob("*.parquet")))

    def flush(force: bool = False) -> None:
        """Write the buffered results as a new part file (when it has 5,000 rows or on force)."""
        nonlocal buf, part_id
        if buf and (force or len(buf) >= 5000):
            pd.DataFrame(buf).to_parquet(DATES_DIR / f"{scope}_{part_id:05d}.parquet",
                                         index=False)
            part_id += 1
            buf = []

    def work(chunk: list[dict]) -> None:
        """Date one batch of occurrences (runs in a worker thread)."""
        nonlocal n_done
        res = date_batch(client, chunk)
        with lock:
            buf.extend(res)
            n_done += len(res)
            flush()
            if n_done % (batch * workers * 5) < batch:
                rate = n_done / max(time.time() - t0, 1)
                print(f"[collect] {n_done:,}/{len(todo):,} ({rate:.1f}/s, cost {client.cost})",
                      flush=True)

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, chunks))
    with lock:
        flush(force=True)
    print(f"[collect] done: {n_done:,} in {time.time() - t0:.0f}s, cost {client.cost}", flush=True)


def merge_dates() -> pd.DataFrame:
    """Join the parts into ``dates.parquet`` and compute the arrival of each copy.

    ``arrival`` is the later of two dates: the first commit with the hash of the skill and
    the creation of the repository. The second corrects repositories created by cloning
    another one, which inherit its old history. Only dated copies keep an arrival.
    """
    dates = pd.concat([pd.read_parquet(p) for p in sorted(DATES_DIR.glob("*.parquet"))],
                      ignore_index=True).drop_duplicates(["repo_full_name", "path",
                                                          "content_hash"], keep="last")
    for c in ("repo_created_at", "first_hash_commit_at"):
        dates[c] = pd.to_datetime(dates[c], utc=True)
    dates["arrival"] = dates[["first_hash_commit_at", "repo_created_at"]].max(axis=1)
    # A copy seen in the snapshot cannot have arrived after the end of the GitSkills crawl.
    # Such dates come from recreated repositories or rewritten histories, so the copy is
    # treated as not dated (protocol, section 13, deviation 1).
    late = (dates["status"] == "datada") & (dates["arrival"] > CRAWL_END)
    dates.loc[late, "status"] = "data_posterior_ao_retrato"
    dates.loc[dates["status"] != "datada", "arrival"] = pd.NaT
    dates.to_parquet(DATES, index=False)
    return dates


# ---------------------------------------------------------------------------
# Step 3: stars of the origin repository at T0
# ---------------------------------------------------------------------------

# Reads the stars of a repository in chronological order, with the date of each one.
STARGAZERS = """f{i}: repository(owner: $o{i}, name: $n{i}) {{
  stargazers(first: 100, after: $c{i}, orderBy: {{field: STARRED_AT, direction: ASC}}) {{
    totalCount pageInfo {{ hasNextPage endCursor }} edges {{ starredAt }} }} }}"""


def origins(dates: pd.DataFrame, occ: pd.DataFrame) -> pd.DataFrame:
    """T0 and origin repository of each skill, from the dated copies outside aggregators.

    The origin is the copy that arrived first (ties broken by repository name). The same
    rule is applied again with the table of the study in ``cohort.timing``.
    """
    d = dates.merge(occ[["content_hash", "repo_full_name", "path", "aggregator"]],
                    on=["content_hash", "repo_full_name", "path"])
    d = d[~d["aggregator"] & d["arrival"].notna()].sort_values(["content_hash", "arrival",
                                                                 "repo_full_name"])
    return d.drop_duplicates("content_hash", keep="first")[["content_hash", "repo_full_name",
                                                            "arrival"]]


def collect_stars(batch: int, workers: int) -> None:
    """Read the dated stars of every origin repository, until T0 or ``STAR_CAP`` stars.

    In practice the GitHub API returned no dates for the stars, so the stars at T0 could
    not be rebuilt (protocol, section 13, deviation 3).
    """
    occ = pd.read_parquet(OCCURRENCES)
    org = origins(merge_dates(), occ)
    # For each repository, the latest T0 among its skills tells how far to read its stars.
    need = org.groupby("repo_full_name")["arrival"].max().reset_index()
    need = need[need["arrival"] <= SNAPSHOT]
    client, rows, lock, t0 = Client(), [], threading.Lock(), time.time()

    def work(chunk: list[dict]) -> None:
        """Read the stars of one batch of repositories, page by page."""
        state = {i: {"times": [], "cursor": None, "status": "ok"} for i in range(len(chunk))}
        pending = list(state)
        while pending:
            decl, parts, var = [], [], {}
            for i in pending:
                owner, name = chunk[i]["repo_full_name"].split("/", 1)
                var.update({f"o{i}": owner, f"n{i}": name, f"c{i}": state[i]["cursor"]})
                decl += [f"$o{i}: String!", f"$n{i}: String!", f"$c{i}: String"]
                parts.append(STARGAZERS.format(i=i))
            q = "query(" + ", ".join(decl) + ") { rateLimit { cost remaining resetAt } " \
                + " ".join(parts) + " }"
            data = client.query(q, var).get("data")
            nxt = []
            for i in pending:
                repo = (data or {}).get(f"f{i}")
                if repo is None:
                    state[i]["status"] = "falha_de_coleta" if data is None else \
                        "repositorio_inacessivel"
                    continue
                sg = repo["stargazers"]
                state[i]["times"] += [e["starredAt"] for e in sg["edges"]]
                last = pd.Timestamp(state[i]["times"][-1]) if state[i]["times"] else None
                # Keep reading while there are more pages, the last star read is not past T0
                # and the cap has not been reached.
                if (sg["pageInfo"]["hasNextPage"] and last is not None
                        and last <= chunk[i]["arrival"] and len(state[i]["times"]) < STAR_CAP):
                    state[i]["cursor"] = sg["pageInfo"]["endCursor"]
                    nxt.append(i)
            pending = nxt
        with lock:
            for i, r in enumerate(chunk):
                rows.append({"repo_full_name": r["repo_full_name"], "status": state[i]["status"],
                             "starred_at": state[i]["times"][:STAR_CAP]})
            if len(rows) % (batch * workers * 5) < batch:
                print(f"[stars] {len(rows):,}/{len(need):,} ({time.time() - t0:.0f}s, "
                      f"cost {client.cost})", flush=True)

    recs = need.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, [recs[i:i + batch] for i in range(0, len(recs), batch)]))
    pd.DataFrame(rows).to_parquet(STARS, index=False)
    print(f"[stars] done: {len(rows):,} repositories, cost {client.cost}", flush=True)


# ---------------------------------------------------------------------------
# Command line: ``skillreuse collect <step>``
# ---------------------------------------------------------------------------

def collect_main() -> None:
    """Command ``skillreuse collect {sample,dates,stars,merge}``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["sample", "dates", "stars", "merge"])
    parser.add_argument("--scope", choices=["main", "aggregators"], default="main")
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    a = parser.parse_args()
    if a.step == "sample":
        occ, counts = build_sample()
        COLLECTION.mkdir(parents=True, exist_ok=True)
        occ.to_parquet(OCCURRENCES, index=False)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({"phase": 1, "protocol": "paper/protocol.md secao 2",
                                        "created_utc": datetime.now(UTC).isoformat(),
                                        "seed": SAMPLE_SEED, "aggregator_min_skills":
                                        AGGREGATOR_MIN_SKILLS, "counts": counts}, indent=2)
                            + "\n")
        print(json.dumps(counts, indent=2))
    elif a.step == "dates":
        collect_dates(a.scope, a.batch, a.workers, a.limit)
    elif a.step == "stars":
        collect_stars(max(a.batch // 2, 10), a.workers)
    else:
        print(merge_dates()["status"].value_counts())
