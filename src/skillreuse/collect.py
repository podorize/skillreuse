"""Fase 1 do protocolo (paper/protocol.md, secoes 2.1 e 2.2): amostra e coleta das datas.

* Agregador: repositorio com 100 ou mais conteudos distintos no GitSkills.
* Populacao: conteudos com pelo menos uma ocorrencia fora de agregadores.
* Amostra: 150.000 primeiros conteudos na ordem de ``md5(content_hash || ':20261121')``.
* Coleta (API GraphQL do GitHub, somente leitura): para cada ocorrencia, o commit mais antigo
  do caminho no ramo padrao em que o arquivo tem exatamente o hash do conteudo, e a data de
  criacao do repositorio. Chegada = a mais tardia das duas.
* Estrelas do repositorio de origem em T0 (contadas ate 10.000).

Saidas em ``data/derived/collection/``; o progresso e gravado em partes e a coleta e retomavel.
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
import duckdb  # noqa: PLC0415
import yaml

import pandas as pd

from skillreuse.download import sql_path

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

AGGREGATOR_MIN_SKILLS = config["aggregator_min_skills"]
SAMPLE_SEED = config["sample_seed"]
SAMPLE_SIZE = config["sample_size"]

SNAPSHOT = pd.Timestamp(config["snapshot"], tz="UTC")
CRAWL_END = pd.Timestamp(config["crawl_end"], tz="UTC")

STAR_CAP = config["star_cap"]
MAX_COMMITS = config["max_commits"]
HISTORY_PAGE = config["history_page"]

COLLECTION = Path(config["collection"])
OCCURRENCES = Path(config["occurrences"])
DATES_DIR = Path(config["dates_dir"])
STARS_DIR = Path(config["stars_dir"])
DATES = Path(config["dates"])
STARS = Path(config["stars"])
MANIFEST = Path(config["manifest"])
DEFAULT_ARTIFACTS = Path(config["default_artifacts"])


# ---------------------------------------------------------------------------
# Amostra
# ---------------------------------------------------------------------------

def build_sample() -> tuple[pd.DataFrame, dict[str, int]]:

    con = duckdb.connect()
    con.sql("pragma threads=4; pragma memory_limit='3GB'")
    con.sql(f"""create table o as select file_sha as h, repo_full_name as r,
                split_part(repo_full_name, '/', 1) as owner, path, filename, dedup_primary
                from read_parquet('{sql_path(DEFAULT_ARTIFACTS)}')""")
    con.sql("create table rs as select r, count(distinct h) as repo_skills from o group by r")
    con.sql(f"""create table pop as select h from o join rs using (r)
                group by h having bool_or(repo_skills < {AGGREGATOR_MIN_SKILLS})""")
    con.sql(f"""create table sample as select h from pop
                order by md5(h || ':{SAMPLE_SEED}'), h limit {SAMPLE_SIZE}""")
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
# Cliente GraphQL com repeticao, espera de limite e paralelismo
# ---------------------------------------------------------------------------

class Client:
    def __init__(self) -> None:
        self.token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                                    check=True).stdout.strip()
        self.lock = threading.Lock()
        self.cost = 0
        self.pause_until = 0.0

    def query(self, q: str, variables: dict) -> dict:
        for attempt in range(4):  # 1 tentativa + 3 repeticoes
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
                if rl.get("remaining", 5000) < 300:  # espera o reinicio da cota
                    reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
                    self.pause_until = reset.timestamp() + 5
            if res.get("data") is None and res.get("errors"):
                time.sleep(5 * 2 ** attempt)
                continue
            return res
        return {"data": None, "errors": [{"type": "COLLECTION_FAILED"}]}


HISTORY = """createdAt defaultBranchRef {{ target {{ ... on Commit {{
  history(first: {page}, path: $p{i}, after: $c{i}) {{ totalCount
    pageInfo {{ hasNextPage endCursor }} nodes {{ committedDate file(path: $p{i}) {{ oid }} }} }}
}} }} }}"""


def date_batch(client: Client, rows: list[dict]) -> list[dict]:
    """Data, em cada ocorrencia, do primeiro commit com o hash do conteudo."""
    state = {i: {"nodes": [], "cursor": None, "status": None, "created": None, "commits": None}
             for i in range(len(rows))}
    pending = list(state)
    for _page in range(MAX_COMMITS // HISTORY_PAGE):
        if not pending:
            break
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
        nxt = []
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
    out = []
    for i, r in enumerate(rows):
        s = state[i]
        match = [n["committedDate"] for n in s["nodes"]
                 if (n.get("file") or {}).get("oid") == r["content_hash"]]
        status = s["status"] or ("datada" if match else
                                 "arquivo_ausente" if not s["nodes"] else "hash_nao_encontrado")
        out.append({"content_hash": r["content_hash"], "repo_full_name": r["repo_full_name"],
                    "path": r["path"], "status": status, "repo_created_at": s["created"],
                    "first_hash_commit_at": min(match) if match else None,
                    "commits_on_path": s["commits"], "commits_read": len(s["nodes"])})
    return out


def collect_dates(scope: str, batch: int, workers: int, limit: int | None = None) -> None:
    occ = pd.read_parquet(OCCURRENCES, filters=[("aggregator", "==", scope == "aggregators")])
    DATES_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    for part in DATES_DIR.glob("*.parquet"):
        d = pd.read_parquet(part, columns=["repo_full_name", "path", "content_hash"])
        done |= set(zip(d["repo_full_name"], d["path"], d["content_hash"], strict=True))
    todo = [r for r in occ.to_dict("records")
            if (r["repo_full_name"], r["path"], r["content_hash"]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[coleta] {scope}: {len(todo):,} ocorrencias a datar ({len(done):,} ja feitas)",
          flush=True)
    print("[debug] inicializando cliente GitHub...", flush=True)
    client, t0, buf, n_done = Client(), time.time(), [], 0
    print("[debug] cliente GitHub inicializado.", flush=True)
    lock = threading.Lock()
    part_id = len(list(DATES_DIR.glob("*.parquet")))

    def flush(force: bool = False) -> None:
        nonlocal buf, part_id
        if buf and (force or len(buf) >= 5000):
            pd.DataFrame(buf).to_parquet(DATES_DIR / f"{scope}_{part_id:05d}.parquet",
                                         index=False)
            part_id += 1
            buf = []

    def work(chunk: list[dict]) -> None:
        nonlocal n_done

        print(f"[debug] iniciando lote com {len(chunk)} ocorrencias",flush=True)
        res = date_batch(client, chunk)
        print(f"[debug] lote finalizado com {len(chunk)} ocorrencias",flush=True)
        with lock:
            buf.extend(res)
            n_done += len(res)
            flush()
            if n_done % (batch * workers * 5) < batch:
                rate = n_done / max(time.time() - t0, 1)
                print(f"[coleta] {n_done:,}/{len(todo):,} ({rate:.1f}/s, custo {client.cost})",
                      flush=True)

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, chunks))
    with lock:
        flush(force=True)
    print(f"[coleta] fim: {n_done:,} em {time.time() - t0:.0f}s, custo {client.cost}", flush=True)


def merge_dates() -> pd.DataFrame:
    dates = pd.concat([pd.read_parquet(p) for p in sorted(DATES_DIR.glob("*.parquet"))],
                      ignore_index=True).drop_duplicates(["repo_full_name", "path",
                                                          "content_hash"], keep="last")
    for c in ("repo_created_at", "first_hash_commit_at"):
        dates[c] = pd.to_datetime(dates[c], utc=True)
    dates["arrival"] = dates[["first_hash_commit_at", "repo_created_at"]].max(axis=1)
    # uma copia vista no retrato nao pode ter chegado depois do fim da coleta do GitSkills
    late = (dates["status"] == "datada") & (dates["arrival"] > CRAWL_END)
    dates.loc[late, "status"] = "data_posterior_ao_retrato"
    dates.loc[dates["status"] != "datada", "arrival"] = pd.NaT
    dates.to_parquet(DATES, index=False)
    return dates


# ---------------------------------------------------------------------------
# Estrelas do repositorio de origem em T0
# ---------------------------------------------------------------------------

STARGAZERS = """f{i}: repository(owner: $o{i}, name: $n{i}) {{
  stargazers(first: 100, after: $c{i}, orderBy: {{field: STARRED_AT, direction: ASC}}) {{
    totalCount pageInfo {{ hasNextPage endCursor }} edges {{ starredAt }} }} }}"""


def origins(dates: pd.DataFrame, occ: pd.DataFrame) -> pd.DataFrame:
    """T0 e repositorio de origem de cada conteudo (copias datadas fora de agregadores)."""
    d = dates.merge(occ[["content_hash", "repo_full_name", "path", "aggregator"]],
                    on=["content_hash", "repo_full_name", "path"])
    d = d[~d["aggregator"] & d["arrival"].notna()].sort_values(["content_hash", "arrival",
                                                                 "repo_full_name"])
    return d.drop_duplicates("content_hash", keep="first")[["content_hash", "repo_full_name",
                                                            "arrival"]]


def collect_stars(batch: int, workers: int) -> None:
    occ = pd.read_parquet(OCCURRENCES)
    org = origins(merge_dates(), occ)
    org = org[org["arrival"] <= SNAPSHOT]

    need = (
        org.groupby("repo_full_name")["arrival"]
        .max()
        .reset_index()
    )
    STARS_DIR.mkdir(parents=True, exist_ok=True)

    # Repositórios com coleta concluída em execuções anteriores.
    done = set()
    for part in STARS_DIR.glob("*.parquet"):
        previous = pd.read_parquet(
            part, columns=["repo_full_name", "status"]
        )
        done.update(
            previous.loc[previous["status"] == "ok", "repo_full_name"]
        )

    need = need[~need["repo_full_name"].isin(done)]

    print(f"[estrelas] {len(need):,} repositorios a coletar ({len(done):,} ja concluidos)", flush=True)

    if need.empty:
        print("[estrelas] todos os repositorios ja foram coletados", flush=True)
    else:
        client = Client()
        lock = threading.Lock()
        t0 = time.time()
        buf = []
        n_done = 0
        part_id = len(list(STARS_DIR.glob("*.parquet")))

        def flush(force: bool = False) -> None:
            nonlocal buf, part_id
            if buf and (force or len(buf) >= 100):
                pd.DataFrame(buf).to_parquet(
                    STARS_DIR / f"stars_{part_id:05d}.parquet",
                    index=False,
                )
                part_id += 1
                buf = []

        def work(chunk: list[dict]) -> None:
            nonlocal n_done

            state = {i: {"times": [], "cursor": None, "status": "ok"}
                     for i in range(len(chunk))
                     }
            pending = list(state)

            while pending:
                decl, parts, var = [], [], {}

                for i in pending:
                    owner, name = chunk[i]["repo_full_name"].split("/", 1)
                    var.update({
                        f"o{i}": owner,
                        f"n{i}": name,
                        f"c{i}": state[i]["cursor"],
                    })
                    decl += [
                        f"$o{i}: String!",
                        f"$n{i}: String!",
                        f"$c{i}: String",
                    ]
                    parts.append(STARGAZERS.format(i=i))

                q = (
                    "query(" + ", ".join(decl) + ") { "
                    "rateLimit { cost remaining resetAt } "
                    + " ".join(parts) + " }"
                )

                data = client.query(q, var).get("data")
                nxt = []

                for i in pending:
                    repo = (data or {}).get(f"f{i}")

                    if repo is None:
                        state[i]["status"] = (
                            "falha_de_coleta"
                            if data is None
                            else "repositorio_inacessivel"
                        )
                        continue

                    sg = repo["stargazers"]
                    state[i]["times"].extend(
                        e["starredAt"] for e in sg["edges"]
                    )

                    last = (
                        pd.Timestamp(state[i]["times"][-1])
                        if state[i]["times"]
                        else None
                    )

                    if (
                        sg["pageInfo"]["hasNextPage"]
                        and last is not None
                        and last <= chunk[i]["arrival"]
                        and len(state[i]["times"]) < STAR_CAP
                    ):
                        state[i]["cursor"] = sg["pageInfo"]["endCursor"]
                        nxt.append(i)

                pending = nxt

            with lock:
                for i, r in enumerate(chunk):
                    buf.append({
                        "repo_full_name": r["repo_full_name"],
                        "status": state[i]["status"],
                        "starred_at": state[i]["times"][:STAR_CAP],
                    })

                n_done += len(chunk)
                flush()

                if n_done % (batch * workers * 5) < batch:
                    rate = n_done / max(time.time() - t0, 1)
                    print(
                        f"[estrelas] {n_done:,}/{len(need):,} "
                        f"({rate:.1f}/s, custo {client.cost})",
                        flush=True,
                    )

        recs = need.to_dict("records")
        chunks = [
            recs[i:i + batch]
            for i in range(0, len(recs), batch)
        ]

        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(work, chunks))
        finally:
            with lock:
                flush(force=True)

        print(
            f"[estrelas] coleta parcial concluida: "
            f"{n_done:,} repositorios, custo {client.cost}",
            flush=True,
        )

    # Consolida todas as partes, inclusive as de execuções anteriores.
    parts = sorted(STARS_DIR.glob("*.parquet"))

    if parts:
        stars = pd.concat(
            [pd.read_parquet(part) for part in parts],
            ignore_index=True,
        )

        stars = stars.drop_duplicates(
            "repo_full_name",
            keep="last",
        )

        stars.to_parquet(STARS, index=False)

        print(
            f"[estrelas] arquivo final atualizado: "
            f"{len(stars):,} repositorios",
            flush=True,
        )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_main() -> None:
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
