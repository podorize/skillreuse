"""Verified download of GitSkills (fixed revision) and paths of the raw data.

GitSkills is a Hugging Face dataset with two tables, ``artifacts`` (one row per SKILL.md
file found on GitHub) and ``repos`` (one row per repository). The study always uses the
same revision, and every Parquet file is checked against the SHA-256 published by
Hugging Face, so everyone reads exactly the same data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from skillreuse.config import CONFIG

# The dataset, its fixed revision and where it is saved (config/config.yaml, "dataset").
DATASET_ID = CONFIG["dataset"]["id"]
REVISION = CONFIG["dataset"]["revision"]
DEFAULT_OUTPUT = Path(CONFIG["dataset"]["folder"])
DEFAULT_MANIFEST = Path(CONFIG["dataset"]["manifest"])
TABLES = tuple(CONFIG["dataset"]["tables"])
# The download is refused if it would leave less free disk space than this.
RESERVE_BYTES = CONFIG["dataset"]["reserve_gib"] * 1024**3
# Glob patterns of the downloaded tables, read by DuckDB in collect.py and cohort.py.
DEFAULT_ARTIFACTS = Path(CONFIG["dataset"]["artifacts"])
DEFAULT_REPOS = Path(CONFIG["dataset"]["repos"])


@dataclass(frozen=True)
class RemoteFile:
    """A Parquet file of the dataset, with the size and SHA-256 published by Hugging Face."""

    path: str
    size: int
    sha256: str


def api_url(table: str) -> str:
    """URL of the Hugging Face API that lists the files of one table."""
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{encoded_revision}/data/{table}?recursive=true&expand=true"
    )


def resolve_url(remote_path: str) -> str:
    """URL that downloads one file of the fixed revision."""
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


def fetch_json(url: str) -> Any:
    """Read a JSON document from a URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "SkillReuse/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def parse_remote_files(payload: Any, table: str) -> list[RemoteFile]:
    """Turn the API answer into a sorted list of Parquet files that can be verified.

    Anything that is not a Parquet file of ``table`` is ignored. A file without size or
    SHA-256 is an error, because it could not be verified.
    """
    if not isinstance(payload, list):
        raise ValueError("Unexpected answer from the Hugging Face API")
    prefix = f"data/{table}/"
    files = []
    for entry in payload:
        path = entry.get("path", "")
        lfs = entry.get("lfs") or {}
        is_expected_file = (
            entry.get("type") == "file"
            and path.startswith(prefix)
            and path.endswith(".parquet")
        )
        if not is_expected_file:
            continue
        size = int(entry.get("size", 0))
        digest = str(lfs.get("oid", ""))
        if size <= 0 or len(digest) != 64:
            raise ValueError(f"Incomplete metadata for {path}")
        files.append(RemoteFile(path=path, size=size, sha256=digest))
    if not files:
        raise ValueError(f"No Parquet file found for table {table}")
    return sorted(files, key=lambda item: item.path)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks of 8 MiB."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing(path: Path, remote: RemoteFile) -> bool:
    """True if the file already exists and is correct; False if it is missing.

    A file that exists but has the wrong size or checksum stops the download, so a
    corrupt file is never used by mistake.
    """
    if not path.exists():
        return False
    if path.stat().st_size != remote.size:
        raise RuntimeError(
            f"Final file with the wrong size: {path}. "
            "Remove it manually before trying again."
        )
    digest = sha256_file(path)
    if digest != remote.sha256:
        raise RuntimeError(
            f"Wrong checksum in the final file: {path}. "
            "Remove it manually before trying again."
        )
    return True


def download_file(remote: RemoteFile, destination: Path) -> None:
    """Download one file to ``<name>.part`` and rename it only after it is verified.

    If a ``.part`` file exists, the download continues from where it stopped (HTTP
    Range). The final file therefore only exists when it is complete and correct.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(destination, remote):
        print(f"Already OK: {destination.name}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size > remote.size:
        raise RuntimeError(f"Partial download larger than expected: {partial}")

    headers = {"User-Agent": "SkillReuse/0.1"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(resolve_url(remote.path), headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP failure while downloading {remote.path}: {error.code}") from error

    # Status 206 (partial content) means the server accepted to resume.
    status = getattr(response, "status", response.getcode())
    if current_size and status != 206:
        response.close()
        raise RuntimeError(
            f"The server did not accept to resume {partial}. "
            "Remove the `.part` file to restart this item."
        )

    mode = "ab" if current_size else "wb"
    with response, partial.open(mode) as stream, tqdm(
        total=remote.size,
        initial=current_size,
        unit="B",
        unit_scale=True,
        desc=destination.name,
        file=sys.stdout,
    ) as progress:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            progress.update(len(chunk))

    if partial.stat().st_size != remote.size:
        raise RuntimeError(
            f"Incomplete download of {remote.path}: "
            f"{partial.stat().st_size} of {remote.size} bytes"
        )
    digest = sha256_file(partial)
    if digest != remote.sha256:
        raise RuntimeError(f"Wrong checksum after downloading {remote.path}")
    partial.replace(destination)


def collect_remote_files() -> list[RemoteFile]:
    """List the Parquet files of all tables of the dataset."""
    files: list[RemoteFile] = []
    for table in TABLES:
        files.extend(parse_remote_files(fetch_json(api_url(table)), table))
    return files


def ensure_disk_space(output: Path, files: list[RemoteFile]) -> None:
    """Stop before downloading if less than ``RESERVE_BYTES`` of disk would remain."""
    output.mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for remote in files:
        local = output / Path(remote.path).relative_to("data")
        downloaded = local.stat().st_size if local.exists() else 0
        partial = local.with_suffix(local.suffix + ".part")
        downloaded += partial.stat().st_size if partial.exists() else 0
        missing_bytes += max(0, remote.size - downloaded)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes - missing_bytes < RESERVE_BYTES:
        raise RuntimeError(
            "Not enough space: the download would leave less than 8 GiB free. "
            f"Free={free_bytes / 1024**3:.2f} GiB, "
            f"still to download={missing_bytes / 1024**3:.2f} GiB."
        )


def run_download(output: Path, manifest_path: Path) -> dict[str, Any]:
    """Download every file (skipping those already verified) and write a manifest."""
    files = collect_remote_files()
    ensure_disk_space(output, files)
    for remote in files:
        local = output / Path(remote.path).relative_to("data")
        download_file(remote, local)

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "tables": list(TABLES),
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
        "files": [asdict(item) for item in files],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def download_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def download_main() -> None:
    """Command ``skillreuse download``."""
    args = download_parse_args()
    manifest = run_download(args.output, args.manifest)
    print(
        f"Download finished: {manifest['file_count']} files, "
        f"{manifest['total_bytes'] / 1024**3:.2f} GiB."
    )



def sql_path(path: Path) -> str:
    """Escape a path to be used as a SQL string literal in DuckDB."""
    return str(path).replace("'", "''")
