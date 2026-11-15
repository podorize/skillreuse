"""Download and validate the GitSkills Parquet mirror."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

from utils.hashing import sha256_file

DATASET_ID = "mvaccargiu/gitskills"
DATASET_REVISION = "ebab17454a7c236f8f26b183567f1a126f42e3f8"

TABLES = ("artifacts", "repos")

def api_url(table: str) -> str:
    encoded_revision = urllib.parse.quote(DATASET_REVISION, safe="")
    return (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{encoded_revision}/data/{table}?recursive=true&expand=true"
    )

def resolve_url(remote_path: str) -> str:
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    encoded_revision = urllib.parse.quote(DATASET_REVISION, safe="")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )

def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "SkillReuse/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def parse_remote_files(payload: Any, table: str) -> list[dict]:
    """Converte a resposta da API em uma lista ordenada de Parquets verificaveis."""
    if not isinstance(payload, list):
        raise ValueError("Resposta inesperada da API do Hugging Face")
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
            raise ValueError(f"Metadados incompletos para {path}")
        files.append({"path": path,
                      "size": size,
                      "sha256": digest,
                      })
    if not files:
        raise ValueError(f"Nenhum Parquet encontrado para a tabela {table}")
    return sorted(files, key=lambda item: item["path"])

def validate_existing(path: Path, remote: dict) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != remote["size"]:
        raise RuntimeError(
            f"Arquivo final com tamanho incorreto: {path}. "
            "Remova-o manualmente antes de tentar novamente."
        )
    digest = sha256_file(path)
    if digest != remote["sha256"]:
        raise RuntimeError(
            f"Checksum incorreto no arquivo final: {path}. "
            "Remova-o manualmente antes de tentar novamente."
        )
    return True

def download_file(remote: dict, destination: Path) -> None:
    """Baixa para `.part`, retomando quando o servidor aceita HTTP Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(destination, remote):
        print(f"OK existente: {destination.name}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size > remote["size"]:
        raise RuntimeError(f"Download parcial maior que o esperado: {partial}")

    headers = {"User-Agent": "SkillReuse/0.1"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(resolve_url(remote["path"]), headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Falha HTTP ao baixar {remote['path']}: {error.code}") from error

    status = getattr(response, "status", response.getcode())
    if current_size and status != 206:
        response.close()
        raise RuntimeError(
            f"Servidor nao aceitou retomada de {partial}. "
            "Remova o arquivo `.part` para reiniciar este item."
        )

    mode = "ab" if current_size else "wb"
    with response, partial.open(mode) as stream, tqdm(
        total=remote["size"],
        initial=current_size,
        unit="B",
        unit_scale=True,
        desc=destination.name,
        file=sys.stdout,
    ) as progress:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            progress.update(len(chunk))

    if partial.stat().st_size != remote["size"]:
        raise RuntimeError(
            f"Download incompleto de {remote['path']}: "
            f"{partial.stat().st_size} de {remote['size']} bytes"
        )
    digest = sha256_file(partial)
    if digest != remote["sha256"]:
        raise RuntimeError(f"Checksum incorreto apos baixar {remote['path']}")
    partial.replace(destination)

def collect_remote_files() -> list[dict]:
    files: list[dict] = []
    
    for table in TABLES:
        url = api_url(table)
        response = fetch_json(url)
        table_files = parse_remote_files(response, table)

        files.extend(table_files)
    return files

def download_files_parallel(
    files: list[dict],
    output: Path,
    max_workers: int = 4,
) -> None:
    """Baixa vários arquivos simultaneamente."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_file,
                remote,
                output / Path(remote["path"]).relative_to("data"),
            ): remote
            for remote in files
        }

        for future in as_completed(futures):
            remote = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Falha ao baixar {remote['path']}"
                ) from error

def run_download(output: Path, manifest_path: Path, max_workers: int = 4,) -> dict[str, Any]:
    files = collect_remote_files()
    # ensure_disk_space(output, files)

    download_files_parallel(
        files=files,
        output=output,
        max_workers=max_workers,
    )

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "tables": list(TABLES),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Manifesto de download salvo em {manifest_path} "
        f"({len(files)} arquivos, {manifest['total_bytes'] / 1024**3:.2f} GiB)"
    )
    return manifest