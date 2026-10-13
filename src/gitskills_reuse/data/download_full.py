"""Baixa e valida as tabelas essenciais do espelho Parquet do GitSkills."""

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

DATASET_ID = "mvaccargiu/gitskills"
REVISION = "ebab17454a7c236f8f26b183567f1a126f42e3f8"
DEFAULT_OUTPUT = Path("data/raw/gitskills-full")
DEFAULT_MANIFEST = Path("reports/tables/full_download_manifest.json")
TABLES = ("artifacts", "repos")
RESERVE_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str


def api_url(table: str) -> str:
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{encoded_revision}/data/{table}?recursive=true&expand=true"
    )


def resolve_url(remote_path: str) -> str:
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "SkillReuse/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def parse_remote_files(payload: Any, table: str) -> list[RemoteFile]:
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
        files.append(RemoteFile(path=path, size=size, sha256=digest))
    if not files:
        raise ValueError(f"Nenhum Parquet encontrado para a tabela {table}")
    return sorted(files, key=lambda item: item.path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing(path: Path, remote: RemoteFile) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != remote.size:
        raise RuntimeError(
            f"Arquivo final com tamanho incorreto: {path}. "
            "Remova-o manualmente antes de tentar novamente."
        )
    digest = sha256_file(path)
    if digest != remote.sha256:
        raise RuntimeError(
            f"Checksum incorreto no arquivo final: {path}. "
            "Remova-o manualmente antes de tentar novamente."
        )
    return True


def download_file(remote: RemoteFile, destination: Path) -> None:
    """Baixa para `.part`, retomando quando o servidor aceita HTTP Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(destination, remote):
        print(f"OK existente: {destination.name}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size > remote.size:
        raise RuntimeError(f"Download parcial maior que o esperado: {partial}")

    headers = {"User-Agent": "SkillReuse/0.1"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = urllib.request.Request(resolve_url(remote.path), headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Falha HTTP ao baixar {remote.path}: {error.code}") from error

    status = getattr(response, "status", response.getcode())
    if current_size and status != 206:
        response.close()
        raise RuntimeError(
            f"Servidor nao aceitou retomada de {partial}. "
            "Remova o arquivo `.part` para reiniciar este item."
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
            f"Download incompleto de {remote.path}: "
            f"{partial.stat().st_size} de {remote.size} bytes"
        )
    digest = sha256_file(partial)
    if digest != remote.sha256:
        raise RuntimeError(f"Checksum incorreto apos baixar {remote.path}")
    partial.replace(destination)


def collect_remote_files() -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for table in TABLES:
        files.extend(parse_remote_files(fetch_json(api_url(table)), table))
    return files


def ensure_disk_space(output: Path, files: list[RemoteFile]) -> None:
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
            "Espaco insuficiente: o download preservaria menos de 8 GiB livres. "
            f"Livres={free_bytes / 1024**3:.2f} GiB, "
            f"faltam baixar={missing_bytes / 1024**3:.2f} GiB."
        )


def run_download(output: Path, manifest_path: Path) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_download(args.output, args.manifest)
    print(
        f"Download concluido: {manifest['file_count']} arquivos, "
        f"{manifest['total_bytes'] / 1024**3:.2f} GiB."
    )


if __name__ == "__main__":
    main()
