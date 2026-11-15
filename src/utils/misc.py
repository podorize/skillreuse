import shutil
from pathlib import Path

RESERVE_BYTES = 8 * 1024**3  # 8 GiB


def ensure_disk_space(output: Path, files: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for remote in files:
        local = output / Path(remote["path"]).relative_to("data")
        downloaded = local.stat().st_size if local.exists() else 0
        partial = local.with_suffix(local.suffix + ".part")
        downloaded += partial.stat().st_size if partial.exists() else 0
        missing_bytes += max(0, remote["size"] - downloaded)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes - missing_bytes < RESERVE_BYTES:
        raise RuntimeError(
            "Espaco insuficiente: o download preservaria menos de 8 GiB livres. "
            f"Livres={free_bytes / 1024**3:.2f} GiB, "
            f"faltam baixar={missing_bytes / 1024**3:.2f} GiB."
        )
