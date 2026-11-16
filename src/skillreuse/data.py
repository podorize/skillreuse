import argparse
from pathlib import Path

from utils.download import run_download

DEFAULT_OUTPUT = Path("data/raw/gitskills-full")

DEFAULT_MANIFEST = Path("data/raw/gitskills-full/manifest.json")

# ==========================================================================
# Baixa e valida as tabelas essenciais do espelho Parquet do GitSkills.
# ==========================================================================

def download_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()

def download_main() -> None:
    args = download_parse_args()
    manifest = run_download(args.output, args.manifest, 4)
    print(
        f"Download concluido: {manifest['file_count']} arquivos, "
        f"{manifest['total_bytes'] / 1024**3:.2f} GiB."
    )