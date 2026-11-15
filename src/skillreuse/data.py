"""Download, agregacao do alvo e amostra deterministica do GitSkills."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils.download import run_download
from utils.preprocess import run_pipeline

DEFAULT_OUTPUT = Path("data/raw/gitskills-full")

DEFAULT_MANIFEST = Path("results/tables/full_download_manifest.json")

DEFAULT_ARTIFACTS = Path("data/raw/gitskills-full/artifacts/*.parquet")

DEFAULT_REPOS = Path("data/raw/gitskills-full/repos/*.parquet")

DEFAULT_INDEX = Path("data/interim/full_population_index.parquet")

DEFAULT_SAMPLE = Path("data/processed/final_modeling_sample.parquet")

DEFAULT_TABLES = Path("results/tables")

DEFAULT_FIGURES = Path("results/figures")

DEFAULT_TEMP = Path("work/duckdb-tmp")

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

# ==========================================================================
# Constroi estatisticas populacionais e a amostra final de modelagem.
# ==========================================================================

def build_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--population-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--modeling-sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--reuse-threshold", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--sampling-seed", type=int, default=20261009)
    return parser.parse_args()

def build_main() -> None:
    args = build_parse_args()
    manifest = run_pipeline(
        artifacts_glob=args.artifacts,
        repos_glob=args.repos,
        population_index=args.population_index,
        modeling_sample=args.modeling_sample,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        temp_dir=args.temp_dir,
        reuse_threshold=args.reuse_threshold,
        sample_size=args.sample_size,
        sampling_seed=args.sampling_seed,
    )
    print("Populacao e amostra final geradas com sucesso.")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))