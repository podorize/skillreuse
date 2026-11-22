"""Settings of the study, read from ``config/config.yaml`` at the root of the repository.

Every module takes its constants (seeds, sizes, dates, paths, regular expressions and the
lists of attributes and algorithms) from ``CONFIG``, so all of them are in one file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load(path: Path = CONFIG_FILE) -> dict:
    """Read the settings file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CONFIG = load()
