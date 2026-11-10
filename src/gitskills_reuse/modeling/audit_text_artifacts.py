"""Audita artefatos lexicais nos termos usados pela explicabilidade."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(
    "reports/modeling/count_regression/explainability_hybrid/global_feature_importance.csv"
)
DEFAULT_OUTPUT = Path("reports/modeling/count_regression/explainability_hybrid")

URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
IDENTIFIER_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b", re.IGNORECASE)


def audit_terms(input_path: Path = DEFAULT_INPUT) -> tuple[pd.DataFrame, dict[str, object]]:
    table = pd.read_csv(input_path)
    text_mask = table["block"].eq("text")
    text = table.loc[text_mask, "feature"].astype(str)
    result = table.loc[text_mask, ["feature", "coefficient_log1p", "absolute_coefficient"]].copy()
    result["url_artifact"] = text.map(lambda value: bool(URL_PATTERN.search(value)))
    result["date_artifact"] = text.map(lambda value: bool(DATE_PATTERN.search(value)))
    result["numeric_artifact"] = text.map(lambda value: bool(NUMBER_PATTERN.search(value)))
    result["identifier_artifact"] = text.map(lambda value: bool(IDENTIFIER_PATTERN.search(value)))
    result["artifact_any"] = result[
        ["url_artifact", "date_artifact", "numeric_artifact", "identifier_artifact"]
    ].any(axis=1)
    counts = result[
        ["url_artifact", "date_artifact", "numeric_artifact", "identifier_artifact", "artifact_any"]
    ].sum()
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "text_terms": int(len(result)),
        "artifact_counts": {key: int(value) for key, value in counts.items()},
        "artifact_fraction": {
            key: float(value / len(result)) if len(result) else 0.0
            for key, value in counts.items()
        },
        "interpretation": "artifact flags are audit indicators, not proof of leakage",
    }
    return result, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, manifest = audit_terms(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "text_artifact_audit.csv", index=False)
    (args.output_dir / "text_artifact_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(result.loc[result["artifact_any"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
