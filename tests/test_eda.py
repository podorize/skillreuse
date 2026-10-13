import pandas as pd

from gitskills_reuse.analysis.eda import (
    attribute_catalog,
    constant_feature_summary,
    normalize_empty_strings,
    numeric_summary,
)


def test_numeric_summary_detects_iqr_outlier() -> None:
    frame = pd.DataFrame({"value": [1, 1, 2, 2, 100]})

    summary = numeric_summary(frame, ["value"]).iloc[0]

    assert summary["median"] == 2
    assert summary["outlier_count_1_5_iqr"] == 1
    assert summary["outlier_percentage_1_5_iqr"] == 20


def test_attribute_catalog_distinguishes_identifiers_and_target() -> None:
    catalog = attribute_catalog().set_index("attribute")

    assert catalog.loc["content_hash", "scale"] == "nominal"
    assert catalog.loc["content_hash", "role"] == "identificador; nao e feature"
    assert catalog.loc["reused", "type"] == "qualitativo binario"
    assert catalog.loc["text_word_count", "scale"] == "racional"


def test_empty_strings_are_missing_and_constant_features_are_found() -> None:
    frame = pd.DataFrame(
        {
            "content_hash": ["a", "b"],
            "representative_repo_id": ["r1", "r2"],
            "reused": [0, 1],
            "split": ["train", "test"],
            "language": [" ", "Python"],
            "constant": [1, 1],
        }
    )

    normalized = normalize_empty_strings(frame)
    constants = constant_feature_summary(normalized)

    assert pd.isna(normalized.loc[0, "language"])
    assert set(constants["feature"]) == {"language", "constant"}
