import pandas as pd

from gitskills_reuse.features.build import (
    extract_markdown_features,
    make_grouped_split,
    split_fingerprint,
)


def test_extract_markdown_features() -> None:
    text = """# Skill
## Steps
- first item
1. second item
- [x] checked
> note

```bash
$ uv run test
```

Use `config.yaml` and see [documentation](https://example.com/docs).
| A | B |
|---|---|
"""
    features = extract_markdown_features(text)

    assert features["markdown_heading_count"] == 2
    assert features["markdown_max_heading_depth"] == 2
    assert features["markdown_unordered_list_items"] == 2
    assert features["markdown_ordered_list_items"] == 1
    assert features["markdown_fenced_code_blocks"] == 1
    assert features["markdown_inline_code_count"] == 1
    assert features["markdown_link_count"] == 1
    assert features["markdown_url_count"] == 1
    assert features["markdown_blockquote_count"] == 1
    assert features["markdown_checkbox_count"] == 1
    assert features["markdown_table_row_count"] == 2
    assert features["markdown_shell_prompt_count"] == 1


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for group_id in range(30):
        for item in range(2):
            rows.append(
                {
                    "content_hash": f"hash-{group_id}-{item}",
                    "representative_repo_id": f"repo-{group_id}",
                    "reused": int((group_id + item) % 4 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_is_disjoint_and_deterministic() -> None:
    frame = synthetic_frame()
    first, first_manifest = make_grouped_split(frame, test_size=0.2, random_seed=42)
    second, second_manifest = make_grouped_split(frame, test_size=0.2, random_seed=42)

    assert first.equals(second)
    assert first_manifest == second_manifest

    train_groups = set(frame.loc[first == "train", "representative_repo_id"])
    test_groups = set(frame.loc[first == "test", "representative_repo_id"])
    assert train_groups.isdisjoint(test_groups)
    assert first_manifest["group_overlap"] == 0


def test_split_fingerprint_is_order_independent() -> None:
    assignments = pd.DataFrame(
        {
            "content_hash": ["b", "a"],
            "representative_repo_id": ["r2", "r1"],
            "reused": [0, 1],
            "split": ["train", "test"],
        }
    )
    shuffled = assignments.sample(frac=1, random_state=10)

    assert split_fingerprint(assignments) == split_fingerprint(shuffled)
