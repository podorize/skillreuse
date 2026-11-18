
from skillreuse.features import extract_markdown_features


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
