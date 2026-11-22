"""Text attributes of a SKILL.md file (the 27 attributes of the model, protocol section 3).

Every attribute is a simple count or ratio computed with a regular expression on the raw
text, with no Markdown parser. This keeps them easy to audit: each one can be checked by
hand on any file. ``tests/test_features.py`` shows an example of every count.

All copies of a skill have exactly the same text, so these attributes describe what the
other developers saw when they decided to copy it.
"""

from __future__ import annotations

import re

import pandas as pd

from skillreuse.config import CONFIG

# The regular expressions of the attributes; config/config.yaml ("text_patterns") explains
# what each one matches. MULTILINE makes ^ and $ match at the start and end of every line.
PATTERNS = CONFIG["text_patterns"]
WORD_PATTERN = re.compile(PATTERNS["word"], flags=re.UNICODE)
SENTENCE_PATTERN = re.compile(PATTERNS["sentence"])
HEADING_PATTERN = re.compile(PATTERNS["heading"], flags=re.MULTILINE)
UNORDERED_LIST_PATTERN = re.compile(PATTERNS["unordered_list"], flags=re.MULTILINE)
ORDERED_LIST_PATTERN = re.compile(PATTERNS["ordered_list"], flags=re.MULTILINE)
FENCE_PATTERN = re.compile(PATTERNS["fence"], flags=re.MULTILINE)
INLINE_CODE_PATTERN = re.compile(PATTERNS["inline_code"])
MARKDOWN_LINK_PATTERN = re.compile(PATTERNS["markdown_link"])
URL_PATTERN = re.compile(PATTERNS["url"])
BLOCKQUOTE_PATTERN = re.compile(PATTERNS["blockquote"], flags=re.MULTILINE)
CHECKBOX_PATTERN = re.compile(PATTERNS["checkbox"], flags=re.MULTILINE)
TABLE_ROW_PATTERN = re.compile(PATTERNS["table_row"], flags=re.MULTILINE)
SHELL_PROMPT_PATTERN = re.compile(PATTERNS["shell_prompt"], flags=re.MULTILINE)


def extract_markdown_features(value: object) -> dict[str, int | float]:
    """Compute the 21 text and Markdown attributes of one SKILL.md.

    A value that is not a string counts as an empty text. The attributes measure size
    (characters, lines, words, sentences), style (lexical diversity, word length, words per
    sentence), Markdown structure, code and links.
    """
    text = value if isinstance(value, str) else ""
    words = WORD_PATTERN.findall(text.lower())
    n_words = len(words)
    n_unique_words = len(set(words))
    lines = text.splitlines()
    heading_marks = HEADING_PATTERN.findall(text)  # the '#' marks of each heading
    n_sentences = len(SENTENCE_PATTERN.findall(text))
    if words and n_sentences == 0:  # a text with words but no final punctuation is one sentence
        n_sentences = 1

    def count(pattern: re.Pattern) -> int:
        return len(pattern.findall(text))

    return {
        "text_char_count": len(text),
        "text_line_count": len(lines),
        "text_nonempty_line_count": sum(bool(line.strip()) for line in lines),
        "text_word_count": n_words,
        "text_unique_word_count": n_unique_words,
        "text_lexical_diversity": n_unique_words / n_words if words else 0.0,
        "text_average_word_length": sum(len(word) for word in words) / n_words if words else 0.0,
        "text_sentence_count": n_sentences,
        "text_average_sentence_words": n_words / n_sentences if n_sentences else 0.0,
        "markdown_heading_count": len(heading_marks),
        "markdown_max_heading_depth": max((len(mark) for mark in heading_marks), default=0),
        "markdown_unordered_list_items": count(UNORDERED_LIST_PATTERN),
        "markdown_ordered_list_items": count(ORDERED_LIST_PATTERN),
        "markdown_fenced_code_blocks": count(FENCE_PATTERN) // 2,  # two fences make one block
        "markdown_inline_code_count": count(INLINE_CODE_PATTERN),
        "markdown_link_count": count(MARKDOWN_LINK_PATTERN),
        "markdown_url_count": count(URL_PATTERN),
        "markdown_blockquote_count": count(BLOCKQUOTE_PATTERN),
        "markdown_checkbox_count": count(CHECKBOX_PATTERN),
        "markdown_table_row_count": count(TABLE_ROW_PATTERN),
        "markdown_shell_prompt_count": count(SHELL_PROMPT_PATTERN),
    }


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the text attributes to a table that has the columns ``content``, ``name`` and
    ``description``, keeping all the original columns.

    The four front matter attributes added here are the length of the name and of the
    description (0 when missing) and a 0/1 flag for each one missing. The other front
    matter attributes (``frontmatter_valid``, ``body_chars``) come ready in GitSkills.
    """
    extracted = pd.DataFrame(
        [extract_markdown_features(content) for content in frame["content"]],
        index=frame.index,
    )
    result = pd.concat([frame.copy(), extracted], axis=1)
    result["frontmatter_name_char_count"] = result["name"].fillna("").str.len()
    result["frontmatter_description_char_count"] = result["description"].fillna("").str.len()
    result["frontmatter_name_missing"] = result["name"].isna().astype("int8")
    result["frontmatter_description_missing"] = result["description"].isna().astype("int8")
    return result
