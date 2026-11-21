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

# A word: letters and digits, with hyphens or apostrophes inside ("don't", "built-in").
# Underscores are not part of a word.
WORD_PATTERN = re.compile(r"\b[^\W_]+(?:[-'][^\W_]+)*\b", flags=re.UNICODE)

# End of a sentence: one or more of . ! ? followed by a space or by the end of the text.
SENTENCE_PATTERN = re.compile(r"[.!?]+(?:\s|$)")

# A heading: 1 to 6 '#' at the start of a line, then a space and some text.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+\S", flags=re.MULTILINE)

# A bullet item: a line starting with -, + or *, then a space and some text.
UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-+*]\s+\S", flags=re.MULTILINE)

# A numbered item: a line starting with a number and a dot or parenthesis ("1." or "1)").
ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s+\S", flags=re.MULTILINE)

# A code fence: a line starting with ``` or ~~~. Each code block has two fences.
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)", flags=re.MULTILINE)

# Inline code: text between single backticks, on one line. Double backticks are ignored.
INLINE_CODE_PATTERN = re.compile(r"(?<!`)`[^`\n]+`(?!`)")

# A Markdown link [text](target). It also matches images, which start with '!'.
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\([^)]+\)")

# A web address starting with http:// or https://.
URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

# A quote: a line starting with '>'.
BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?", flags=re.MULTILINE)

# A task list item such as "- [ ] todo" or "- [x] done".
CHECKBOX_PATTERN = re.compile(r"^\s*[-+*]\s+\[[ xX]\]\s+", flags=re.MULTILINE)

# A table row: a line that starts and ends with '|'. The separator line "|---|---|" also
# matches, so a table with a header has one more row than it shows.
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$", flags=re.MULTILINE)

# A shell prompt: a line starting with '$ ' followed by a command.
SHELL_PROMPT_PATTERN = re.compile(r"^\s*\$\s+\S", flags=re.MULTILINE)


def extract_markdown_features(value: object) -> dict[str, int | float]:
    """Compute the text and Markdown attributes of one SKILL.md (21 attributes).

    A value that is not a string is treated as an empty text. The attributes are:

    * size: characters, lines, non-empty lines, words, unique words, sentences;
    * style: lexical diversity (unique words / words), average word length and average
      number of words per sentence;
    * Markdown structure: headings, deepest heading, bullet items, numbered items, table
      rows, quotes, checkboxes;
    * code: code blocks, inline code spans, shell prompts;
    * links: Markdown links and URLs.
    """
    text = value if isinstance(value, str) else ""
    words = WORD_PATTERN.findall(text.lower())
    unique_words = set(words)
    lines = text.splitlines()
    headings = HEADING_PATTERN.findall(text)
    sentence_count = len(SENTENCE_PATTERN.findall(text))
    # A text with words but no final punctuation still counts as one sentence.
    if words and sentence_count == 0:
        sentence_count = 1

    return {
        "text_char_count": len(text),
        "text_line_count": len(lines),
        "text_nonempty_line_count": sum(bool(line.strip()) for line in lines),
        "text_word_count": len(words),
        "text_unique_word_count": len(unique_words),
        "text_lexical_diversity": len(unique_words) / len(words) if words else 0.0,
        "text_average_word_length": (
            sum(len(word) for word in words) / len(words) if words else 0.0
        ),
        "text_sentence_count": sentence_count,
        "text_average_sentence_words": len(words) / sentence_count if sentence_count else 0.0,
        "markdown_heading_count": len(headings),
        # findall returns the '#' marks of each heading, so their length is the depth.
        "markdown_max_heading_depth": max((len(mark) for mark in headings), default=0),
        "markdown_unordered_list_items": len(UNORDERED_LIST_PATTERN.findall(text)),
        "markdown_ordered_list_items": len(ORDERED_LIST_PATTERN.findall(text)),
        # Two fences make one block; a lone final fence is not counted.
        "markdown_fenced_code_blocks": len(FENCE_PATTERN.findall(text)) // 2,
        "markdown_inline_code_count": len(INLINE_CODE_PATTERN.findall(text)),
        "markdown_link_count": len(MARKDOWN_LINK_PATTERN.findall(text)),
        "markdown_url_count": len(URL_PATTERN.findall(text)),
        "markdown_blockquote_count": len(BLOCKQUOTE_PATTERN.findall(text)),
        "markdown_checkbox_count": len(CHECKBOX_PATTERN.findall(text)),
        "markdown_table_row_count": len(TABLE_ROW_PATTERN.findall(text)),
        "markdown_shell_prompt_count": len(SHELL_PROMPT_PATTERN.findall(text)),
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
