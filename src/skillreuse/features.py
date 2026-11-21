"""Atributos de texto e estrutura Markdown do ``SKILL.md``."""

from __future__ import annotations

import re

import pandas as pd
import yaml 

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

WORD_PATTERN = re.compile(config["word_pattern"], flags=re.UNICODE)

SENTENCE_PATTERN = re.compile(config["sentence_pattern"])

HEADING_PATTERN = re.compile(config["heading_pattern"], flags=re.MULTILINE)

UNORDERED_LIST_PATTERN = re.compile(config["unordered_list_pattern"], flags=re.MULTILINE)

ORDERED_LIST_PATTERN = re.compile(config["ordered_list_pattern"], flags=re.MULTILINE)

FENCE_PATTERN = re.compile(config["fence_pattern"], flags=re.MULTILINE)

INLINE_CODE_PATTERN = re.compile(config["inline_code_pattern"])

MARKDOWN_LINK_PATTERN = re.compile(config["markdown_link_pattern"])

URL_PATTERN = re.compile(config["url_pattern"])

BLOCKQUOTE_PATTERN = re.compile(config["blockquote_pattern"], flags=re.MULTILINE)

CHECKBOX_PATTERN = re.compile(config["checkbox_pattern"], flags=re.MULTILINE)

TABLE_ROW_PATTERN = re.compile(config["table_row_pattern"], flags=re.MULTILINE)

SHELL_PROMPT_PATTERN = re.compile(config["shell_prompt_pattern"], flags=re.MULTILINE)

def extract_markdown_features(value: object) -> dict[str, int | float]:
    """Extrai medidas simples e auditaveis de texto e estrutura Markdown."""
    text = value if isinstance(value, str) else ""
    words = WORD_PATTERN.findall(text.lower())
    unique_words = set(words)
    lines = text.splitlines()
    headings = HEADING_PATTERN.findall(text)
    sentence_count = len(SENTENCE_PATTERN.findall(text))
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
        "markdown_max_heading_depth": max((len(mark) for mark in headings), default=0),
        "markdown_unordered_list_items": len(UNORDERED_LIST_PATTERN.findall(text)),
        "markdown_ordered_list_items": len(ORDERED_LIST_PATTERN.findall(text)),
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
    """Adiciona atributos derivados sem remover texto ou identificadores."""
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
