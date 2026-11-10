"""Variantes deterministicas do texto para auditoria de artefatos lexicais."""

from __future__ import annotations

import re

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
IDENTIFIER_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b", re.IGNORECASE)


def normalize_text(value: str, variant: str = "raw") -> str:
    """Retorna texto bruto ou texto sem tokens potencialmente artefatuais."""
    if variant == "raw":
        return value
    if variant != "drop_artifacts":
        raise ValueError(f"Variante textual desconhecida: {variant}")
    cleaned = URL_PATTERN.sub(" ", value)
    cleaned = IDENTIFIER_PATTERN.sub(" ", cleaned)
    cleaned = DATE_PATTERN.sub(" ", cleaned)
    cleaned = NUMBER_PATTERN.sub(" ", cleaned)
    return cleaned
