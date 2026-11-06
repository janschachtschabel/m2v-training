"""Parsing and cleaning of keyword fields (free-text export fields and list-valued dataset fields)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from m2vtrain.text import URL

_SEPARATORS = re.compile(r"\s*(?:,|;|\s\+\s)\s*")
_WHITESPACE = re.compile(r"\s+")
_CATEGORY_PREFIX = "kategorie:"
MIN_KEYWORD_CHARS = 2
MAX_KEYWORD_CHARS = 60


def split_keywords(raw: str | None) -> list[str]:
    """Split a raw keyword field on ',', ';' and ' + ' into stripped, non-empty parts."""
    if not raw:
        return []
    return [part.strip() for part in _SEPARATORS.split(raw) if part.strip()]


def load_stoplist(path: Path | None = None) -> set[str]:
    """Load a stoplist (one keyword per line, '#' comments) as a lowercase set.

    Without a path, the bundled default list of generic platform/format tags is used.
    """
    if path is None:
        text = resources.files("m2vtrain.resources").joinpath("keyword_stoplist.txt").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    lines = (line.strip().lower() for line in text.splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def _clean_one(keyword: str, stoplist: set[str]) -> str | None:
    keyword = _WHITESPACE.sub(" ", keyword).strip(" '\"")
    lowered = keyword.lower()
    if not MIN_KEYWORD_CHARS <= len(keyword) <= MAX_KEYWORD_CHARS:
        return None
    if lowered.startswith(_CATEGORY_PREFIX) or lowered in stoplist or keyword.isdigit() or URL.match(keyword):
        return None
    return keyword


def clean_keywords(raw: str | None, stoplist: set[str]) -> list[str]:
    """Split, normalize and filter a raw keyword field; keeps first spelling of case-insensitive duplicates."""
    return clean_keyword_list(split_keywords(raw), stoplist)


def clean_keyword_list(keywords: Sequence[str], stoplist: set[str]) -> list[str]:
    """Normalize and filter an already split keyword list (e.g. from a dataset with list-valued keywords)."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in keywords:
        keyword = _clean_one(str(part), stoplist)
        if keyword is None or keyword.lower() in seen:
            continue
        seen.add(keyword.lower())
        cleaned.append(keyword)
    return cleaned
