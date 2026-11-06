"""Small text helpers shared by data cleaning and keyword parsing."""

from __future__ import annotations

import re

# A URL may contain balanced (...) groups (Wikipedia links); trailing sentence punctuation is not part of it.
URL = re.compile(r"(?:https?://|www\.)(?:\([^\s()]*\)|[^\s()])*(?:\([^\s()]*\)|[^\s.,;:!?()\]])")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]")
_SPACES = re.compile(r"[ \t]+")


def strip_urls(text: str) -> str:
    """Remove URLs and the empty brackets / extra spaces they leave behind (newlines are kept)."""
    without = _EMPTY_BRACKETS.sub("", URL.sub("", text))
    return "\n".join(_SPACES.sub(" ", line).strip() for line in without.split("\n")).strip()
