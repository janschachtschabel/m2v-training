"""Leakage-safe train/test split: near-duplicate records always end up on the same side."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

_FILE_EXTENSION = re.compile(r"\.(?:pdf|pptx?|docx?|odt|odp|xlsx?|mp4|mp3|html?|png|jpe?g)\b", re.IGNORECASE)
_NON_WORD = re.compile(r"[\W_]+")
MIN_DESCRIPTION_KEY_CHARS = 50


def normalize_key(text: str) -> str:
    """Lowercase, drop file extensions, punctuation and quotes, collapse whitespace."""
    return _NON_WORD.sub(" ", _FILE_EXTENSION.sub(" ", text.lower())).strip()


def _find(parent: list[int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _groups(df: pd.DataFrame) -> list[int]:
    """Union-find over rows sharing a normalized title, a normalized (long enough) description or a non-empty
    `group` value (e.g. the topic of a synthetic record), if the frame has a `group` column."""
    parent = list(range(len(df)))
    first_row_for_key: dict[str, int] = {}
    groups = df["group"] if "group" in df else [""] * len(df)
    for row, (title, description, group) in enumerate(zip(df["title"], df["description"], groups)):
        keys = [f"t:{normalize_key(title)}"]
        normalized_description = normalize_key(description)
        if len(normalized_description) >= MIN_DESCRIPTION_KEY_CHARS:
            keys.append(f"d:{normalized_description}")
        if group:
            keys.append(f"g:{group}")
        for key in keys:
            other = first_row_for_key.setdefault(key, row)
            parent[_find(parent, row)] = _find(parent, other)
    return [_find(parent, row) for row in range(len(df))]


def _bucket(key: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) / 0xFFFFFFFFFFFFFFFF


def _group_keys(df: pd.DataFrame) -> list[str]:
    """Per row the key of its group: the group's smallest normalized title, independent of row order."""
    roots = _groups(df)
    group_key: dict[int, str] = {}
    for root, title in zip(roots, df["title"]):
        key = normalize_key(title)
        group_key[root] = min(group_key.get(root, key), key)
    return [group_key[root] for root in roots]


def assign_folds(df: pd.DataFrame, folds: int, seed: int) -> list[int]:
    """Deterministic fold number (0 .. folds-1) per row; near-duplicates and rows of one group share a fold.

    :raises ValueError: if `folds` is below 2.
    """
    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}.")
    return [min(int(_bucket(key, seed) * folds), folds - 1) for key in _group_keys(df)]


def split_train_test(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split deterministically by group, so near-duplicates (same normalized title or description) never straddle it.

    :raises ValueError: if `test_fraction` is not in (0, 1).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}.")
    in_test = [_bucket(key, seed) < test_fraction for key in _group_keys(df)]
    mask = pd.Series(in_test, index=df.index)
    return df[~mask], df[mask]
