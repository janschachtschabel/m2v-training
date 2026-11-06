"""Noun/lemma text transform: reduce German text to the base forms of its nouns."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from enum import Enum
from functools import lru_cache

import simplemma

# Letters/digits without underscores (fill-in blanks like "____" or "x_0" are not words), hyphenated compounds kept.
WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
_LEMMA_CACHE_SIZE = 500_000


class InputMode(str, Enum):
    """How texts are fed to the static model: unchanged, as noun lemmas, or a random mix of both."""

    RAW = "raw"
    LEMMA = "lemma"
    MIXED = "mixed"


@lru_cache(maxsize=_LEMMA_CACHE_SIZE)
def lemmatize(word: str) -> str:
    """German base form of a single word (simplemma; unknown words come back unchanged)."""
    return simplemma.lemmatize(word, lang="de")


def is_word(token: str) -> bool:
    """Tokens worth counting or embedding: no pure numbers, no single characters."""
    return len(token) > 1 and not token.isdigit()


def is_noun_lemma(lemma: str) -> bool:
    # German noun lemmas are capitalized, so this works without a POS tagger.
    return lemma[:1].isupper()


def _lowercase_noun_lemma(word: str) -> str | None:
    """Noun lemma of a lowercase query word ("schule", "aufgaben"), or None if it is not a noun.

    simplemma often reads lowercase nouns as verbs (schule -> schulen), so the capitalized spelling is tried.
    It is accepted unless it is an unchanged -en form, which is almost always a nominalized infinitive
    (lernen -> Lernen). Known false positive: adjectives with a noun twin (grün -> Grün).
    """
    capitalized = word[0].upper() + word[1:]
    lemma = lemmatize(capitalized)
    if not is_noun_lemma(lemma):
        return None
    if lemma == capitalized and word.endswith("en"):
        return None
    return lemma


def noun_lemmas(text: str) -> list[str]:
    """Base forms of the nouns in `text`, in order of appearance.

    All-lowercase input (typical for search queries and keywords) gets a fallback for nouns that simplemma
    only recognizes when capitalized.
    """
    query_style = text == text.lower()
    lemmas = []
    for word in WORD.findall(text):
        if not is_word(word):
            continue
        lemma = lemmatize(word)
        if not is_noun_lemma(lemma) and query_style:
            lemma = _lowercase_noun_lemma(word) or lemma
        if is_noun_lemma(lemma) and is_word(lemma):
            lemmas.append(lemma)
    return lemmas


def to_lemma_text(text: str) -> str:
    """Noun lemmas joined by spaces; the original text if it contains no nouns (never returns an empty input)."""
    return " ".join(noun_lemmas(text)) or text


def transform_inputs(
    texts: Sequence[str], mode: InputMode, lemma_share: float = 0.5, seed: int = 0
) -> list[str]:
    """Apply the input mode to a batch of texts.

    In MIXED mode a deterministic random `lemma_share` of the texts is converted.

    :raises ValueError: if `lemma_share` is outside [0, 1].
    """
    if not 0.0 <= lemma_share <= 1.0:
        raise ValueError(f"lemma_share must be in [0, 1], got {lemma_share}.")
    mode = InputMode(mode)
    if mode is InputMode.RAW:
        return list(texts)
    if mode is InputMode.LEMMA:
        return [to_lemma_text(text) for text in texts]
    rng = random.Random(seed)
    return [to_lemma_text(text) if rng.random() < lemma_share else text for text in texts]
