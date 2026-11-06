"""Turn prepared documents (and keyword lists) into tokenlearn texts and contrastive pairs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from m2vtrain.lemma import to_lemma_text


_SENTENCE_END = (".", "!", "?", ":")


def document_text(row: pd.Series, with_keywords: bool) -> str:
    """'Title. Description[. kw1, kw2]' with empty parts left out and no doubled punctuation."""
    parts = [part for part in (row["title"], row["description"]) if part]
    if with_keywords and len(row["keywords"]):
        parts.append(", ".join(row["keywords"]))
    text = ""
    for part in parts:
        if text:
            text += " " if text.endswith(_SENTENCE_END) else ". "
        text += part
    return text


def _sample(items: list, limit: int | None, seed: int) -> list:
    if limit is None or limit >= len(items):
        return items
    return random.Random(seed).sample(items, limit)


def _list(row, column: str) -> list:
    value = row.get(column) if hasattr(row, "get") else None
    return list(value) if value is not None and not isinstance(value, str) else []


def document_texts(row, chunk_chars: int | None = None, questions: bool = True) -> list[str]:
    """All tokenlearn texts of one document: 'title. description. keywords' first, then full-text chunks
    (if `chunk_chars` is set), questions (if `questions`) and answers, where the document has them."""
    from m2vtrain.sources import chunk_text  # local import: sources imports this module's neighbours

    texts = [document_text(row, with_keywords=True)]
    fulltext = row.get("fulltext") if hasattr(row, "get") else ""
    if fulltext and chunk_chars:
        texts += chunk_text(fulltext, chunk_chars)
    texts += [text for text in (_list(row, "questions") if questions else []) + _list(row, "answers") if text]
    return texts


def training_texts(df: pd.DataFrame, max_texts: int | None, seed: int) -> list[str]:
    """Texts (with keywords) whose teacher embeddings become the tokenlearn targets."""
    texts = [document_text(row, with_keywords=True) for _, row in df.iterrows()]
    return _sample(texts, max_texts, seed)


@dataclass(frozen=True)
class Example:
    """One tokenlearn example: the static model's input and which teacher text gives its target.

    `document` is the row label of the source document, or None for keyword-list terms (training data of every
    fold). `transform` says whether the run's input mode (raw/lemma/mixed) still applies to `text`; explicit
    spelling variants of keyword terms are used as they are.
    """

    teacher_index: int
    text: str
    document: object = None
    transform: bool = True


@dataclass
class TokenlearnPool:
    teacher_texts: list[str] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)

    def select(self, documents: set | None) -> list[Example]:
        """Examples of the given documents (row labels) plus all keyword-list examples; None selects all."""
        if documents is None:
            return list(self.examples)
        return [example for example in self.examples if example.document is None or example.document in documents]


def keyword_variants(term: str, lowercase: bool, lemma: bool) -> list[str]:
    """Spellings of a keyword term that should map to its teacher embedding: the term, its lowercase form and
    its noun base forms (each only if different)."""
    variants = [term]
    if lowercase:
        variants.append(term.lower())
    if lemma:
        variants.append(to_lemma_text(term))
    return list(dict.fromkeys(variant for variant in variants if variant))


def build_tokenlearn_pool(
    df: pd.DataFrame,
    max_texts: int | None,
    seed: int,
    chunk_chars: int | None = None,
    questions: bool = True,
    keyword_terms: Sequence[str] = (),
    lowercase_variants: bool = True,
    lemma_variants: bool = True,
) -> TokenlearnPool:
    """Teacher texts and examples for tokenlearn.

    Document texts come first, in row order, sampled to `max_texts` (for frames without full texts, questions
    or keyword lists this is exactly `training_texts`, so cached teacher embeddings stay valid). Keyword terms
    follow, each with its spelling variants as extra examples.
    """
    units = [(label, text) for label, row in df.iterrows() for text in document_texts(row, chunk_chars, questions)]
    units = _sample(units, max_texts, seed)
    pool = TokenlearnPool()
    for label, text in units:
        pool.examples.append(Example(len(pool.teacher_texts), text, label, True))
        pool.teacher_texts.append(text)
    for term in dict.fromkeys(keyword_terms):
        index = len(pool.teacher_texts)
        pool.teacher_texts.append(term)
        pool.examples += [Example(index, variant, None, False) for variant in keyword_variants(term, lowercase_variants, lemma_variants)]
    return pool


def build_pairs(
    df: pd.DataFrame, max_pairs: int | None, seed: int, questions: bool = True
) -> tuple[list[str], list[str]]:
    """Contrastive pairs: each keyword -> its document (without keywords), title -> description and, where the
    document has them, each question -> its document."""
    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        document = document_text(row, with_keywords=False)
        pairs.extend((keyword, document) for keyword in row["keywords"])
        if row["description"]:
            pairs.append((row["title"], row["description"]))
        if questions:
            pairs.extend((question, document) for question in _list(row, "questions") if question)
    pairs = _sample(pairs, max_pairs, seed)
    return [anchor for anchor, _ in pairs], [positive for _, positive in pairs]


def keyword_list_pairs(
    df: pd.DataFrame, keyword_terms: Sequence[str], max_docs_per_keyword: int, seed: int
) -> tuple[list[str], list[str]]:
    """Pairs keyword term -> document (without keywords) for documents whose title, description or full text
    mention the term as whole words."""
    from m2vtrain.sources import keyword_mentions

    rows = list(df.iterrows())
    search_texts = [
        " ".join([row["title"], row["description"], str(row.get("fulltext") or "")]) for _, row in rows
    ]
    mentions = keyword_mentions(keyword_terms, search_texts, max_docs_per_keyword, seed)
    anchors, positives = [], []
    for term, indices in mentions.items():
        for index in indices:
            anchors.append(term)
            positives.append(document_text(rows[index][1], with_keywords=False))
    return anchors, positives
