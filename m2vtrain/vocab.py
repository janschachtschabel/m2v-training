"""Domain vocabulary for distillation: corpus words, their lemmas, lowercase variants and keywords."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import re

from m2vtrain.lemma import WORD, is_noun_lemma, is_word, lemmatize

# model2vec adds entries the teacher's pre-tokenizer splits as substring-matching AddedTokens; restricting them to
# letters, digits, spaces and hyphens keeps "/", "(" or "&" fragments from breaking unrelated words.
_SAFE_KEYWORD = re.compile(r"[^\W_]+(?:[ -][^\W_]+)*")


@dataclass(frozen=True)
class VocabOptions:
    """Which entries go into the vocabulary.

    Inflected forms, lemmas and lowercase variants are all kept so the model works on raw text as well as on
    noun-lemma input.
    """

    min_word_count: int = 5
    keyword_min_docs: int = 3
    max_vocab_size: int = 200_000
    add_lemmas: bool = True
    add_lowercase_variants: bool = True
    include_keywords: bool = True


def _word_counts(texts: Iterable[str], min_count: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(word for word in WORD.findall(text) if is_word(word))
    return Counter({word: count for word, count in counts.items() if count >= min_count})


def _lemma_counts(words: Counter[str]) -> Counter[str]:
    lemmas: Counter[str] = Counter()
    for word, count in words.items():
        lemma = lemmatize(word)
        if is_noun_lemma(lemma):
            lemmas[lemma] += count
    return lemmas


def _keyword_counts(keyword_lists: Iterable[Sequence[str]], min_docs: int) -> Counter[str]:
    """Document frequency per keyword, grouped case-insensitively under the most frequent spelling."""
    docs: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    for keywords in keyword_lists:
        for keyword in {keyword.lower(): keyword for keyword in keywords if _SAFE_KEYWORD.fullmatch(keyword)}.values():
            docs[keyword.lower()] += 1
            spellings.setdefault(keyword.lower(), Counter())[keyword] += 1
    return Counter(
        {spellings[key].most_common(1)[0][0]: count for key, count in docs.items() if count >= min_docs}
    )


def _merge_max(target: Counter[str], source: Counter[str]) -> None:
    for entry, count in source.items():
        target[entry] = max(target[entry], count)


def _lowercase_variants(entries: Counter[str]) -> Counter[str]:
    """Lowercase copies of single words; multi-word entries are skipped (each would be another substring match)."""
    return Counter({entry.lower(): count for entry, count in entries.items() if entry != entry.lower() and " " not in entry})


def build_vocabulary(
    texts: Iterable[str],
    keyword_lists: Iterable[Sequence[str]],
    options: VocabOptions,
    extra_keywords: Sequence[str] = (),
) -> list[str]:
    """Build the vocabulary: keywords (and their lowercase variants) first, then corpus words and lemmas.

    Keywords are always kept; `max_vocab_size` caps the total by dropping the rarest corpus-derived entries.
    `extra_keywords` (e.g. from a keyword list) count as keywords regardless of `keyword_min_docs`; they need
    `include_keywords` too.
    Note: model2vec appends these entries after the teacher vocabulary, so their SIF weights are nearly flat;
    tokenlearn's learned token weights do the real weighting.
    """
    keywords: Counter[str] = Counter()
    if options.include_keywords:
        keywords = _keyword_counts(keyword_lists, options.keyword_min_docs)
        known = {entry.lower() for entry in keywords}
        for keyword in extra_keywords:
            if _SAFE_KEYWORD.fullmatch(keyword) and keyword.lower() not in known:
                known.add(keyword.lower())
                keywords[keyword] = 1
        if options.add_lowercase_variants:
            _merge_max(keywords, _lowercase_variants(keywords))

    words = _word_counts(texts, options.min_word_count)
    if options.add_lemmas:
        _merge_max(words, _lemma_counts(words))
    if options.add_lowercase_variants:
        _merge_max(words, _lowercase_variants(words))

    vocabulary = [entry for entry, _ in keywords.most_common()]
    seen = set(vocabulary)
    room = max(options.max_vocab_size - len(vocabulary), 0)
    vocabulary += [entry for entry, _ in words.most_common() if entry not in seen][:room]
    return vocabulary
