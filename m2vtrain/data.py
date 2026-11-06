"""Loading, cleaning and splitting of WLO metadata exports."""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simplemma

from m2vtrain.keywords import clean_keywords, load_stoplist
from m2vtrain.text import strip_urls

COLUMNS = {
    "properties.sys:node-uuid": "id",
    "properties.cclom:title": "title",
    "properties.cclom:general_description": "description",
    "properties.cclom:general_keyword": "keywords_raw",
    "properties.cclom:general_language": "language",
    "properties.ccm:taxonid": "subjects",
}
REQUIRED_COLUMN = "properties.cclom:title"
PREPARED_COLUMNS = ["id", "title", "description", "keywords", "language", "subjects"]
_LANGUAGE_NAMES = {"deutsch": "de", "german": "de", "englisch": "en", "english": "en"}
_WHITESPACE = re.compile(r"\s+")
_MIN_BOILERPLATE_LINE_CHARS = 20
# simplemma scores are word-coverage ratios; below this, or without a clear winner, the language stays unknown.
_MIN_LANGUAGE_SCORE = 0.5
_MIN_LANGUAGE_MARGIN = 0.1


@dataclass(frozen=True)
class DataOptions:
    """How raw exports are filtered and cleaned."""

    languages: tuple[str, ...] = ("de",)
    include_unknown_language: bool = True
    detect_missing_language: bool = True
    max_description_chars: int = 1000
    boilerplate_min_repeats: int = 20
    stoplist_path: Path | None = None


def normalize_language(raw: str | None) -> str:
    """Map raw language values ('de_DE, en', 'Deutsch', ...) to a two-letter code of the first entry."""
    if not raw or not raw.strip():
        return ""
    first = raw.split(",")[0].strip().lower()
    return _LANGUAGE_NAMES.get(first, first[:2])


def _split_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def export_columns(path: Path) -> list[str]:
    """The column headers of an export, read without loading its rows."""
    return list(pd.read_csv(path, sep=";", nrows=0).columns)


def load_exports(paths: Sequence[Path]) -> pd.DataFrame:
    """Read one or more WLO exports (';'-separated CSV, optionally .gz) into the internal column layout.

    :raises ValueError: if no path is given or a file has no title column.
    """
    if not paths:
        raise ValueError("load_exports needs at least one export file.")
    frames = []
    for path in paths:
        header = export_columns(path)
        if REQUIRED_COLUMN not in header:
            raise ValueError(f"{path} has no '{REQUIRED_COLUMN}' column; is it a WLO export?")
        frame = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            keep_default_na=False,
            usecols=[column for column in COLUMNS if column in header],
        )
        frames.append(frame.rename(columns=COLUMNS).reindex(columns=list(COLUMNS.values()), fill_value=""))
    df = pd.concat(frames, ignore_index=True)
    df["language"] = df["language"].map(normalize_language)
    df["subjects"] = df["subjects"].map(_split_list)
    return df[["id", "title", "description", "keywords_raw", "language", "subjects"]]


def remove_boilerplate(texts: Sequence[str], min_repeats: int) -> list[str]:
    """Drop lines that occur in at least `min_repeats` different texts (channel footers, ads, ...)."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update({line.strip() for line in text.split("\n") if len(line.strip()) >= _MIN_BOILERPLATE_LINE_CHARS})
    boilerplate = {line for line, count in counts.items() if count >= min_repeats}
    return ["\n".join(line for line in text.split("\n") if line.strip() not in boilerplate).strip() for text in texts]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


def _detect_language(text: str, candidates: tuple[str, ...]) -> str:
    scores = [(lang, score) for lang, score in simplemma.langdetect(text, lang=candidates) if lang != "unk"]
    if not scores:
        return ""
    best, best_score = scores[0]
    runner_up = scores[1][1] if len(scores) > 1 else 0.0
    if best_score < _MIN_LANGUAGE_SCORE or best_score - runner_up < _MIN_LANGUAGE_MARGIN:
        return ""
    return best


def _filter_language(df: pd.DataFrame, options: DataOptions) -> pd.DataFrame:
    language = df["language"]
    if options.detect_missing_language:
        candidates = tuple(dict.fromkeys((*options.languages, "de", "en")))
        missing = language == ""
        texts = df.loc[missing, "title"] + " " + df.loc[missing, "description"]
        language = language.copy()
        language[missing] = texts.map(lambda text: _detect_language(text, candidates))
    keep = language.isin(options.languages)
    if options.include_unknown_language:
        keep |= language == ""
    return df[keep].assign(language=language[keep])


def prepare(df: pd.DataFrame, options: DataOptions) -> pd.DataFrame:
    """Clean loaded exports: drop untitled rows, filter languages, clean texts and keywords, dedupe."""
    df = df.assign(**{column: df[column].map(html.unescape) for column in ("title", "description", "keywords_raw")})
    df = df[df["title"].str.strip() != ""]
    # Exact duplicates go first, otherwise a resource exported 25 times would count as boilerplate.
    df = df[~df[["title", "description"]].duplicated()]
    df = _filter_language(df, options)
    if df.empty:
        return pd.DataFrame(columns=PREPARED_COLUMNS)

    descriptions = [strip_urls(text) for text in df["description"]]
    descriptions = remove_boilerplate(descriptions, options.boilerplate_min_repeats)
    descriptions = [_truncate(_WHITESPACE.sub(" ", text).strip(), options.max_description_chars) for text in descriptions]

    stoplist = load_stoplist(options.stoplist_path)
    df = df.assign(
        title=df["title"].map(lambda text: _WHITESPACE.sub(" ", text).strip()),
        description=descriptions,
        keywords=[clean_keywords(raw, stoplist) for raw in df["keywords_raw"]],
    )
    duplicate = df["title"].str.lower().str.cat(df["description"].str.lower(), sep="\x00").duplicated()
    return df[~duplicate][PREPARED_COLUMNS].reset_index(drop=True)
