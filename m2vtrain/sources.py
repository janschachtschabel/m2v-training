"""Openly licensed data sources: the synthetic German education dataset and plain keyword lists.

Both produce data that can be documented and published alongside a model:

- `load_synthetic` reads `JanSchachtschabel/synthetic-german-educational-materials-50k` (CC BY 4.0, LLM-written
  texts) from the Hugging Face Hub or a local copy and maps it to the internal document layout.
- `load_keyword_lists` reads plain term lists (one keyword per line). Such a list holds only terms, no
  documents; `extract_keyword_list` builds one from local metadata exports.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m2vtrain.data import PREPARED_COLUMNS
from m2vtrain.keywords import clean_keyword_list, load_stoplist

logger = logging.getLogger(__name__)

SYNTHETIC_DATASET = "JanSchachtschabel/synthetic-german-educational-materials-50k"
SYNTHETIC_LICENSE = "cc-by-4.0"
# Optional document columns besides PREPARED_COLUMNS; WLO rows get empty values.
EXTRA_COLUMNS = ("fulltext", "questions", "answers", "group", "source")
DOCUMENT_COLUMNS = [*PREPARED_COLUMNS, *EXTRA_COLUMNS]
_MARKDOWN = re.compile(r"(^|\s)[#>*_`|-]+(?=\s)|[*_`]{1,3}(?=\S)|(?<=\S)[*_`]{1,3}", re.MULTILINE)
_SPACES = re.compile(r"[ \t]+")
_HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")


@dataclass(frozen=True)
class SyntheticOptions:
    """How the synthetic dataset is read.

    `dataset` is a Hub dataset id, a local folder with parquet files or a single parquet file. All splits are
    read by default; train/test separation is done by `m2vtrain` itself (grouped by topic), so every record can
    serve for training in k-fold cross-validation.
    """

    enabled: bool = False
    dataset: str = SYNTHETIC_DATASET
    revision: str | None = None
    splits: tuple[str, ...] = ("train", "validation", "test")
    use_fulltext: bool = True
    fulltext_chunk_chars: int = 1200
    use_questions: bool = True
    max_questions_per_doc: int | None = 3
    use_answers: bool = False
    max_documents: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", tuple(self.splits))
        if self.fulltext_chunk_chars < 200:
            raise ValueError(f"synthetic.fulltext_chunk_chars must be at least 200, got {self.fulltext_chunk_chars}.")
        if self.max_questions_per_doc is not None and self.max_questions_per_doc < 1:
            raise ValueError("synthetic.max_questions_per_doc must be at least 1 (or null for all).")
        if self.max_documents is not None and self.max_documents < 1:
            raise ValueError("synthetic.max_documents must be at least 1 (or null for all).")


@dataclass(frozen=True)
class KeywordListOptions:
    """Extra keyword lists (plain text, one term per line, optional `<tab>count`).

    The terms are used without any documents: as vocabulary entries, as tokenlearn texts (the static model learns
    the teacher embedding of each term, also from its lowercase and noun-base-form spelling), and as queries for
    keyword -> document pairs with documents of the corpus that mention the term.
    """

    files: tuple[str, ...] = ()
    min_count: int = 1
    max_keywords: int | None = None
    add_to_vocabulary: bool = True
    tokenlearn: bool = True
    lowercase_inputs: bool = True
    lemma_inputs: bool = True
    pairs: bool = True
    max_docs_per_keyword: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(str(item) for item in self.files))
        if self.min_count < 1:
            raise ValueError("keyword_lists.min_count must be at least 1.")
        if self.max_keywords is not None and self.max_keywords < 1:
            raise ValueError("keyword_lists.max_keywords must be at least 1 (or null for all).")
        if self.max_docs_per_keyword < 1:
            raise ValueError("keyword_lists.max_docs_per_keyword must be at least 1.")


# --- synthetic dataset -------------------------------------------------------------------------------------------


def _split_of(path: Path) -> str:
    """'train' for 'train-00000-of-00001.parquet' or 'train.parquet'."""
    return path.name.split("-")[0].split(".")[0]


def synthetic_files(options: SyntheticOptions) -> list[Path]:
    """Parquet files of the selected splits, downloaded from the Hub when `dataset` is no local path.

    :raises FileNotFoundError: if no parquet file of the selected splits is found.
    """
    local = Path(options.dataset).expanduser()
    if local.is_file():
        files = [local]
    elif local.is_dir():
        files = sorted(local.rglob("*.parquet"))
    elif not _HUB_ID.fullmatch(options.dataset):
        raise FileNotFoundError(f"Synthetic dataset not found: {options.dataset}")
    else:
        from huggingface_hub import HfApi, hf_hub_download

        names = [
            name
            for name in HfApi().list_repo_files(options.dataset, repo_type="dataset", revision=options.revision)
            if name.endswith(".parquet")
        ]
        names = [name for name in names if _split_of(Path(name)) in options.splits]
        files = [
            Path(hf_hub_download(options.dataset, name, repo_type="dataset", revision=options.revision)) for name in sorted(names)
        ]
    if not local.is_file():  # a single file given explicitly is read whatever its name
        order = {split: index for index, split in enumerate(options.splits)}
        files = sorted((path for path in files if _split_of(path) in order), key=lambda path: (order[_split_of(path)], path.name))
    if not files:
        raise FileNotFoundError(f"No parquet files for splits {list(options.splits)} in {options.dataset}")
    return files


def _as_list(value) -> list:
    """Parquet list cells arrive as numpy arrays, lists or None."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return list(value)
    except TypeError:
        return []


def _qa_pairs(pairs, doc_id: str, limit: int | None) -> list[dict]:
    """The record's question-answer pairs, at most `limit` of them in a sample seeded by the record id (the
    dataset lists them by question type, so the first ones would all be of one type)."""
    import random

    items = [pair for pair in _as_list(pairs) if isinstance(pair, dict) and pair.get("question")]
    if limit is not None and len(items) > limit:
        chosen = sorted(random.Random(doc_id).sample(range(len(items)), limit))
        items = [items[i] for i in chosen]
    return items


def _qa_field(pairs: list[dict], key: str) -> list[str]:
    return [str(pair.get(key) or "").strip() for pair in pairs if pair.get(key)]


def strip_markdown(text: str) -> str:
    """Plain text from the dataset's Markdown full texts: no heading, list or emphasis markers."""
    lines = (_SPACES.sub(" ", _MARKDOWN.sub(r"\1", line)).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split a plain text into chunks of whole paragraphs (whole sentences for overlong paragraphs)."""
    chunks: list[str] = []
    current = ""
    pieces: list[str] = []
    for paragraph in text.split("\n"):
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces += re.split(r"(?<=[.!?])\s+", paragraph)
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}".strip() if current else piece[:max_chars]
    if current:
        chunks.append(current)
    return chunks


def load_synthetic(options: SyntheticOptions, stoplist: set[str] | None = None) -> pd.DataFrame:
    """The synthetic dataset in the internal document layout (DOCUMENT_COLUMNS).

    `group` is the record's topic: records about the same topic always land on the same side of a split.
    `subjects` holds the subject labels, `fulltext` the Markdown-free full text. `questions` are always loaded
    (the evaluation uses those of held-out records); `use_questions` decides whether training uses them.
    """
    files = synthetic_files(options)
    logger.info("Reading synthetic dataset: %s", ", ".join(path.name for path in files))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if options.max_documents is not None and len(frame) > options.max_documents:
        frame = frame.sample(options.max_documents, random_state=0).sort_index()
    stoplist = load_stoplist() if stoplist is None else stoplist

    def column(name: str, default="") -> pd.Series:
        return frame[name] if name in frame else pd.Series([default] * len(frame), index=frame.index)

    topic = column("topic").fillna("").astype(str)
    topic_uri = column("topic_uri").fillna("").astype(str)
    ids = column("id").fillna("").astype(str)
    qa = [_qa_pairs(pairs, doc_id, options.max_questions_per_doc) for pairs, doc_id in zip(column("qa_pairs", None), ids)]
    documents = pd.DataFrame(
        {
            "id": ids,
            "title": column("title").fillna("").astype(str).str.strip(),
            "description": column("description").fillna("").astype(str).str.strip(),
            "keywords": [clean_keyword_list(_as_list(value), stoplist) for value in column("keywords", None)],
            "language": column("language", "de").fillna("de").astype(str),
            "subjects": [[str(item) for item in _as_list(value)] for value in column("subject", None)],
            "fulltext": [strip_markdown(str(text or "")) if options.use_fulltext else "" for text in column("fulltext")],
            "questions": [_qa_field(pairs, "question") for pairs in qa],  # always kept for the evaluation
            "answers": [_qa_field(pairs, "answer") if options.use_answers else [] for pairs in qa],
            "group": [uri or name for uri, name in zip(topic_uri, topic)],
            "source": "synthetic",
        }
    )
    documents = documents[documents["title"] != ""]
    return documents[DOCUMENT_COLUMNS].reset_index(drop=True)


def with_document_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add the optional DOCUMENT_COLUMNS (empty) to prepared export rows."""
    df = df.copy()
    for name in ("fulltext", "group"):
        if name not in df:
            df[name] = ""
    for name in ("questions", "answers"):
        if name not in df:
            df[name] = [[] for _ in range(len(df))]
    df["source"] = source
    return df[DOCUMENT_COLUMNS]


# --- keyword lists -----------------------------------------------------------------------------------------------


def read_keyword_file(path: Path) -> list[tuple[str, int]]:
    """(keyword, count) per line of a keyword file; lines are `term` or `term<TAB>count`, '#' starts a comment."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        term, _, count = line.partition("\t")
        term = term.strip()
        if term:
            entries.append((term, int(count) if count.strip().isdigit() else 1))
    return entries


def load_keyword_lists(options: KeywordListOptions, stoplist: set[str] | None = None) -> list[str]:
    """All terms of the configured keyword files, cleaned and deduplicated case-insensitively (first spelling
    wins), most frequent first.

    :raises FileNotFoundError: if a configured file is missing.
    """
    stoplist = load_stoplist() if stoplist is None else stoplist
    counts: Counter[str] = Counter()
    spelling: dict[str, str] = {}
    for name in options.files:
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Keyword list not found: {name}")
        for term, count in read_keyword_file(path):
            cleaned = clean_keyword_list([term], stoplist)
            if not cleaned:
                continue
            key = cleaned[0].lower()
            spelling.setdefault(key, cleaned[0])
            counts[key] += count
    keys = [key for key, count in counts.most_common() if count >= options.min_count]
    if options.max_keywords is not None:
        keys = keys[: options.max_keywords]
    return [spelling[key] for key in keys]


def keyword_document_frequencies(keyword_lists: Iterable[Sequence[str]]) -> Counter[str]:
    """Number of documents per keyword, grouped case-insensitively under the most frequent spelling."""
    docs: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    for keywords in keyword_lists:
        for keyword in {keyword.lower(): keyword for keyword in keywords}.values():
            docs[keyword.lower()] += 1
            spellings.setdefault(keyword.lower(), Counter())[keyword] += 1
    return Counter({spellings[key].most_common(1)[0][0]: count for key, count in docs.items()})


def write_keyword_list(path: Path, frequencies: Counter[str], min_docs: int, with_counts: bool = False) -> int:
    """Write the keywords used by at least `min_docs` documents, most frequent first. Returns the number written.

    Without `with_counts` the file holds only the terms, which says nothing about the size of the source data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [(term, count) for term, count in frequencies.most_common() if count >= min_docs]
    kept.sort(key=lambda item: (-item[1], item[0].lower()))
    lines = [f"{term}\t{count}" if with_counts else term for term, count in kept]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(kept)


# --- keyword -> document pairs -----------------------------------------------------------------------------------

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
_MAX_KEYWORD_WORDS = 4  # longer keywords are rare and would multiply the n-grams per text


def _ngrams(words: list[str], max_n: int) -> set[str]:
    grams = set()
    for n in range(1, max_n + 1):
        grams.update(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


def keyword_mentions(keywords: Sequence[str], texts: Sequence[str], max_docs: int, seed: int = 0) -> dict[str, list[int]]:
    """For each keyword, up to `max_docs` indices of texts that mention it as whole words (case-insensitive).

    Matching runs over word n-grams of the texts against a set of keywords, so it scales to large keyword lists.
    Among more than `max_docs` mentioning texts the earliest are taken in a seeded shuffle of the text order.
    """
    import random

    wanted: dict[str, str] = {}
    for keyword in keywords:
        normalized = " ".join(_TOKEN.findall(keyword.lower()))
        if normalized:
            wanted.setdefault(normalized, keyword)
    if not wanted:
        return {}
    max_n = min(max(len(key.split(" ")) for key in wanted), _MAX_KEYWORD_WORDS)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    found: dict[str, list[int]] = {}
    for index in order:
        words = _TOKEN.findall(texts[index].lower())
        for gram in wanted.keys() & _ngrams(words, max_n):
            hits = found.setdefault(wanted[gram], [])
            if len(hits) < max_docs:
                hits.append(index)
    return found
