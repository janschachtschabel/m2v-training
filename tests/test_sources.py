from collections import Counter

import pandas as pd
import pytest

from m2vtrain.sources import (
    DOCUMENT_COLUMNS,
    KeywordListOptions,
    SyntheticOptions,
    chunk_text,
    keyword_mentions,
    load_keyword_lists,
    load_synthetic,
    read_keyword_file,
    strip_markdown,
    synthetic_files,
    with_document_columns,
    write_keyword_list,
)


def test_load_synthetic_maps_the_dataset_to_documents(synthetic_dataset):
    df = load_synthetic(SyntheticOptions(enabled=True, dataset=str(synthetic_dataset)))

    assert list(df.columns) == DOCUMENT_COLUMNS
    assert len(df) == 60
    first = df.iloc[0]
    assert first["title"] == "Bruch und die Material 0"
    assert first["keywords"] == ["Bruch", "die"]  # stoplist removes "Erklärvideo"
    assert first["subjects"] == ["Mathematik"]
    assert first["group"].startswith("https://d-nb.info/gnd/bruch")
    assert "**" not in first["fulltext"] and "#" not in first["fulltext"]
    assert len(first["questions"]) == 2 and first["answers"] == []
    assert set(df["source"]) == {"synthetic"}


def test_synthetic_split_selection_and_limits(synthetic_dataset):
    files = synthetic_files(SyntheticOptions(dataset=str(synthetic_dataset), splits=("test",)))
    assert [path.name for path in files] == ["test-00000-of-00001.parquet"]

    df = load_synthetic(
        SyntheticOptions(dataset=str(synthetic_dataset), splits=("train",), max_questions_per_doc=1, use_answers=True)
    )
    assert len(df) == 50
    assert all(len(questions) == 1 for questions in df["questions"])
    assert all(len(answers) == 1 for answers in df["answers"])


def test_synthetic_files_fail_without_matching_split(synthetic_dataset):
    with pytest.raises(FileNotFoundError):
        synthetic_files(SyntheticOptions(dataset=str(synthetic_dataset / "nothing")))


def test_synthetic_options_validate():
    with pytest.raises(ValueError):
        SyntheticOptions(fulltext_chunk_chars=10)
    with pytest.raises(ValueError):
        SyntheticOptions(max_questions_per_doc=0)


def test_strip_markdown_and_chunking():
    text = strip_markdown("# Titel\n\nEin **fetter** Satz.\n- Punkt *eins*\n> Zitat")
    assert text == "Titel\nEin fetter Satz.\nPunkt eins\nZitat"
    chunks = chunk_text("Satz eins. Satz zwei.\nAbsatz drei ist da.\n" + "Lang. " * 100, max_chars=60)
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert chunks[0].startswith("Satz eins. Satz zwei. Absatz drei ist da.")
    assert "".join(chunks).count("Lang.") == 100  # an overlong paragraph is split at sentence ends, nothing lost


def test_keyword_files_round_trip(tmp_path):
    path = tmp_path / "kw.txt"
    written = write_keyword_list(path, Counter({"Bruch": 5, "Licht": 2, "Zelle": 7}), min_docs=3)
    assert written == 2
    assert path.read_text(encoding="utf-8") == "Zelle\nBruch\n"  # terms only, most frequent first

    write_keyword_list(path, Counter({"Bruch": 5}), min_docs=1, with_counts=True)
    assert read_keyword_file(path) == [("Bruch", 5)]


def test_load_keyword_lists_cleans_dedupes_and_filters(tmp_path):
    (tmp_path / "a.txt").write_text("# Kommentar\nBruch\t4\nbruch\t2\nYouTube\nLicht\t1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Zelle\n", encoding="utf-8")
    options = KeywordListOptions(files=(str(tmp_path / "a.txt"), str(tmp_path / "b.txt")))
    assert load_keyword_lists(options) == ["Bruch", "Licht", "Zelle"]
    assert load_keyword_lists(KeywordListOptions(files=options.files, min_count=2)) == ["Bruch"]
    with pytest.raises(FileNotFoundError):
        load_keyword_lists(KeywordListOptions(files=(str(tmp_path / "missing.txt"),)))


def test_keyword_mentions_match_whole_words_case_insensitively():
    texts = ["Die Bruchrechnung", "Ein Bruch und Licht", "Satz des Pythagoras", "bruch klein"]
    found = keyword_mentions(["Bruch", "Satz des Pythagoras", "Zelle"], texts, max_docs=5)
    assert sorted(found["Bruch"]) == [1, 3]  # not "Bruchrechnung"
    assert found["Satz des Pythagoras"] == [2]
    assert "Zelle" not in found
    assert len(keyword_mentions(["Bruch"], texts, max_docs=1)["Bruch"]) == 1


def test_with_document_columns_adds_empty_extras():
    df = pd.DataFrame({"id": ["1"], "title": ["T"], "description": [""], "keywords": [[]], "language": ["de"], "subjects": [[]]})
    out = with_document_columns(df, "export")
    assert list(out.columns) == DOCUMENT_COLUMNS
    assert out.iloc[0]["questions"] == [] and out.iloc[0]["group"] == "" and out.iloc[0]["source"] == "export"
