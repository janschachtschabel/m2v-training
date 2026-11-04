import pandas as pd
import pytest

from m2vtrain.data import (
    DataOptions,
    load_exports,
    normalize_language,
    prepare,
    remove_boilerplate,
    strip_urls,
)
from m2vtrain.split import split_train_test

from tests.conftest import make_row, write_wlo_export


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("de", "de"), ("de_DE", "de"), ("Deutsch", "de"), ("de_DE, en", "de"), ("en", "en"), ("", ""), (None, "")],
)
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


def test_load_exports_maps_columns(export_path):
    df = load_exports([export_path])

    assert list(df.columns) == ["id", "title", "description", "keywords_raw", "language", "subjects"]
    first = df.iloc[0]
    assert first["title"] == "Bruchrechnung einfach erklärt"
    assert first["language"] == "de"
    assert first["subjects"] == ["http://w3id.org/openeduhub/vocabs/discipline/380"]


def test_load_exports_concatenates_files(tmp_path, export_path):
    second = write_wlo_export(tmp_path / "second.csv", [make_row("Zweite Datei")], gz=False)
    assert len(load_exports([export_path, second])) == len(load_exports([export_path])) + 1


def test_load_exports_rejects_file_without_title_column(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("a;b\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cclom:title"):
        load_exports([path])


def test_strip_urls():
    assert strip_urls("Mehr unter https://example.org/x?a=1 und www.test.de/y.") == "Mehr unter und ."


def test_strip_urls_handles_parentheses_and_leaves_no_empty_brackets():
    text = "Siehe https://de.wikipedia.org/wiki/Satz_(Mathematik) und (https://example.org) hier"
    assert strip_urls(text) == "Siehe und hier"


def test_load_exports_requires_at_least_one_path():
    with pytest.raises(ValueError, match="at least one"):
        load_exports([])


def test_remove_boilerplate_drops_lines_repeated_across_documents():
    shared = "Subscribe to our channel for more great videos every week"
    texts = [f"Eigener Inhalt {i}\n{shared}" for i in range(5)] + ["Nur eigener Inhalt"]
    cleaned = remove_boilerplate(texts, min_repeats=3)
    assert cleaned[0] == "Eigener Inhalt 0"
    assert cleaned[-1] == "Nur eigener Inhalt"


def test_prepare_dedupes_cleans_and_filters(export_path):
    options = DataOptions(languages=("de",), include_unknown_language=False, detect_missing_language=True)
    df = prepare(load_exports([export_path]), options)

    titles = df["title"].tolist()
    assert titles.count("Photosynthese") == 1  # duplicate title + description removed
    assert "Normal force in an elevator" not in titles  # English filtered out
    assert df[df["title"] == "Ohne Sprache"].iloc[0]["language"] == "de"  # missing language detected
    assert "" not in titles  # rows without title dropped
    bruch = df[df["title"] == "Bruchrechnung einfach erklärt"].iloc[0]
    assert "https://" not in bruch["description"]
    assert bruch["keywords"] == ["Bruchrechnung", "Brüche"]  # Erklärvideo is on the stoplist


def test_language_detection_rejects_uncertain_guesses(tmp_path):
    rows = [make_row("Photosynthèse", "La photosynthèse est un processus bioénergétique des plantes.", language="")]
    path = write_wlo_export(tmp_path / "fr.csv.gz", rows)
    options = DataOptions(include_unknown_language=False)
    assert prepare(load_exports([path]), options).empty


def test_duplicate_rows_do_not_count_as_boilerplate(tmp_path):
    line = "Monsterkarten für Dungeons and Syntax mit vielen Aufgaben"
    path = write_wlo_export(tmp_path / "dup.csv.gz", [make_row("Monsterkarten", line)] * 25)
    df = prepare(load_exports([path]), DataOptions(boilerplate_min_repeats=20))
    assert df.iloc[0]["description"] == line


def test_prepare_decodes_html_entities(tmp_path):
    path = write_wlo_export(tmp_path / "html.csv.gz", [make_row("Lesen &amp; Schreiben", "Satz &quot;A&quot;", "Lesen &amp; Schreiben")])
    row = prepare(load_exports([path]), DataOptions()).iloc[0]
    assert (row["title"], row["description"], row["keywords"]) == ("Lesen & Schreiben", 'Satz "A"', ["Lesen & Schreiben"])


def test_prepare_keeps_english_when_requested(export_path):
    df = prepare(load_exports([export_path]), DataOptions(languages=("de", "en")))
    assert "Normal force in an elevator" in df["title"].tolist()


def test_prepare_truncates_long_descriptions(tmp_path):
    path = write_wlo_export(tmp_path / "long.csv.gz", [make_row("Lang", "Wort " * 1000)])
    df = prepare(load_exports([path]), DataOptions(max_description_chars=100))
    assert len(df.iloc[0]["description"]) <= 100


def test_split_keeps_near_duplicates_together():
    shared = "Eine ausführliche Beschreibung des Webinars zur Öffnung von Bildungsmaterialien."
    rows = []
    for i in range(200):
        rows.append({"title": f"Webinar {i}.pdf", "description": shared if i % 2 else f"Text {i}"})
        rows.append({"title": f"Webinar {i}.pptx", "description": shared if i % 2 else f"Anderer Text {i}"})
        rows.append({"title": f'Kombinationskarten "Teil {i}"', "description": ""})
        rows.append({"title": f"Kombinationskarten Teil {i}", "description": ""})
    df = pd.DataFrame(rows)
    train, test = split_train_test(df, test_fraction=0.3, seed=3)

    normalized = lambda titles: {t.replace('"', "").replace(".pdf", "").replace(".pptx", "") for t in titles}
    assert normalized(train["title"]).isdisjoint(normalized(test["title"]))
    assert shared not in set(test["description"]) or shared not in set(train["description"])


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="test_fraction"):
        split_train_test(pd.DataFrame({"title": ["a"], "description": [""]}), test_fraction=1.0, seed=0)


def test_split_is_deterministic_and_keeps_same_titles_together():
    df = pd.DataFrame({"title": [f"Titel {i % 50}" for i in range(500)], "description": [""] * 500})
    train_a, test_a = split_train_test(df, test_fraction=0.2, seed=1)
    train_b, test_b = split_train_test(df, test_fraction=0.2, seed=1)

    assert test_a.index.equals(test_b.index)
    assert set(train_a["title"]).isdisjoint(set(test_a["title"]))
    assert 0.05 < len(test_a) / len(df) < 0.4


def test_folds_keep_groups_together_and_cover_all_rows():
    from m2vtrain.split import assign_folds

    df = pd.DataFrame(
        {
            "title": [f"Titel {i}" for i in range(40)],
            "description": [""] * 40,
            "group": [f"thema-{i % 8}" for i in range(40)],
        }
    )
    folds = assign_folds(df, 4, seed=1)
    assert set(folds) <= {0, 1, 2, 3} and len(folds) == 40
    by_group: dict[str, set[int]] = {}
    for group, fold in zip(df["group"], folds):
        by_group.setdefault(group, set()).add(fold)
    assert all(len(fold_set) == 1 for fold_set in by_group.values())
    assert folds == assign_folds(df.iloc[::-1].reset_index(drop=True), 4, seed=1)[::-1]  # order-independent
    with pytest.raises(ValueError):
        assign_folds(df, 1, seed=1)


def test_group_column_also_applies_to_the_single_split():
    df = pd.DataFrame({"title": [f"T {i}" for i in range(30)], "description": [""] * 30, "group": ["a"] * 15 + ["b"] * 15})
    train, test = split_train_test(df, 0.5, seed=0)
    assert set(train["group"]).isdisjoint(set(test["group"]))
