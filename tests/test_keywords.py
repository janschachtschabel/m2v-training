from m2vtrain.keywords import clean_keywords, load_stoplist, split_keywords


def test_split_keywords_handles_comma_semicolon_and_plus():
    assert split_keywords("creative commons + lizenzen, Musik; Video") == [
        "creative commons",
        "lizenzen",
        "Musik",
        "Video",
    ]


def test_split_keywords_empty_and_missing():
    assert split_keywords("") == []
    assert split_keywords(None) == []


def test_clean_keywords_drops_stopwords_categories_and_numbers():
    stoplist = {"erklärvideo", "kostenlos"}
    raw = "Bruchrechnung, Erklärvideo, KOSTENLOS, Kategorie:Klexikon-Artikel, 2021, x, Brüche"
    assert clean_keywords(raw, stoplist) == ["Bruchrechnung", "Brüche"]


def test_clean_keywords_collapses_whitespace_and_dedupes_case_insensitively():
    assert clean_keywords("  Satz   des Pythagoras , satz des pythagoras", set()) == ["Satz des Pythagoras"]


def test_clean_keywords_drops_overlong_entries():
    assert clean_keywords("a" * 61 + ", Geometrie", set()) == ["Geometrie"]


def test_default_stoplist_contains_generic_platform_terms():
    stoplist = load_stoplist()
    for term in ("erklärvideo", "computer science", "kostenlos", "lernen"):
        assert term in stoplist
    assert "mathematik" not in stoplist


def test_clean_keywords_drops_urls():
    assert clean_keywords("https://ecole.science/wiki/Hauptseite, www.test.de, Optik", set()) == ["Optik"]


def test_default_stoplist_keeps_topic_words():
    stoplist = load_stoplist()
    assert "internet" not in stoplist and "film" not in stoplist
