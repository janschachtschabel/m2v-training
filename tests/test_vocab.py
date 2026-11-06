from m2vtrain.vocab import VocabOptions, build_vocabulary

TEXTS = [
    "Die Brüche und die Bruchrechnung",
    "Mit Brüchen rechnen",
    "Brüche kürzen",
    "Photosynthese der Pflanzen",
    "Einmaligeswort 2024",
]
KEYWORDS = [["Satz des Pythagoras", "Geometrie"], ["satz des pythagoras"], ["Seltenes Keyword"]]


def test_vocabulary_contains_forms_lemmas_lowercase_and_keywords():
    vocab = build_vocabulary(TEXTS, KEYWORDS, VocabOptions(min_word_count=2, keyword_min_docs=2))

    assert "Brüche" in vocab  # attested inflected form
    assert "Bruch" in vocab  # lemma, although it never appears as such
    assert "brüche" in vocab and "bruch" in vocab  # lowercase variants
    assert "Satz des Pythagoras" in vocab
    assert "Seltenes Keyword" not in vocab  # below keyword_min_docs
    assert "Einmaligeswort" not in vocab  # below min_word_count
    assert "2024" not in vocab


def test_vocabulary_is_ordered_by_frequency_and_unique():
    vocab = build_vocabulary(TEXTS, KEYWORDS, VocabOptions(min_word_count=1, keyword_min_docs=1))

    assert len(vocab) == len(set(vocab))
    assert vocab.index("Brüche") < vocab.index("Photosynthese")


def test_vocabulary_size_cap_applies_to_corpus_entries():
    vocab = build_vocabulary(TEXTS, [], VocabOptions(min_word_count=1, max_vocab_size=3))
    assert len(vocab) == 3


def test_optional_parts_can_be_switched_off():
    options = VocabOptions(min_word_count=2, keyword_min_docs=2, add_lemmas=False, add_lowercase_variants=False)
    vocab = build_vocabulary(TEXTS, KEYWORDS, options)

    assert "Bruch" not in vocab
    assert "brüche" not in vocab
    assert "Brüche" in vocab


def test_keywords_survive_a_tight_size_cap():
    texts = ["häufig wort text"] * 10
    vocab = build_vocabulary(texts, [["Seltenes Fachwort"]] * 3, VocabOptions(min_word_count=1, keyword_min_docs=3, max_vocab_size=1))
    assert "Seltenes Fachwort" in vocab
    assert "seltenes fachwort" not in vocab  # no lowercase copies of multi-word keywords


def test_vocabulary_skips_underscore_tokens_and_punctuated_multiword_keywords():
    texts = ["Lücke ____ und x_0 im Text"] * 5
    keywords = [["Schweißer/in", "Berlin (allg.)", "Open Educational Resources", "OER-Kompetenzprofil"]] * 3
    vocab = build_vocabulary(texts, keywords, VocabOptions(min_word_count=1, keyword_min_docs=3))

    assert "____" not in vocab and "x_0" not in vocab
    assert "Schweißer/in" not in vocab and "Berlin (allg.)" not in vocab
    assert "Open Educational Resources" in vocab and "OER-Kompetenzprofil" in vocab


def test_extra_keywords_are_always_kept():
    from m2vtrain.vocab import VocabOptions, build_vocabulary

    vocabulary = build_vocabulary(
        ["ein Text"], [], VocabOptions(min_word_count=1, keyword_min_docs=5, max_vocab_size=3), extra_keywords=["Photosynthese", "a/b"]
    )
    assert vocabulary[:2] == ["Photosynthese", "photosynthese"]
    assert "a/b" not in vocabulary
