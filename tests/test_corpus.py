import pandas as pd

from m2vtrain.corpus import build_pairs, document_text, training_texts


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Brüche", "Licht", "Ohne"],
            "description": ["Brüche addieren", "Licht und Pflanze", ""],
            "keywords": [["Bruch", "Mathe"], [], ["Satz"]],
        }
    )


def test_document_text_with_and_without_keywords():
    row = _frame().iloc[0]
    assert document_text(row, with_keywords=False) == "Brüche. Brüche addieren"
    assert document_text(row, with_keywords=True) == "Brüche. Brüche addieren. Bruch, Mathe"
    assert document_text(_frame().iloc[2], with_keywords=False) == "Ohne"


def test_document_text_does_not_double_punctuation():
    row = {"title": "Was ist Licht?", "description": "Eine Erklärung", "keywords": []}
    assert document_text(row, with_keywords=False) == "Was ist Licht? Eine Erklärung"


def test_training_texts_includes_keywords_and_samples_deterministically():
    df = _frame()
    assert training_texts(df, max_texts=None, seed=0)[0] == "Brüche. Brüche addieren. Bruch, Mathe"
    sampled = training_texts(df, max_texts=2, seed=0)
    assert len(sampled) == 2
    assert sampled == training_texts(df, max_texts=2, seed=0)


def test_build_pairs_keyword_to_document_and_title_to_description():
    anchors, positives = build_pairs(_frame(), max_pairs=None, seed=0)
    pairs = set(zip(anchors, positives))

    assert ("Bruch", "Brüche. Brüche addieren") in pairs
    assert ("Mathe", "Brüche. Brüche addieren") in pairs
    assert ("Licht", "Licht und Pflanze") in pairs  # title -> description
    assert ("Satz", "Ohne") in pairs
    assert all(positive for positive in positives)


def test_build_pairs_respects_cap():
    anchors, positives = build_pairs(_frame(), max_pairs=2, seed=0)
    assert len(anchors) == len(positives) == 2


def test_tokenlearn_pool_without_extras_equals_training_texts():
    from m2vtrain.corpus import build_tokenlearn_pool

    df = _frame()
    for limit in (None, 2):
        pool = build_tokenlearn_pool(df, limit, seed=3)
        assert pool.teacher_texts == training_texts(df, limit, seed=3)  # cached teacher embeddings stay valid
        assert all(example.transform for example in pool.examples)


def test_tokenlearn_pool_adds_fulltext_questions_and_keyword_variants():
    from m2vtrain.corpus import build_tokenlearn_pool

    df = _frame().assign(
        fulltext=["Erster Absatz.\nZweiter Absatz.", "", ""],
        questions=[["Was ist ein Bruch?"], [], []],
        answers=[[], [], []],
    )
    pool = build_tokenlearn_pool(df, None, seed=0, chunk_chars=200, keyword_terms=["Brüche"])

    assert "Erster Absatz. Zweiter Absatz." in pool.teacher_texts
    assert "Was ist ein Bruch?" in pool.teacher_texts
    keyword = [example for example in pool.examples if example.document is None]
    assert {example.text for example in keyword} == {"Brüche", "brüche", "Bruch"}
    assert len({example.teacher_index for example in keyword}) == 1  # all variants learn the same target
    assert not any(example.transform for example in keyword)
    # the documents of one fold and every keyword example
    selected = pool.select({0})
    assert {example.document for example in selected} == {0, None}

    without_questions = build_tokenlearn_pool(df, None, seed=0, questions=False)
    assert "Was ist ein Bruch?" not in without_questions.teacher_texts


def test_build_pairs_adds_question_pairs():
    df = _frame().assign(questions=[["Wie addiert man Brüche?"], [], []])
    anchors, positives = build_pairs(df, max_pairs=None, seed=0)
    assert ("Wie addiert man Brüche?", "Brüche. Brüche addieren") in set(zip(anchors, positives))
    anchors, _ = build_pairs(df, max_pairs=None, seed=0, questions=False)
    assert "Wie addiert man Brüche?" not in anchors


def test_keyword_list_pairs_link_terms_to_mentioning_documents():
    from m2vtrain.corpus import keyword_list_pairs

    df = _frame().assign(fulltext=["", "Die Photosynthese braucht Licht.", ""])
    anchors, positives = keyword_list_pairs(df, ["Photosynthese", "Zelle"], max_docs_per_keyword=2, seed=0)
    assert list(zip(anchors, positives)) == [("Photosynthese", "Licht. Licht und Pflanze")]
