import pytest

from m2vtrain.lemma import InputMode, noun_lemmas, to_lemma_text, transform_inputs


def test_noun_lemmas_keeps_only_nouns_in_base_form():
    assert noun_lemmas("Die Schülerinnen rechnen mit Brüchen und Gleichungen.") == ["Schülerin", "Bruch", "Gleichung"]


def test_noun_lemmas_restores_capitalisation_of_lowercase_queries():
    assert noun_lemmas("photosynthese") == ["Photosynthese"]


def test_noun_lemmas_recognises_lowercase_query_nouns():
    assert noun_lemmas("schule klasse aufgaben pflanzen brüche") == ["Schule", "Klasse", "Aufgabe", "Pflanze", "Bruch"]


def test_noun_lemmas_keeps_verbs_out_of_lowercase_queries():
    assert noun_lemmas("lernen rechnet gehen erklärt schnell und mit") == []


def test_noun_lemmas_skips_single_letters_and_underscores():
    assert noun_lemmas("Siehe S. 12 und x_0 im Heft") == ["Heft"]


def test_to_lemma_text_falls_back_to_original_without_nouns():
    assert to_lemma_text("schnell und einfach") == "schnell und einfach"


def test_transform_inputs_raw_returns_input_unchanged():
    texts = ["Die Brüche", "Der Satz"]
    assert transform_inputs(texts, InputMode.RAW) == texts


def test_transform_inputs_lemma_converts_every_text():
    assert transform_inputs(["Die Brüche", "Mit Gleichungen"], InputMode.LEMMA) == ["Bruch", "Gleichung"]


def test_transform_inputs_mixed_converts_a_deterministic_share():
    texts = [f"Die Brüche {i}" for i in range(400)]
    first = transform_inputs(texts, InputMode.MIXED, lemma_share=0.5, seed=7)
    second = transform_inputs(texts, InputMode.MIXED, lemma_share=0.5, seed=7)

    converted = sum(text == "Bruch" for text in first)
    assert first == second
    assert 150 < converted < 250


def test_transform_inputs_mixed_share_bounds():
    texts = ["Die Brüche", "Mit Gleichungen"]
    assert transform_inputs(texts, InputMode.MIXED, lemma_share=0.0) == texts
    assert transform_inputs(texts, InputMode.MIXED, lemma_share=1.0) == ["Bruch", "Gleichung"]


def test_transform_inputs_rejects_invalid_share():
    with pytest.raises(ValueError, match="lemma_share"):
        transform_inputs(["x"], InputMode.MIXED, lemma_share=1.5)
