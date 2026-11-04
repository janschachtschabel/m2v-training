import numpy as np
import pandas as pd
import pytest

from m2vtrain.evaluate import (
    EvalOptions,
    evaluate,
    keyword_retrieval,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    subject_probe,
    tfidf_encoder,
    title_retrieval,
    with_input_mode,
)
from m2vtrain.lemma import InputMode

WORDS = ["bruch", "licht", "pflanze", "satz", "zelle", "atom", "kraft", "wort", "zahl", "staat"]


def bag_of_words(texts):
    """Lexical oracle encoder: one dimension per known word (lowercased)."""
    vectors = np.array([[text.lower().count(word) for word in WORDS] for text in texts], dtype=np.float32) + 1e-6
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def random_encoder(texts):
    return np.random.RandomState(len(texts)).randn(len(texts), 16).astype(np.float32)


def _frame(n_per_word: int = 4) -> pd.DataFrame:
    rows = []
    for i, word in enumerate(WORDS):
        for j in range(n_per_word):
            rows.append(
                {
                    "title": f"{word.capitalize()} Titel {j}",
                    "description": f"Text über {word} Nummer {j}",
                    "keywords": [word.capitalize()],
                    "subjects": [f"subject-{i % 3}"],
                }
            )
    return pd.DataFrame(rows)


def test_ranking_metrics():
    ranking = [3, 1, 7]
    assert recall_at_k(ranking, {1, 9}, k=3) == 0.5
    assert ndcg_at_k([1, 3], {1}, k=2) == 1.0
    assert ndcg_at_k([3, 1], {1}, k=2) == pytest.approx(1 / np.log2(3))
    assert mrr_at_k([4, 5, 2], 2, k=3) == pytest.approx(1 / 3)
    assert mrr_at_k([4, 5, 2], 2, k=2) == 0.0


def test_keyword_retrieval_separates_good_from_random_encoder():
    df = _frame()
    options = EvalOptions(max_queries=50)

    good = keyword_retrieval(bag_of_words, df, options)
    bad = keyword_retrieval(random_encoder, df, options)

    assert good["keyword_queries"] == len(WORDS)
    assert good["keyword_ndcg@10"] > 0.9
    assert bad["keyword_ndcg@10"] < good["keyword_ndcg@10"]
    assert "keyword_lower_ndcg@10" in good


def test_keyword_retrieval_documents_exclude_keywords():
    df = _frame().assign(description="", title="ohne Bezug")  # only the keyword field mentions the word
    result = keyword_retrieval(bag_of_words, df, EvalOptions())
    assert result["keyword_ndcg@10"] < 0.9


def test_title_retrieval():
    result = title_retrieval(bag_of_words, _frame(n_per_word=1), EvalOptions())
    assert result["title_queries"] == len(WORDS)
    assert result["title_mrr@10"] == 1.0


def test_subject_probe_learns_from_train_split():
    train, test = _frame(n_per_word=8), _frame(n_per_word=2)
    result = subject_probe(bag_of_words, train, test, EvalOptions(min_examples_per_subject=2))
    assert result["subject_macro_f1"] > 0.5
    assert result["subject_classes"] == 3


def test_tfidf_encoder_is_a_lexical_baseline():
    encode = tfidf_encoder(["bruch licht", "pflanze satz"])
    vectors = encode(["bruch", "satz"])
    assert vectors.shape[0] == 2


def test_with_input_mode_lemmatizes_before_encoding():
    seen = []
    encode = with_input_mode(lambda texts: seen.extend(texts) or np.zeros((len(texts), 2)), InputMode.LEMMA)
    encode(["Die Brüche"])
    assert seen == ["Bruch"]


def test_evaluate_runs_all_tasks():
    result = evaluate(bag_of_words, _frame(n_per_word=8), _frame(), EvalOptions(min_examples_per_subject=2))
    for key in ("keyword_ndcg@10", "title_mrr@10", "subject_macro_f1"):
        assert key in result


def test_recall_is_capped_at_k_relevant_documents():
    assert recall_at_k(list(range(10)), set(range(20)), k=10) == 1.0


def test_rankings_break_ties_by_document_order():
    from m2vtrain.evaluate import _rankings

    constant = lambda texts: np.ones((len(texts), 2), dtype=np.float32)
    assert _rankings(constant(["q"]), constant(["a", "b", "c"]), k=3).tolist() == [[0, 1, 2]]


def test_sparse_encoders_are_ranked_by_cosine():
    from scipy import sparse

    from m2vtrain.evaluate import _rankings

    # Unnormalized counts: a dot product would prefer the long document 1, cosine prefers the exact match 0.
    docs = sparse.csr_matrix(np.array([[1.0, 0.0], [5.0, 5.0]]))
    assert _rankings(sparse.csr_matrix(np.array([[1.0, 0.0]])), docs, k=1).tolist() == [[0]]


def test_documents_are_encoded_once_per_keyword_task():
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return bag_of_words(texts)

    df = _frame()
    keyword_retrieval(counting, df, EvalOptions())
    assert calls.count(len(df)) == 1


def test_tasks_report_zero_queries_on_empty_input():
    df = _frame().assign(keywords=[[] for _ in range(len(_frame()))], description="")
    assert keyword_retrieval(bag_of_words, df, EvalOptions()) == {"keyword_queries": 0}
    assert title_retrieval(bag_of_words, df, EvalOptions()) == {"title_queries": 0}
    single = _frame().assign(subjects=[["subject-0"]] * len(_frame()))
    assert subject_probe(bag_of_words, single, single, EvalOptions(min_examples_per_subject=2))["subject_classes"] == 1


def test_title_retrieval_skips_duplicate_titles_and_descriptions():
    df = pd.DataFrame(
        {
            "title": ["Bruch", "bruch!", "Licht", "Pflanze", "Zelle"],
            "description": ["Text über bruch", "Anderer Text über bruch", "Text über licht", "Gleich", "Gleich"],
        }
    )
    assert title_retrieval(bag_of_words, df, EvalOptions())["title_queries"] == 1  # only "Licht" is unambiguous


def test_subject_macro_f1_averages_over_test_subjects_only():
    def by_word(texts):
        return np.array([[t.count("alpha"), t.count("beta"), t.count("gamma")] for t in texts], dtype=np.float32) + 1e-3

    def frame(spec):
        return pd.DataFrame(
            [{"title": text, "description": "", "keywords": [], "subjects": [subject]} for text, subject in spec]
        )

    train = frame([("alpha", "A")] * 5 + [("beta", "B")] * 5 + [("gamma", "C")] * 5)
    test = frame([("alpha", "A")] * 2 + [("beta", "B"), ("gamma", "B")])  # C never occurs in the test split
    result = subject_probe(by_word, train, test, EvalOptions(min_examples_per_subject=2))
    # A: F1 1.0; B: precision 1, recall 0.5 -> 0.667. The C false positive must not add a third, zero class.
    assert result["subject_macro_f1"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_subject_classes_are_counted_before_subsampling():
    train = _frame(n_per_word=8)
    result = subject_probe(bag_of_words, train, _frame(), EvalOptions(min_examples_per_subject=20, max_probe_train=30))
    assert result["subject_classes"] == 3


def test_comparison_models_can_be_model2vec_or_sentence_transformers(tiny_teacher, tmp_path):
    from model2vec import StaticModel

    from m2vtrain.evaluate import load_baseline
    from m2vtrain.targets import TeacherOptions
    from m2vtrain.train import distill_model

    static_dir = tmp_path / "m2v"
    static = distill_model(TeacherOptions(model_name=tiny_teacher), [], pca_dims=8)
    static.save_pretrained(static_dir)
    texts = ["Bruch und Licht", "die Pflanze"]

    assert np.allclose(load_baseline(str(static_dir))(texts), StaticModel.from_pretrained(static_dir).encode(texts))
    transformer = load_baseline(tiny_teacher, device="cpu")(texts)  # a plain BERT dir: loaded with sentence-transformers
    assert transformer.shape == (2, 32)
    assert np.allclose(np.linalg.norm(transformer, axis=1), 1.0, atol=1e-5)
    with pytest.raises(OSError):
        load_baseline(str(tmp_path / "gibt-es-nicht"))


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "model2vec"}, True),
        ({"apply_pca": 256, "normalize": True, "architectures": None}, True),  # older Model2Vec configs
        ({"model_type": "bert", "architectures": ["BertModel"]}, False),
    ],
)
def test_model2vec_detection_by_config(tmp_path, config, expected):
    import json

    from m2vtrain.evaluate import is_model2vec

    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert is_model2vec(str(tmp_path)) is expected


def test_model2vec_detection_does_not_guess_when_offline(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    from m2vtrain.evaluate import is_model2vec

    def offline(*args, **kwargs):
        raise LocalEntryNotFoundError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(OSError):
        is_model2vec("minishlab/potion-multilingual-128M")


def test_local_folder_without_model_config_is_rejected(tmp_path):
    from m2vtrain.evaluate import is_model2vec

    with pytest.raises(FileNotFoundError):
        is_model2vec(str(tmp_path))  # exists, but holds no model
