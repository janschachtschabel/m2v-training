"""Domain evaluation: keyword->resource retrieval, title->description retrieval and a subject linear probe."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from m2vtrain.corpus import document_text
from m2vtrain.lemma import InputMode, transform_inputs
from m2vtrain.split import normalize_key
from m2vtrain.targets import load_sentence_transformer

Encoder = Callable[[list[str]], "np.ndarray | sparse.spmatrix"]
K = 10


def is_model2vec(name: str) -> bool:
    """Whether `name` (local folder or Hub id) is a Model2Vec model, judged by its config.json.

    Transformers configs always name their `model_type`; Model2Vec configs say "model2vec" or, from older
    versions, nothing.

    Called at the start of a run to check comparison model names, so a wrong one fails in seconds.

    :raises FileNotFoundError: if `name` looks like a local path that does not exist or holds no config.json.
    :raises OSError: if the Hub cannot be reached or has no such model.
    """
    path = Path(name).expanduser()
    if path.exists():
        config_file = path / "config.json"
        if not config_file.exists():  # both Model2Vec and transformers folders have one
            raise FileNotFoundError(f"Vergleichsmodell {name} ist kein Modellordner (config.json fehlt).")
    elif path.is_absolute() or name.startswith((".", "~")) or "\\" in name:
        raise FileNotFoundError(f"Vergleichsmodell {name} nicht gefunden.")
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        try:
            config_file = Path(hf_hub_download(name, "config.json"))
        except RemoteEntryNotFoundError:  # not LocalEntryNotFoundError: offline or network trouble must not decide
            return False
    config = json.loads(config_file.read_text(encoding="utf-8"))
    return config.get("model_type", "model2vec") == "model2vec" or "StaticModel" in (config.get("architectures") or [])


def load_baseline(name: str, device: str | None = None, batch_size: int = 64) -> Encoder:
    """Encoder for a comparison model: Model2Vec models directly, any other model with sentence-transformers.

    Transformer baselines get no query/passage prefixes (e5 models expect them, so they score slightly low).
    """
    if is_model2vec(name):
        from model2vec import StaticModel

        return StaticModel.from_pretrained(name).encode
    model = load_sentence_transformer(name, device)
    return lambda texts: model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True)


@dataclass(frozen=True)
class EvalOptions:
    max_queries: int = 1000
    min_relevant_docs: int = 2
    max_title_pairs: int = 5000
    max_probe_train: int = 20000
    min_examples_per_subject: int = 20
    # Inverse regularization of the subject probe. The default C=1 over-regularizes L2-normalized features and
    # compresses differences between runs; on the education data used so far macro-F1 saturates around C=10 (C=100 adds nothing).
    probe_c: float = 10.0
    max_question_queries: int = 2000
    seed: int = 42


def recall_at_k(ranking: Sequence[int], relevant: set[int], k: int = K) -> float:
    """Capped recall: found relevant documents / min(relevant, k), so 1.0 is reachable for frequent keywords."""
    return len(set(ranking[:k]) & relevant) / min(len(relevant), k)


def ndcg_at_k(ranking: Sequence[int], relevant: set[int], k: int = K) -> float:
    dcg = sum(1 / math.log2(rank + 2) for rank, doc in enumerate(ranking[:k]) if doc in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def mrr_at_k(ranking: Sequence[int], target: int, k: int = K) -> float:
    for rank, doc in enumerate(ranking[:k]):
        if doc == target:
            return 1 / (rank + 1)
    return 0.0


def _normalized(vectors) -> np.ndarray | sparse.spmatrix:
    """L2-normalized rows (dense or sparse), so dot products are cosine similarities for any encoder."""
    return normalize(vectors if sparse.issparse(vectors) else np.asarray(vectors, dtype=np.float32))


def _rankings(query_vectors, doc_vectors, k: int = K) -> np.ndarray:
    """Top-k document indices per query by cosine similarity; ties go to the earlier document (deterministic)."""
    scores = _normalized(query_vectors) @ _normalized(doc_vectors).T
    if sparse.issparse(scores):
        scores = scores.toarray()
    return np.argsort(-scores, axis=1, kind="stable")[:, : min(k, scores.shape[1])]


def keyword_retrieval(encode: Encoder, test_df: pd.DataFrame, options: EvalOptions) -> dict[str, float]:
    """Each keyword is a query; relevant are the test documents tagged with it. Documents are embedded without
    their keywords, so the model has to recognise the concept in title and description. Queries are also run
    lowercased, as users type them."""
    docs = [document_text(row, with_keywords=False) for _, row in test_df.iterrows()]
    relevant: dict[str, set[int]] = {}
    spelling: dict[str, str] = {}
    for index, keywords in enumerate(test_df["keywords"]):
        for keyword in keywords:
            relevant.setdefault(keyword.lower(), set()).add(index)
            spelling.setdefault(keyword.lower(), keyword)
    keys = sorted(key for key, docs_ in relevant.items() if len(docs_) >= options.min_relevant_docs)
    if len(keys) > options.max_queries:
        keys = random.Random(options.seed).sample(keys, options.max_queries)
    if not keys:
        return {"keyword_queries": 0}

    result: dict[str, float] = {"keyword_queries": len(keys)}
    doc_vectors = encode(docs)
    for prefix, queries in (("keyword", [spelling[key] for key in keys]), ("keyword_lower", keys)):
        rankings = _rankings(encode(queries), doc_vectors)
        result[f"{prefix}_ndcg@10"] = float(np.mean([ndcg_at_k(r, relevant[key]) for r, key in zip(rankings, keys)]))
        result[f"{prefix}_recall@10"] = float(np.mean([recall_at_k(r, relevant[key]) for r, key in zip(rankings, keys)]))
    return result


def title_retrieval(encode: Encoder, test_df: pd.DataFrame, options: EvalOptions) -> dict[str, float]:
    """Find each document's description from its title.

    Rows whose normalized title or description occurs more than once are left out: identical queries with
    different targets (or identical targets) cap the reachable score and let tie order decide.
    """
    rows = test_df[test_df["description"] != ""]
    titles, descriptions = rows["title"].map(normalize_key), rows["description"].map(normalize_key)
    rows = rows[~titles.duplicated(keep=False) & ~descriptions.duplicated(keep=False)]
    if len(rows) > options.max_title_pairs:
        rows = rows.sample(options.max_title_pairs, random_state=options.seed)
    if rows.empty:
        return {"title_queries": 0}
    rankings = _rankings(encode(rows["title"].tolist()), encode(rows["description"].tolist()))
    return {
        "title_queries": len(rows),
        "title_mrr@10": float(np.mean([mrr_at_k(r, i) for i, r in enumerate(rankings)])),
        "title_recall@10": float(np.mean([recall_at_k(r, {i}) for i, r in enumerate(rankings)])),
    }


def question_retrieval(encode: Encoder, test_df: pd.DataFrame, options: EvalOptions) -> dict[str, float]:
    """Find a document (title and description, no keywords) from one of its questions, as a search query.

    Only for documents that carry questions (the synthetic dataset); one seeded question per document.
    """
    if "questions" not in test_df:
        return {}
    rng = random.Random(options.seed)
    queries, targets = [], []
    for index, questions in enumerate(test_df["questions"]):
        questions = [question for question in (questions if questions is not None else []) if question]
        if questions:
            queries.append(rng.choice(questions))
            targets.append(index)
    if not queries:
        return {}
    if len(queries) > options.max_question_queries:
        chosen = sorted(rng.sample(range(len(queries)), options.max_question_queries))
        queries, targets = [queries[i] for i in chosen], [targets[i] for i in chosen]
    docs = [document_text(row, with_keywords=False) for _, row in test_df.iterrows()]
    rankings = _rankings(encode(queries), encode(docs))
    return {
        "question_queries": len(queries),
        "question_mrr@10": float(np.mean([mrr_at_k(r, t) for r, t in zip(rankings, targets)])),
        "question_recall@10": float(np.mean([recall_at_k(r, {t}) for r, t in zip(rankings, targets)])),
    }


def _single_subject(df: pd.DataFrame) -> pd.DataFrame:
    single = df[df["subjects"].map(len) == 1]
    return single.assign(subject=single["subjects"].map(lambda subjects: subjects[0]))


def subject_probe(
    encode: Encoder, train_df: pd.DataFrame, test_df: pd.DataFrame, options: EvalOptions
) -> dict[str, float]:
    """Logistic-regression probe on frozen embeddings, trained on train-split documents with exactly one subject.

    Classes need `min_examples_per_subject` documents in the whole train split; macro-F1 averages over the
    subjects present in the test split, so every run is scored on the same label set. Few documents carry a
    single hochschulfaechersystematik subject, so this mainly probes the school discipline vocabulary.
    """
    train = _single_subject(train_df)
    counts = train["subject"].value_counts()
    classes = set(counts[counts >= options.min_examples_per_subject].index)
    train = train[train["subject"].isin(classes)]
    if len(train) > options.max_probe_train:
        train = train.sample(options.max_probe_train, random_state=options.seed)
    test = _single_subject(test_df)
    test = test[test["subject"].isin(classes)]
    if len(classes) < 2 or test.empty:
        return {"subject_classes": len(classes)}

    def texts(frame: pd.DataFrame) -> list[str]:
        return [document_text(row, with_keywords=False) for _, row in frame.iterrows()]

    features = _normalized(encode(texts(train)))
    # saga is ~15x faster than lbfgs on high-dimensional sparse TF-IDF features; lbfgs suits dense embeddings.
    probe = LogisticRegression(C=options.probe_c, max_iter=2000, solver="saga" if sparse.issparse(features) else "lbfgs")
    probe.fit(features, train["subject"])
    predicted = probe.predict(_normalized(encode(texts(test))))
    return {
        "subject_classes": len(classes),
        "subject_test_docs": len(test),
        "subject_macro_f1": float(
            f1_score(test["subject"], predicted, average="macro", labels=sorted(set(test["subject"])), zero_division=0)
        ),
    }


def evaluate(
    encode: Encoder, train_df: pd.DataFrame, test_df: pd.DataFrame, options: EvalOptions
) -> dict[str, float]:
    """All domain tasks for one encoder."""
    return {
        **keyword_retrieval(encode, test_df, options),
        **title_retrieval(encode, test_df, options),
        **question_retrieval(encode, test_df, options),
        **subject_probe(encode, train_df, test_df, options),
    }


def with_input_mode(encode: Encoder, mode: InputMode) -> Encoder:
    """Wrap an encoder so queries and documents go through the same input transform (raw or noun-lemma)."""
    mode = InputMode(mode)
    if mode is InputMode.MIXED:
        raise ValueError("Evaluation uses a fixed input mode: raw or lemma.")
    return lambda texts: encode(transform_inputs(texts, mode))


def tfidf_encoder(fit_texts: Sequence[str]) -> Encoder:
    """Lexical TF-IDF baseline fitted on the training corpus."""
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    vectorizer.fit(fit_texts)
    return lambda texts: vectorizer.transform(texts)
