import numpy as np
import pytest

from m2vtrain.targets import TeacherOptions, encode_with_teacher, load_teacher, load_transformer, reduce_targets
from m2vtrain.train import TrainOptions, align_to_targets, distill_model, finalize, pair_finetune, tokenlearn

TEXTS = ["Bruch und Licht", "die Pflanze", "Satz des Pythagoras", "Brüche der Pflanze"] * 25
FAST = TrainOptions(max_epochs=30, device="cpu", early_stopping_patience=None)


def _mean_cosine(model, texts, targets) -> float:
    encoded = model.encode(texts)
    encoded /= np.linalg.norm(encoded, axis=1, keepdims=True)
    targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    return float(np.mean(np.sum(encoded * targets, axis=1)))


def test_distill_model_adds_vocabulary_and_reduces_dimensions(tiny_teacher):
    model = distill_model(TeacherOptions(model_name=tiny_teacher), ["Bruchrechnung", "Satz des Pythagoras"], pca_dims=16)

    assert model.dim == 16
    assert "Bruchrechnung" in model.tokenizer.encode("die Bruchrechnung").tokens


def test_tokenlearn_moves_embeddings_towards_targets(tiny_teacher):
    model = distill_model(TeacherOptions(model_name=tiny_teacher), [], pca_dims=16)
    targets = np.random.RandomState(1).randn(len(TEXTS), 16).astype(np.float32)

    before = _mean_cosine(model, TEXTS, targets)
    trained = tokenlearn(model, TEXTS, targets, FAST)

    assert trained.dim == 16
    assert _mean_cosine(trained, TEXTS, targets) > before + 0.1


def test_tokenlearn_generalizes_to_texts_with_unseen_tokens(tiny_teacher, tmp_path):
    """Tokens absent from the training texts must stay in the same basis as the trained ones."""
    letters_seen, letters_unseen = "abcdefghijklm", "nopqrstuvwxyz"
    rng = np.random.RandomState(0)

    def words(letters, n):
        return [" ".join("".join(rng.choice(list(letters), 5)) for _ in range(4)) for _ in range(n)]

    train_texts, test_texts = words(letters_seen, 300), words(letters_unseen, 100)
    teacher = TeacherOptions(model_name=tiny_teacher, device="cpu")
    all_targets = reduce_targets(encode_with_teacher(train_texts + test_texts, teacher, cache_dir=tmp_path), 16)
    model = distill_model(teacher, [], pca_dims=16)

    train_targets, test_targets = all_targets[:300], all_targets[300:]
    unaligned = _mean_cosine(model, test_texts, test_targets)
    aligned_only = _mean_cosine(align_to_targets(model, train_texts, train_targets), test_texts, test_targets)

    trained = tokenlearn(model, train_texts, train_targets, TrainOptions(max_epochs=20, device="cpu"))
    held_out = _mean_cosine(trained, test_texts, test_targets)

    # A random tiny teacher gives little to generalize; the point is that unseen tokens share the target basis
    # (clearly better than unaligned) and training on seen tokens does not pull them apart again.
    assert held_out > unaligned + 0.05
    assert held_out >= aligned_only - 0.02


def test_tokenlearn_rejects_target_dimension_mismatch(tiny_teacher):
    model = distill_model(TeacherOptions(model_name=tiny_teacher), [], pca_dims=16)
    with pytest.raises(ValueError, match="dimensions"):
        tokenlearn(model, TEXTS, np.zeros((len(TEXTS), 8), dtype=np.float32), FAST)


def test_train_options_reject_endless_training():
    with pytest.raises(ValueError, match="early_stopping_patience"):
        TrainOptions(max_epochs=-1, early_stopping_patience=None)


def test_pair_finetune_brings_pairs_closer(tiny_teacher):
    model = distill_model(TeacherOptions(model_name=tiny_teacher), [], pca_dims=16)
    anchors = ["Bruch", "Licht", "Satz"] * 20
    positives = ["Pflanze", "Pythagoras", "die"] * 20

    def pair_similarity(m) -> float:
        a, b = m.encode(anchors[:3]), m.encode(positives[:3])
        return float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))

    before = pair_similarity(model)
    tuned = pair_finetune(model, anchors, positives, FAST)
    assert pair_similarity(tuned) > before


def test_finalize_quantizes(tiny_teacher):
    model = distill_model(TeacherOptions(model_name=tiny_teacher), [], pca_dims=16)

    assert finalize(model, quantize_to="int8").embedding.dtype == np.int8
    assert len(finalize(model, quantize_to="float16", vocabulary_quantization=20).embedding) == 20


def _computed_buffers_are_intact(model) -> bool:
    import torch

    return bool(
        torch.equal(model.position_ids.cpu(), torch.arange(model.config.max_position_embeddings))
        and torch.allclose(model.inv_freq.float().cpu()[:2], torch.tensor([1.0, 10000 ** (-2 / model.config.hidden_size)]))
    )


def test_remote_code_teacher_gets_its_computed_buffers_back(remote_teacher):
    import torch

    options = TeacherOptions(model_name=remote_teacher, trust_remote_code=True, device="cpu")
    model = load_transformer(options)
    assert _computed_buffers_are_intact(model)
    assert model.embeddings.weight.dtype == torch.float32  # float16 weights are slow on the CPU

    teacher = load_teacher(options)
    inner = next(module for module in teacher.modules() if module.__class__.__name__ == "RemoteModel")
    assert _computed_buffers_are_intact(inner)
    assert inner.embeddings.weight.dtype == torch.float32


def test_remote_code_teacher_distills_and_encodes(remote_teacher, tmp_path):
    options = TeacherOptions(model_name=remote_teacher, trust_remote_code=True, device="cpu")
    model = distill_model(options, ["Bruchrechnung"], pca_dims=8)
    assert model.dim == 8
    vectors = encode_with_teacher(TEXTS[:8], options, tmp_path)
    assert vectors.shape == (8, 32) and np.isfinite(vectors).all()
