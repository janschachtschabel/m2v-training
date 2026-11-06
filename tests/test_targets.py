import numpy as np
import pytest

from m2vtrain import targets
from m2vtrain.targets import TeacherOptions, encode_with_teacher, reduce_targets


def test_encode_with_teacher_returns_normalized_vectors_and_reports_progress(tiny_teacher, tmp_path):
    calls = []
    vectors = encode_with_teacher(
        ["Bruch", "Licht", "die Pflanze"],
        TeacherOptions(model_name=tiny_teacher, device="cpu", batch_size=2),
        cache_dir=tmp_path,
        progress=lambda done, total: calls.append((done, total)),
        chunk_size=2,
    )

    assert vectors.shape == (3, 32)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    assert calls[-1] == (3, 3)


def test_encode_with_teacher_uses_cache_on_second_call(tiny_teacher, tmp_path, monkeypatch):
    options = TeacherOptions(model_name=tiny_teacher, device="cpu")
    first = encode_with_teacher(["Bruch", "Licht"], options, cache_dir=tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("teacher must not be loaded when the cache is warm")

    monkeypatch.setattr(targets, "load_teacher", fail)
    second = encode_with_teacher(["Bruch", "Licht"], options, cache_dir=tmp_path)
    np.testing.assert_array_equal(first, second)


def test_document_prefix_changes_the_cache_key(tiny_teacher, tmp_path):
    encode_with_teacher(["Bruch"], TeacherOptions(model_name=tiny_teacher, device="cpu"), cache_dir=tmp_path)
    encode_with_teacher(
        ["Bruch"], TeacherOptions(model_name=tiny_teacher, device="cpu", document_prefix="passage: "), cache_dir=tmp_path
    )
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_auto_device_is_accepted(tiny_teacher, tmp_path):
    vectors = encode_with_teacher(["Bruch"], TeacherOptions(model_name=tiny_teacher, device="auto"), cache_dir=tmp_path)
    assert vectors.shape == (1, 32)


def test_empty_input_is_rejected(tiny_teacher, tmp_path):
    with pytest.raises(ValueError, match="no texts"):
        encode_with_teacher([], TeacherOptions(model_name=tiny_teacher), cache_dir=tmp_path)


def test_interrupted_featurization_resumes_from_finished_chunks(tiny_teacher, tmp_path, monkeypatch):
    options = TeacherOptions(model_name=tiny_teacher, device="cpu")
    texts = ["Bruch", "Licht", "die Pflanze", "Satz", "des"]

    def stop_after_first_chunk(done, total):
        if done < total:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        encode_with_teacher(texts, options, cache_dir=tmp_path, progress=stop_after_first_chunk, chunk_size=2)

    encoded_batches = []
    real_loader = targets.load_teacher

    def spying_loader(opts):
        teacher = real_loader(opts)
        real_encode = teacher.encode
        teacher.encode = lambda batch, **kwargs: encoded_batches.append(list(batch)) or real_encode(batch, **kwargs)
        return teacher

    monkeypatch.setattr(targets, "load_teacher", spying_loader)
    resumed = encode_with_teacher(texts, options, cache_dir=tmp_path, chunk_size=2)

    assert encoded_batches == [["die Pflanze", "Satz"], ["des"]]
    fresh = encode_with_teacher(texts, options, cache_dir=tmp_path / "fresh", chunk_size=2)
    np.testing.assert_allclose(resumed, fresh, rtol=1e-5, atol=1e-6)
    assert not list(tmp_path.glob("*.chunk*"))  # chunk files are cleaned up once the result is cached


def test_changed_local_teacher_invalidates_cache(tiny_teacher, tmp_path):
    import os
    import shutil

    teacher_copy = tmp_path / "teacher"
    shutil.copytree(tiny_teacher, teacher_copy)
    options = TeacherOptions(model_name=str(teacher_copy), device="cpu")
    encode_with_teacher(["Bruch"], options, cache_dir=tmp_path / "cache")
    weights = next(teacher_copy.glob("*.safetensors"))
    os.utime(weights, (weights.stat().st_atime, weights.stat().st_mtime + 100))
    encode_with_teacher(["Bruch"], options, cache_dir=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2


def test_reduce_targets_always_centers():
    vectors = np.random.RandomState(0).randn(50, 32).astype(np.float32) + 5.0
    for dims in (8, 32):
        np.testing.assert_allclose(reduce_targets(vectors, dims).mean(axis=0), 0.0, atol=1e-4)


def test_reduce_targets_to_fewer_dimensions():
    vectors = np.random.RandomState(0).randn(50, 32).astype(np.float32)
    assert reduce_targets(vectors, 8).shape == (50, 8)
    assert reduce_targets(vectors, 32).shape == (50, 32)


def test_reduce_targets_rejects_more_dimensions_than_teacher():
    with pytest.raises(ValueError, match="dimensions"):
        reduce_targets(np.zeros((5, 4), dtype=np.float32), 8)


def test_teacher_options_validate_pooling_dtype_and_length():
    import pytest

    from m2vtrain.targets import TeacherOptions

    for bad in ({"pooling": "cls"}, {"dtype": "int4"}, {"max_seq_length": 2}, {"batch_size": 0}):
        with pytest.raises(ValueError):
            TeacherOptions(model_name="x", **bad)


def test_cache_key_changes_only_with_non_default_length_and_dtype():
    from m2vtrain.targets import TeacherOptions, _cache_key

    base = _cache_key(["a"], TeacherOptions(model_name="x"))
    assert base == _cache_key(["a"], TeacherOptions(model_name="x", max_seq_length=None, dtype="auto"))
    assert base != _cache_key(["a"], TeacherOptions(model_name="x", max_seq_length=512))
    assert base != _cache_key(["a"], TeacherOptions(model_name="x", dtype="bfloat16"))


def test_reduce_targets_with_rows_repeats_and_fits_on_distinct_rows():
    import numpy as np

    from m2vtrain.targets import reduce_targets

    vectors = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    full = reduce_targets(vectors, 4)
    assert np.allclose(reduce_targets(vectors, 4, rows=list(range(20))), full, atol=1e-5)
    rows = [3, 3, 5, 7, 9, 11, 13]
    repeated = reduce_targets(vectors, 4, rows=rows)
    assert repeated.shape == (7, 4) and np.allclose(repeated[0], repeated[1])


def test_two_devices_give_the_same_embeddings_as_one(tiny_teacher, tmp_path, monkeypatch):
    import sys
    import types

    # Spawned workers re-import the parent's __main__; under pytest other tests may have replaced it (the
    # Streamlit app tests do). `python -m m2vtrain` itself has a guarded __main__.
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    texts = ["Bruch", "Photosynthese", "Zellatmung", "Gleichung", "Brüche"]
    single = encode_with_teacher(texts, TeacherOptions(model_name=tiny_teacher, device="cpu"), cache_dir=tmp_path / "a")
    multi = encode_with_teacher(
        texts,
        TeacherOptions(model_name=tiny_teacher, device="cpu", devices=("cpu", "cpu:0"), batch_size=2),
        cache_dir=tmp_path / "b",
        chunk_size=3,
    )
    np.testing.assert_allclose(multi, single, atol=1e-5)
    assert not list((tmp_path / "b").glob("*.partial.npy"))


def test_devices_do_not_change_the_cache_key_and_must_be_distinct(tiny_teacher):
    one = TeacherOptions(model_name=tiny_teacher)
    two = TeacherOptions(model_name=tiny_teacher, devices=["cuda:0", "cuda:1"])
    assert two.devices == ("cuda:0", "cuda:1")
    assert targets._cache_key(["Bruch"], one) == targets._cache_key(["Bruch"], two)
    with pytest.raises(ValueError, match="repeat"):
        TeacherOptions(model_name=tiny_teacher, devices=("cuda:0", "cuda:0"))
