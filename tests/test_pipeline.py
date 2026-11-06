import json

import pytest

from m2vtrain.cli import main
from m2vtrain.config import RunConfig, load_config, save_config
from m2vtrain.pipeline import run_pipeline
from m2vtrain.status import RunState, read_status

from tests.conftest import TEACHER_WORDS, make_row, write_wlo_export

TOPICS = [("Bruch", "Mathematik", "380"), ("Licht", "Physik", "460"), ("Pflanze", "Biologie", "080")]


@pytest.fixture
def small_export(tmp_path):
    rows = []
    for i in range(90):
        word, subject, code = TOPICS[i % 3]
        other = TEACHER_WORDS[i % len(TEACHER_WORDS)]
        rows.append(
            make_row(
                f"{word} und {other} Teil {i}",
                f"Die {word} der {other} im Unterricht, Abschnitt {i}.",
                f"{word}, {other}",
                "de",
                f"http://w3id.org/openeduhub/vocabs/discipline/{code}",
                subject,
            )
        )
    return write_wlo_export(tmp_path / "small.csv.gz", rows)


def _config(tiny_teacher, export, tmp_path, **overrides) -> RunConfig:
    raw = {
        "name": "klein",
        "data_files": [str(export)],
        "test_fraction": 0.3,
        "pca_dims": 16,
        "input_mode": "mixed",
        "teacher": {"model_name": tiny_teacher, "device": "cpu"},
        "vocab": {"min_word_count": 2, "keyword_min_docs": 2},
        "tokenlearn": {"enabled": True, "train": {"max_epochs": 3, "device": "cpu"}},
        "pairs": {"enabled": True, "train": {"max_epochs": 2, "device": "cpu"}},
        "eval": {"min_examples_per_subject": 5},
        "runs_dir": str(tmp_path / "runs"),
        "models_dir": str(tmp_path / "models"),
        "cache_dir": str(tmp_path / "cache"),
    }
    raw.update(overrides)
    return RunConfig.from_dict(raw)


def test_pipeline_trains_evaluates_and_exports(tiny_teacher, small_export, tmp_path):
    config = _config(tiny_teacher, small_export, tmp_path)
    run_dir = tmp_path / "runs" / "r1"

    model_dir = run_pipeline(config, run_dir)

    assert (model_dir / "model.safetensors").exists()
    card = (model_dir / "README.md").read_text(encoding="utf-8")
    assert "keyword_ndcg@10" in card and "klein" in card
    results = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert set(results["model"]) == {"raw", "lemma"}
    assert "tfidf" in results["baselines"]
    assert (run_dir / "vocabulary.txt").exists()
    assert read_status(run_dir).state is RunState.FINISHED


def test_pipeline_failure_is_recorded(tiny_teacher, small_export, tmp_path):
    config = _config(tiny_teacher, small_export, tmp_path, teacher={"model_name": str(tmp_path / "missing")})
    run_dir = tmp_path / "runs" / "r2"

    with pytest.raises(OSError):
        run_pipeline(config, run_dir)
    status = read_status(run_dir)
    assert status.state is RunState.FAILED and status.error


def test_cli_init_and_run(tiny_teacher, small_export, tmp_path):
    config_path = tmp_path / "config.json"
    assert main(["init-config", str(config_path)]) == 0
    assert config_path.exists()

    save_config(_config(tiny_teacher, small_export, tmp_path, pairs={"enabled": False}), config_path)
    run_dir = tmp_path / "runs" / "cli"
    assert main(["run", str(config_path), "--run-dir", str(run_dir)]) == 0
    assert read_status(run_dir).state is RunState.FINISHED


def test_resolve_data_files_expands_globs_and_rejects_missing_files(tmp_path):
    from m2vtrain.pipeline import resolve_data_files

    (tmp_path / "a.csv.gz").touch()
    (tmp_path / "b.csv.gz").touch()
    assert [p.name for p in resolve_data_files((str(tmp_path / "*.csv.gz"),))] == ["a.csv.gz", "b.csv.gz"]
    with pytest.raises(FileNotFoundError):
        resolve_data_files((str(tmp_path / "missing.csv"),))
    with pytest.raises(FileNotFoundError):
        resolve_data_files((str(tmp_path / "*.json"),))


def test_pipeline_refuses_to_overwrite_a_model(tiny_teacher, small_export, tmp_path):
    config = _config(tiny_teacher, small_export, tmp_path)
    existing = tmp_path / "models" / "r3"
    existing.mkdir(parents=True)
    (existing / "model.safetensors").write_bytes(b"old")

    with pytest.raises(FileExistsError):
        run_pipeline(config, tmp_path / "runs" / "r3")
    assert (existing / "model.safetensors").read_bytes() == b"old"
    assert read_status(tmp_path / "runs" / "r3").state is RunState.FAILED


def test_interrupted_pipeline_is_recorded_as_cancelled(tiny_teacher, small_export, tmp_path, monkeypatch):
    import m2vtrain.pipeline as pipeline

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "distill_model", interrupt)
    run_dir = tmp_path / "runs" / "r4"
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(_config(tiny_teacher, small_export, tmp_path), run_dir)
    assert read_status(run_dir).state is RunState.CANCELLED


def test_cli_reports_invalid_config(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"name": "x", "data_files": ["a.csv"], "pca_dim": 3}', encoding="utf-8")
    assert main(["run", str(path)]) == 1
    assert "pca_dim" in capsys.readouterr().err


def test_cli_upload_rejects_folder_without_model(tmp_path, capsys):
    assert main(["upload", str(tmp_path), "jan/x"]) == 1
    assert "model.safetensors" in capsys.readouterr().err


def test_model_folder_appears_only_after_a_successful_run(tiny_teacher, small_export, tmp_path, monkeypatch):
    import m2vtrain.pipeline as pipeline

    def broken(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(pipeline, "write_model_card", broken)
    with pytest.raises(RuntimeError):
        run_pipeline(_config(tiny_teacher, small_export, tmp_path), tmp_path / "runs" / "r5")
    assert not (tmp_path / "models" / "r5").exists()
    assert read_status(tmp_path / "runs" / "r5").state is RunState.FAILED


def test_published_training_config_has_no_local_paths(tiny_teacher, small_export, tmp_path):
    model_dir = run_pipeline(_config(tiny_teacher, small_export, tmp_path, pairs={"enabled": False}), tmp_path / "runs" / "r6")
    published = json.loads((model_dir / "training_config.json").read_text(encoding="utf-8"))
    text = json.dumps(published)
    assert str(tmp_path) not in text
    assert published["data_files"] == ["<local file>"]  # names of local exports are not published
    results = json.loads((tmp_path / "runs" / "r6" / "eval.json").read_text(encoding="utf-8"))
    assert results["test_split"]["documents"] > 0 and results["test_split"]["fingerprint"]
    assert results["eval_options"]["probe_c"] == 10.0


def test_cli_records_invalid_config_in_the_run_dir(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"name": "x", "data_files": []}', encoding="utf-8")
    run_dir = tmp_path / "runs" / "bad"
    assert main(["run", str(path), "--run-dir", str(run_dir)]) == 1
    status = read_status(run_dir)
    assert status.state is RunState.FAILED and "data_files" in status.error


def test_new_run_dirs_are_unique(tmp_path):
    from m2vtrain.cli import new_run_dir

    config = RunConfig(name="gleich/name", data_files=["a.csv"], runs_dir=str(tmp_path))
    first = new_run_dir(config)
    second = new_run_dir(config)
    assert first != second and second.parent == tmp_path and "/" not in first.name


def test_invalid_baseline_fails_before_training(tiny_teacher, small_export, tmp_path, monkeypatch):
    import m2vtrain.pipeline as pipeline

    def must_not_train(*args, **kwargs):
        raise AssertionError("distillation started despite an invalid baseline")

    monkeypatch.setattr(pipeline, "distill_model", must_not_train)
    config = _config(tiny_teacher, small_export, tmp_path, baseline_models=[str(tmp_path / "gibt-es-nicht")])
    with pytest.raises(Exception) as raised:
        run_pipeline(config, tmp_path / "runs" / "r7")
    assert not isinstance(raised.value, AssertionError)
    assert read_status(tmp_path / "runs" / "r7").stage == "Daten vorbereiten"


def test_trained_model_is_kept_when_a_late_stage_fails(tiny_teacher, small_export, tmp_path, monkeypatch):
    import m2vtrain.pipeline as pipeline

    def broken(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(pipeline, "_evaluate_all", broken)
    with pytest.raises(RuntimeError):
        run_pipeline(_config(tiny_teacher, small_export, tmp_path, pairs={"enabled": False}), tmp_path / "runs" / "r8")
    assert (tmp_path / "models" / "r8.partial" / "model.safetensors").exists()
    assert "r8.partial" in read_status(tmp_path / "runs" / "r8").error


def test_model_folder_rename_retries_while_windows_holds_a_file(tmp_path, monkeypatch):
    from pathlib import Path

    from m2vtrain.fsutil import replace_with_retry

    source, target = tmp_path / "a.partial", tmp_path / "a"
    source.mkdir()
    real_replace, calls = Path.replace, []

    def flaky(self, other):
        calls.append(self)
        if len(calls) < 3:
            raise PermissionError("in use")
        return real_replace(self, other)

    monkeypatch.setattr(Path, "replace", flaky)
    replace_with_retry(source, target, seconds=5, pause=0.01)
    assert target.is_dir() and len(calls) == 3


def test_transformer_comparison_models_are_evaluated(tiny_teacher, small_export, tmp_path):
    config = _config(tiny_teacher, small_export, tmp_path, baseline_models=[tiny_teacher], pairs={"enabled": False})
    run_pipeline(config, tmp_path / "runs" / "r8")
    results = json.loads((tmp_path / "runs" / "r8" / "eval.json").read_text(encoding="utf-8"))
    assert set(results["baselines"][tiny_teacher]) == {"raw", "lemma"}


def test_comparison_models_are_loaded_one_at_a_time(tiny_teacher, small_export, tmp_path, monkeypatch):
    import numpy as np

    import m2vtrain.pipeline as pipeline

    events = []

    def fake_load(name, device=None, batch_size=64):
        events.append(f"load {name}")
        return lambda texts: np.ones((len(texts), 4), dtype=np.float32)

    real_evaluate = pipeline.evaluate

    def recording_evaluate(encode, train_df, test_df, options):
        events.append("evaluate")
        return real_evaluate(encode, train_df, test_df, options)

    monkeypatch.setattr(pipeline, "is_model2vec", lambda name: True)
    monkeypatch.setattr(pipeline, "load_baseline", fake_load)
    monkeypatch.setattr(pipeline, "evaluate", recording_evaluate)
    config = _config(tiny_teacher, small_export, tmp_path, baseline_models=["a", "b"], pairs={"enabled": False})
    run_pipeline(config, tmp_path / "runs" / "r9")
    baseline_events = events[events.index("load a") :]
    assert baseline_events == ["load a", "evaluate", "evaluate", "load b", "evaluate", "evaluate"]


def _synthetic_config(tiny_teacher, dataset, tmp_path, **overrides) -> RunConfig:
    keywords = tmp_path / "keywords.txt"
    keywords.write_text("Satz des Pythagoras\nPhotosynthese\nBrüche\n", encoding="utf-8")
    raw = {
        "name": "synth",
        "synthetic": {"enabled": True, "dataset": str(dataset)},
        "keyword_lists": {"files": [str(keywords)]},
        "test_fraction": 0.3,
        "pca_dims": 16,
        "teacher": {"model_name": tiny_teacher, "device": "cpu", "max_seq_length": 64},
        "vocab": {"min_word_count": 2, "keyword_min_docs": 2},
        "tokenlearn": {"enabled": True, "train": {"max_epochs": 2, "device": "cpu"}},
        "pairs": {"enabled": True, "train": {"max_epochs": 1, "device": "cpu"}},
        "eval": {"min_examples_per_subject": 5},
        "runs_dir": str(tmp_path / "runs"),
        "models_dir": str(tmp_path / "models"),
        "cache_dir": str(tmp_path / "cache"),
    }
    raw.update(overrides)
    return RunConfig.from_dict(raw)


def test_synthetic_dataset_and_keyword_list_single_split(tiny_teacher, synthetic_dataset, tmp_path):
    config = _synthetic_config(tiny_teacher, synthetic_dataset, tmp_path)
    run_dir = tmp_path / "runs" / "s1"

    model_dir = run_pipeline(config, run_dir)

    results = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert results["model"]["raw"]["question_queries"] > 0
    assert "question_mrr@10" in results["model"]["lemma"]
    assert results["facts"]["synthetic_documents"] == 60 and results["facts"]["keyword_list_terms"] == 3
    vocabulary = (run_dir / "vocabulary.txt").read_text(encoding="utf-8").splitlines()
    assert "Satz des Pythagoras" in vocabulary and "photosynthese" in vocabulary
    card = (model_dir / "README.md").read_text(encoding="utf-8")
    assert "terms only" in card and "question_mrr@10" in card


def test_cross_validation_trains_k_models_and_a_final_one(tiny_teacher, synthetic_dataset, tmp_path, monkeypatch):
    import m2vtrain.pipeline as pipeline

    encoded = []
    original = pipeline.encode_with_teacher
    monkeypatch.setattr(pipeline, "encode_with_teacher", lambda texts, *a, **k: encoded.append(len(texts)) or original(texts, *a, **k))
    config = _synthetic_config(tiny_teacher, synthetic_dataset, tmp_path, cross_validation={"folds": 3})
    run_dir = tmp_path / "runs" / "cv"

    model_dir = run_pipeline(config, run_dir)

    assert len(encoded) == 1  # one teacher pass for all folds and the final model
    assert (model_dir / "model.safetensors").exists()
    folds = [json.loads((run_dir / "folds" / f"fold-{i}" / "eval.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    assert sum(fold["test_split"]["documents"] for fold in folds) == 60  # every document is tested exactly once
    results = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert results["cv"]["folds"] == 3 and results["test_split"]["folds"] == 3
    values = results["cv"]["values"]["model"]["raw"]["title_mrr@10"]
    assert len(values) == 3 and abs(results["model"]["raw"]["title_mrr@10"] - sum(values) / 3) < 1e-9
    assert "tfidf" in results["cv"]["std"]["baselines"]
    status = read_status(run_dir)
    assert status.state is RunState.FINISHED and status.stages[-1] == "Speichern"
    assert "± " in (model_dir / "README.md").read_text(encoding="utf-8")


def test_cross_validation_without_final_model_saves_no_model(tiny_teacher, synthetic_dataset, tmp_path):
    config = _synthetic_config(
        tiny_teacher,
        synthetic_dataset,
        tmp_path,
        cross_validation={"folds": 2, "final_model": False},
        pairs={"enabled": False},
        keyword_lists={},
    )
    run_dir = tmp_path / "runs" / "cv2"

    assert run_pipeline(config, run_dir) is None
    assert (run_dir / "eval.json").exists()
    assert not (tmp_path / "models").exists() or not any((tmp_path / "models").iterdir())
    assert read_status(run_dir).state is RunState.FINISHED


def test_cli_extract_keywords_writes_only_terms(export_path, tmp_path):
    output = tmp_path / "keywords.txt"
    assert main(["extract-keywords", str(export_path), str(output), "--min-docs", "1"]) == 0
    terms = output.read_text(encoding="utf-8").splitlines()
    assert "Photosynthese" in terms and "Bruchrechnung" in terms
    assert all("\t" not in term for term in terms)
    assert main(["extract-keywords", str(export_path), str(output)]) == 1  # exists, no --force


def test_cli_init_config_for_synthetic_cross_validation(tmp_path):
    path = tmp_path / "c.json"
    assert main(["init-config", str(path), "--synthetic", "--keywords", "data/keywords.txt", "--folds", "5"]) == 0
    config = load_config(path)
    assert config.synthetic.enabled and config.data_files == () and config.cross_validation.folds == 5
    assert config.keyword_lists.files == ("data/keywords.txt",)


def test_cross_validation_with_shared_vocabulary_distills_once(tiny_teacher, synthetic_dataset, tmp_path, monkeypatch):
    import m2vtrain.pipeline as pipeline

    distilled = []
    original = pipeline.distill_model
    monkeypatch.setattr(pipeline, "distill_model", lambda *a, **k: distilled.append(1) or original(*a, **k))
    config = _synthetic_config(
        tiny_teacher, synthetic_dataset, tmp_path, cross_validation={"folds": 2, "shared_vocabulary": True}
    )
    run_dir = tmp_path / "runs" / "shared"

    model_dir = run_pipeline(config, run_dir)

    assert len(distilled) == 1
    assert (model_dir / "model.safetensors").exists() and (run_dir / "vocabulary.txt").exists()
    status = read_status(run_dir)
    assert status.state is RunState.FINISHED
    assert status.stages[:4] == ["Daten vorbereiten", "Lehrer-Embeddings", "Vokabular", "Destillation"]
    folds = [json.loads((run_dir / "folds" / f"fold-{i}" / "eval.json").read_text(encoding="utf-8")) for i in (1, 2)]
    assert folds[0]["model"]["raw"] != folds[1]["model"]["raw"]


def test_cli_compare_prints_cv_runs_with_spread(tmp_path, capsys):
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"teacher": {"model_name": "BAAI/bge-m3"}}), encoding="utf-8")
    results = {
        "model": {"raw": {"keyword_ndcg@10": 0.4}},
        "baselines": {"tfidf": {"raw": {"keyword_ndcg@10": 0.3}}},
        "cv": {"folds": 3, "std": {"model": {"raw": {"keyword_ndcg@10": 0.01}}, "baselines": {}}},
        "test_split": {"fingerprint": "abcdef1234"},
    }
    (run / "eval.json").write_text(json.dumps(results), encoding="utf-8")
    assert main(["compare", str(run)]) == 0
    out = capsys.readouterr().out
    assert "| a | BAAI/bge-m3 | 3 | abcdef12 | 0.400 ± 0.010 |" in out and "tfidf" in out
