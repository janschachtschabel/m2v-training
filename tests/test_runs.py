import json
import os
import time

import pandas as pd
import pytest

from m2vtrain.config import RunConfig, save_config
from m2vtrain.runs import (
    describe_state,
    inspect_export,
    list_data_files,
    list_models,
    list_runs,
    results_frame,
    start_run,
    stop_run,
    tail,
)
from m2vtrain.status import RunState, StatusWriter, read_status

from tests.conftest import make_row, write_wlo_export


def _fake_run(runs_dir, run_id, name="lauf", metrics=None, state="finish"):
    run_dir = runs_dir / run_id
    save_config(RunConfig(name=name, data_files=["data/x.csv.gz"]), run_dir / "config.json")
    writer = StatusWriter(run_dir, stages=["Daten"])
    if state == "finish":
        writer.finish()
    if metrics is not None:
        results = {"model": {"raw": metrics, "lemma": metrics}, "baselines": {"tfidf": {"raw": {"keyword_ndcg@10": 0.2}}}}
        (run_dir / "eval.json").write_text(json.dumps(results), encoding="utf-8")
    return run_dir


def test_list_data_files_finds_csv_exports(tmp_path):
    (tmp_path / "a.csv.gz").write_bytes(b"x")
    (tmp_path / "b.csv").write_bytes(b"xy")
    (tmp_path / "notes.txt").write_bytes(b"x")
    files = list_data_files(tmp_path)
    assert [f.path.name for f in files] == ["a.csv.gz", "b.csv"]
    assert files[1].size_bytes == 2
    assert list_data_files(tmp_path / "missing") == []


def test_inspect_export_counts_rows_and_fields(tmp_path):
    path = write_wlo_export(
        tmp_path / "e.csv.gz", [make_row("A", "Text", "Bruch", "de"), make_row("B", "", "", "en"), make_row("C", "Text", "", "de")]
    )
    info = inspect_export(path)
    assert info["rows"] == 3
    assert info["with_description"] == 2 and info["with_keywords"] == 1
    assert info["languages"] == {"de": 2, "en": 1}
    assert info["columns"] == list(pd.read_csv(path, sep=";", nrows=0).columns)
    assert "properties.cclom:title" in info["columns"]


def test_list_runs_newest_first_with_state_labels(tmp_path):
    _fake_run(tmp_path, "20260101-000000-a", metrics={"keyword_ndcg@10": 0.5})
    time.sleep(0.01)
    _fake_run(tmp_path, "20260102-000000-b", state="running")
    (tmp_path / "not-a-run").mkdir()

    runs = list_runs(tmp_path)
    assert [run.run_id for run in runs] == ["20260102-000000-b", "20260101-000000-a"]
    assert runs[0].status.state is RunState.RUNNING
    assert runs[1].has_eval and not runs[0].has_eval
    assert describe_state(runs[1].status) == "fertig"
    assert describe_state(None) == "unbekannt"


def test_results_frame_has_one_row_per_run_encoder_and_mode(tmp_path):
    _fake_run(tmp_path, "r1", name="eins", metrics={"keyword_ndcg@10": 0.5})
    _fake_run(tmp_path, "r2", name="zwei", metrics={"keyword_ndcg@10": 0.6})

    frame = results_frame(list_runs(tmp_path), mode="raw")
    model_rows = frame[frame["Encoder"] == "Modell"]
    assert sorted(model_rows["Lauf"]) == ["r1", "r2"]
    assert set(frame["Encoder"]) == {"Modell", "tfidf"}
    assert model_rows.set_index("Lauf").loc["r2", "keyword_ndcg@10"] == 0.6
    assert results_frame([], mode="raw").empty


def test_list_models_reads_size_and_metrics(tmp_path):
    model = tmp_path / "r1"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"0" * 2048)
    (model / "eval_results.json").write_text(json.dumps({"model": {"raw": {"keyword_ndcg@10": 0.4}}}), encoding="utf-8")
    (tmp_path / "empty").mkdir()

    models = list_models(tmp_path)
    assert [m.name for m in models] == ["r1"]
    assert models[0].size_bytes >= 2048
    assert models[0].metrics == {"keyword_ndcg@10": 0.4}


def test_tail_returns_the_last_lines(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("\n".join(f"Zeile {i}" for i in range(1000)), encoding="utf-8")
    assert tail(path, lines=2) == "Zeile 998\nZeile 999"
    assert tail(tmp_path / "missing.txt") == ""


def test_started_run_reports_failure_from_the_subprocess(tmp_path):
    config = RunConfig(name="kaputt", data_files=["data/fehlt.csv.gz"])
    run_dir = start_run(config, root=tmp_path)

    assert run_dir.parent == tmp_path / "runs"
    assert read_status(run_dir).state is RunState.RUNNING  # written before the process starts
    deadline = time.time() + 120
    while read_status(run_dir).state is RunState.RUNNING and time.time() < deadline:
        time.sleep(0.5)
    status = read_status(run_dir)
    assert status.state is RunState.FAILED
    assert "fehlt.csv.gz" in status.error
    assert "fehlt.csv.gz" in (run_dir / "run.log").read_text(encoding="utf-8")


def test_started_runs_get_distinct_directories(tmp_path, monkeypatch):
    import m2vtrain.runs as runs

    monkeypatch.setattr(runs, "_launch", lambda run_dir, root: _sleeper())
    config = RunConfig(name="gleich", data_files=["a.csv"])
    assert start_run(config, root=tmp_path) != start_run(config, root=tmp_path)


def test_stop_run_ignores_runs_that_are_not_running(tmp_path):
    run_dir = _fake_run(tmp_path, "done")
    assert stop_run(run_dir) is False
    assert read_status(run_dir).state is RunState.FINISHED


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal path")
def test_stop_run_cancels_a_running_process(tmp_path):
    import subprocess
    import sys

    from m2vtrain.status import Status, write_status

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    run_dir = tmp_path / "running"
    write_status(run_dir, Status(state=RunState.RUNNING, stages=["Daten"], pid=child.pid, heartbeat_at=_now()))
    import m2vtrain.runs as runs

    runs._PROCESSES[run_dir] = child  # as if started by the UI, which reaps its own children

    assert stop_run(run_dir, wait_seconds=2) is True
    assert child.wait(timeout=10) != 0
    assert read_status(run_dir).state is RunState.CANCELLED


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def test_list_models_ignores_unfinished_model_folders(tmp_path):
    partial = tmp_path / "r1.partial"
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"x")
    assert list_models(tmp_path) == []


def _sleeper():
    import subprocess
    import sys

    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)


def test_relative_root_is_resolved(tmp_path, monkeypatch):
    import m2vtrain.runs as runs

    monkeypatch.chdir(tmp_path)
    launched = []
    monkeypatch.setattr(runs, "_launch", lambda run_dir, root: launched.append((run_dir, root)) or _sleeper())
    run_dir = start_run(RunConfig(name="rel", data_files=["a.csv"]), root="arbeit")

    assert run_dir.is_absolute() and (run_dir / "config.json").exists()
    assert launched[0][1] == tmp_path / "arbeit"
    stop_run(run_dir, wait_seconds=1)


def test_failed_launch_is_recorded_without_a_pid(tmp_path, monkeypatch):
    import m2vtrain.runs as runs

    def broken(run_dir, root):
        raise OSError("no python")

    monkeypatch.setattr(runs, "_launch", broken)
    with pytest.raises(OSError):
        start_run(RunConfig(name="x", data_files=["a.csv"]), root=tmp_path)
    (run_dir,) = (tmp_path / "runs").iterdir()
    status = read_status(run_dir)
    assert status.state is RunState.FAILED and "no python" in status.error and status.pid == 0


def test_process_that_dies_before_its_first_status_is_marked_failed(tmp_path, monkeypatch):
    import subprocess
    import sys

    import m2vtrain.runs as runs

    def crashing(run_dir, root):
        with open(run_dir / "console.log", "wb") as console:
            return subprocess.Popen([sys.executable, "-c", "print('ImportError: kaputt'); raise SystemExit(3)"], stdout=console)

    monkeypatch.setattr(runs, "_launch", crashing)
    run_dir = start_run(RunConfig(name="x", data_files=["a.csv"]), root=tmp_path)
    runs._PROCESSES[run_dir].wait(timeout=30)

    (run,) = list_runs(tmp_path / "runs")
    assert run.status.state is RunState.FAILED
    assert "kaputt" in run.status.error


def test_corrupt_files_do_not_break_listings(tmp_path):
    _fake_run(tmp_path / "runs", "good", metrics={"keyword_ndcg@10": 0.5})
    bad = _fake_run(tmp_path / "runs", "bad")
    (bad / "eval.json").write_text('{"model": ', encoding="utf-8")
    empty = _fake_run(tmp_path / "runs", "empty")
    (empty / "eval.json").write_text("[]", encoding="utf-8")
    naive = _fake_run(tmp_path / "runs", "naive")
    status = json.loads((naive / "status.json").read_text(encoding="utf-8"))
    (naive / "status.json").write_text(json.dumps({**status, "heartbeat_at": "2026-01-01T00:00:00"}), encoding="utf-8")

    runs_list = list_runs(tmp_path / "runs")
    assert {run.run_id for run in runs_list} == {"good", "bad", "empty", "naive"}
    assert [describe_state(run.status) for run in runs_list if run.run_id == "naive"] == ["unbekannt"]
    assert set(results_frame(runs_list, mode="raw")["Lauf"]) == {"good"}

    model = tmp_path / "models" / "m"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"x")
    (model / "eval_results.json").write_text("kaputt", encoding="utf-8")
    assert list_models(tmp_path / "models")[0].metrics == {}


def test_results_frame_shows_the_test_split(tmp_path):
    run_dir = _fake_run(tmp_path, "r1", metrics={"keyword_ndcg@10": 0.5})
    results = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    results["test_split"] = {"documents": 10, "fingerprint": "abc"}
    (run_dir / "eval.json").write_text(json.dumps(results), encoding="utf-8")
    assert set(results_frame(list_runs(tmp_path), mode="raw")["Testsplit"]) == {"abc (10)"}


def test_save_data_file_accepts_only_wlo_exports(tmp_path):
    from m2vtrain.runs import save_data_file

    (tmp_path / "src").mkdir()
    export = write_wlo_export(tmp_path / "src" / "e.csv.gz", [make_row("A", "Text", "Bruch", "de")])
    data = tmp_path / "data"
    assert save_data_file("e.csv.gz", export.read_bytes(), data) == data / "e.csv.gz"
    with pytest.raises(FileExistsError):
        save_data_file("e.csv.gz", export.read_bytes(), data)
    with pytest.raises(ValueError, match="cclom:title"):
        save_data_file("fremd.csv", b"a;b\n1;2\n", data)
    with pytest.raises(ValueError):
        save_data_file("kaputt.csv.gz", b"not gzip", data)
    with pytest.raises(ValueError):
        save_data_file("notiz.txt", b"x", data)
    assert sorted(p.name for p in data.iterdir()) == ["e.csv.gz"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal path")
def test_stop_lets_a_real_run_record_its_cancellation(tmp_path, tiny_teacher):
    config = RunConfig.from_dict(
        {
            "name": "stopp",
            "data_files": ["data/e.csv.gz"],
            "teacher": {"model_name": tiny_teacher, "trust_remote_code": False, "device": "cpu"},
            "pca_dims": 16,
        }
    )
    (tmp_path / "data").mkdir()
    write_wlo_export(tmp_path / "data" / "e.csv.gz", [make_row(f"Titel {i}", "Text", "Bruch", "de") for i in range(50)])
    run_dir = start_run(config, root=tmp_path)
    deadline = time.time() + 120
    while (read_status(run_dir).pid == 0 or read_status(run_dir).stage == "Starten") and time.time() < deadline:
        time.sleep(0.2)

    assert stop_run(run_dir, wait_seconds=30) is True
    status = read_status(run_dir)
    assert status.state in (RunState.CANCELLED, RunState.FINISHED)  # a tiny run may finish before the signal
    assert status.message != "Vom Benutzer gestoppt" or status.state is RunState.CANCELLED
