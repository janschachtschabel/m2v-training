import time
from datetime import datetime, timedelta, timezone

import m2vtrain.status as status_module
from m2vtrain.status import RunState, StatusWriter, is_stale, mark_cancelled, read_status


def test_status_lifecycle(tmp_path):
    writer = StatusWriter(tmp_path, stages=["Daten", "Training"])
    writer.stage("Daten")
    writer.progress(5, 10, "halb")
    status = read_status(tmp_path)
    assert (status.state, status.stage, status.stage_index, status.progress) == (RunState.RUNNING, "Daten", 1, 0.5)

    writer.finish()
    assert read_status(tmp_path).state is RunState.FINISHED


def test_failure_records_the_error(tmp_path):
    writer = StatusWriter(tmp_path, stages=["Daten"])
    writer.fail(RuntimeError("kaputt"))
    status = read_status(tmp_path)
    assert status.state is RunState.FAILED
    assert "kaputt" in status.error


def test_missing_status_file_reads_as_none(tmp_path):
    assert read_status(tmp_path) is None


def test_corrupt_status_file_reads_as_none(tmp_path):
    (tmp_path / "status.json").write_text("", encoding="utf-8")
    assert read_status(tmp_path) is None
    (tmp_path / "status.json").write_text('{"state": "exploded"}', encoding="utf-8")
    assert read_status(tmp_path) is None


def test_running_status_goes_stale_when_the_heartbeat_stops(tmp_path):
    writer = StatusWriter(tmp_path, stages=["Daten"])
    writer.stage("Daten")
    status = read_status(tmp_path)
    now = datetime.now(timezone.utc)
    assert not is_stale(status, now=now)
    assert is_stale(status, now=now + timedelta(minutes=10))
    writer.finish()
    assert not is_stale(read_status(tmp_path), now=now + timedelta(minutes=10))


def test_heartbeat_refreshes_without_touching_updated_at(tmp_path):
    writer = StatusWriter(tmp_path, stages=["Daten"], heartbeat_seconds=0.02)
    writer.stage("Daten")
    first = read_status(tmp_path)
    time.sleep(0.2)
    second = read_status(tmp_path)
    writer.finish()

    assert second.heartbeat_at > first.heartbeat_at
    assert second.updated_at == first.updated_at
    assert not writer._heartbeat.is_alive()


def test_failed_progress_writes_do_not_end_the_run(tmp_path, monkeypatch):
    writer = StatusWriter(tmp_path, stages=["Daten"])

    def locked(*args, **kwargs):
        raise PermissionError("locked by virus scanner")

    monkeypatch.setattr(status_module, "write_status", locked)
    writer.stage("Daten")
    writer.progress(1, 2)
    monkeypatch.undo()
    writer.finish()
    assert read_status(tmp_path).state is RunState.FINISHED


def test_mark_cancelled_only_changes_running_runs(tmp_path):
    writer = StatusWriter(tmp_path / "a", stages=["Daten"], heartbeat_seconds=60)
    writer._stop.set()
    writer._heartbeat.join()  # simulate a killed process: no more heartbeats
    mark_cancelled(tmp_path / "a", "gestoppt")
    assert read_status(tmp_path / "a").state is RunState.CANCELLED

    done = StatusWriter(tmp_path / "b", stages=["Daten"])
    done.finish()
    mark_cancelled(tmp_path / "b")
    assert read_status(tmp_path / "b").state is RunState.FINISHED
