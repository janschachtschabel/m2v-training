"""Run status shared between the training process and the UI: `<run_dir>/status.json`, rewritten atomically.

A background thread refreshes a heartbeat while the run is alive, so a reader can tell a killed or crashed
process (stale heartbeat) from a slow stage without probing the process itself.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from m2vtrain.fsutil import replace_with_retry

logger = logging.getLogger(__name__)

STATUS_FILE = "status.json"
HEARTBEAT_SECONDS = 20.0
STALE_AFTER = timedelta(minutes=3)
_REPLACE_RETRIES = 20
_FINAL_WRITE_SECONDS = 10.0


class RunState(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Status:
    state: RunState
    stages: list[str]
    stage: str = ""
    stage_index: int = 0  # 1-based; 0 before the first stage
    progress: float | None = None  # within the current stage, 0..1; None if unknown
    message: str = ""
    error: str = ""
    pid: int = field(default_factory=os.getpid)
    started_at: str = ""
    updated_at: str = ""
    heartbeat_at: str = ""
    finished_at: str = ""
    model_dir: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Status:
        """:raises ValueError, KeyError, TypeError: on a malformed status (e.g. written by hand or another tool)."""
        known = {name: raw[name] for name in cls.__dataclass_fields__ if name in raw}
        status = cls(**{**known, "state": RunState(raw["state"])})
        if status.heartbeat_at and datetime.fromisoformat(status.heartbeat_at).tzinfo is None:
            raise ValueError(f"heartbeat_at without time zone: {status.heartbeat_at}")
        return status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_status(run_dir: Path, status: Status) -> None:
    path = Path(run_dir) / STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{STATUS_FILE}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        file.write(json.dumps(asdict(status), ensure_ascii=False, indent=2))
        file.flush()
        os.fsync(file.fileno())  # so a power loss cannot leave an empty status file behind the replace
    replace_with_retry(tmp, path)


def read_status(run_dir: Path) -> Status | None:
    """The run's status, or None if the run has not written one (yet) or the file is unreadable."""
    path = Path(run_dir) / STATUS_FILE
    for attempt in range(_REPLACE_RETRIES):
        try:
            return Status.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            logger.warning("Unreadable status file %s: %s", path, error)
            return None
        except PermissionError:  # Windows: the writer is just replacing the file
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05)
    return None


def is_stale(status: Status, now: datetime | None = None) -> bool:
    """True if the status says running but the process stopped refreshing its heartbeat (killed or crashed)."""
    if status.state is not RunState.RUNNING or not status.heartbeat_at:
        return False
    now = now or datetime.now(timezone.utc)
    return now - datetime.fromisoformat(status.heartbeat_at) > STALE_AFTER


def mark_cancelled(run_dir: Path, message: str = "") -> None:
    """Record a cancellation from outside the run (the UI after stopping the process).

    Call it only after the process has exited; a still-living process would overwrite it with its next write.
    """
    status = read_status(run_dir)
    if status is None or status.state is not RunState.RUNNING:
        return
    status.state = RunState.CANCELLED
    status.message = message
    status.finished_at = status.updated_at = _now()
    write_status(run_dir, status)


class StatusWriter:
    """Writes the status of the current process's run and keeps its heartbeat fresh until the run ends."""

    def __init__(self, run_dir: Path, stages: Sequence[str], heartbeat_seconds: float = HEARTBEAT_SECONDS):
        self.run_dir = Path(run_dir)
        now = _now()
        self._status = Status(
            state=RunState.RUNNING, stages=list(stages), started_at=now, updated_at=now, heartbeat_at=now
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._write()
        self._heartbeat = threading.Thread(target=self._beat, args=(heartbeat_seconds,), daemon=True)
        self._heartbeat.start()

    @property
    def status(self) -> Status:
        return self._status

    def _write(self, changed: bool = True) -> None:
        """Write the status; `changed` also moves `updated_at` (heartbeats only move `heartbeat_at`)."""
        with self._lock:
            now = _now()
            self._status.heartbeat_at = now
            if changed:
                self._status.updated_at = now
            write_status(self.run_dir, self._status)

    def _update(self, **fields) -> None:
        """Change fields and write them, best effort: a locked file (virus scanner, sync tool) must not end the
        run; the next write or heartbeat catches up."""
        with self._lock:
            for name, value in fields.items():
                setattr(self._status, name, value)
        try:
            self._write()
        except OSError as error:
            logger.warning("Could not write status: %s", error)

    def _beat(self, seconds: float) -> None:
        while not self._stop.wait(seconds):
            try:
                self._write(changed=False)
            except OSError as error:  # a missed heartbeat must not kill the run
                logger.warning("Could not refresh status heartbeat: %s", error)

    def stage(self, name: str) -> None:
        stages = self._status.stages
        index = stages.index(name) + 1 if name in stages else self._status.stage_index
        self._update(stage=name, stage_index=index, progress=None, message="")

    def progress(self, done: int, total: int, message: str = "") -> None:
        self._update(progress=done / total if total else None, message=message)

    def _end(self, state: RunState, error: str = "", message: str = "") -> None:
        self._stop.set()
        self._heartbeat.join()
        with self._lock:
            self._status.state = state
            self._status.error = error
            if message:
                self._status.message = message
            self._status.finished_at = _now()
        # The final state matters most: keep retrying for a while before giving up.
        deadline = time.monotonic() + _FINAL_WRITE_SECONDS
        while True:
            try:
                self._write()
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)

    def finish(self, model_dir: Path | None = None) -> None:
        if model_dir is not None:
            self._status.model_dir = str(model_dir)
        self._end(RunState.FINISHED, message="Fertig")

    def fail(self, error: BaseException) -> None:
        details = "".join(traceback.format_exception_only(type(error), error)).strip()
        self._end(RunState.FAILED, error=details)

    def cancel(self) -> None:
        self._end(RunState.CANCELLED, message="Abgebrochen")
