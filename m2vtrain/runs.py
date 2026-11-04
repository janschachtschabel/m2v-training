"""Run management for the UI: data files, starting and stopping runs as subprocesses, run and model listings.

Kept free of Streamlit so it can be tested and reused; `ui/app.py` only renders what this module returns.
"""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from m2vtrain.cli import new_run_dir
from m2vtrain.config import RunConfig, load_config, save_config
from m2vtrain.data import REQUIRED_COLUMN, export_columns, load_exports
from m2vtrain.pipeline import PARTIAL_SUFFIX
from m2vtrain.status import RunState, Status, is_stale, mark_cancelled, read_status, write_status

logger = logging.getLogger(__name__)

DATA_PATTERNS = ("*.csv.gz", "*.csv")
KEYWORD_LIST_PATTERNS = ("*.txt", "*.tsv")
_PROCESSES: dict[Path, subprocess.Popen] = {}  # runs started by this process (the UI), to notice early crashes
CONSOLE_LOG = "console.log"
STARTING = "Starten"
STOP_WAIT_SECONDS = 15.0
_STATE_LABELS = {
    RunState.RUNNING: "läuft",
    RunState.FINISHED: "fertig",
    RunState.FAILED: "fehlgeschlagen",
    RunState.CANCELLED: "abgebrochen",
}
# Columns of the comparison table besides the metrics.
_CONFIG_COLUMNS = {
    "Lehrer": lambda c: c.teacher.model_name,
    "Dim": lambda c: c.pca_dims,
    "Eingabe Training": lambda c: c.input_mode,
    "Tokenlearn": lambda c: c.tokenlearn.enabled,
    "Paare": lambda c: c.pairs.enabled,
}


@dataclass(frozen=True)
class DataFile:
    path: Path
    size_bytes: int
    modified: datetime


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    path: Path
    config: RunConfig | None
    status: Status | None

    @property
    def has_eval(self) -> bool:
        return (self.path / "eval.json").exists()

    def results(self) -> dict:
        """eval.json, or {} if it is missing or unreadable."""
        return _read_json(self.path / "eval.json")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    size_bytes: int
    modified: datetime
    metrics: dict  # raw-input metrics of the model, empty if not evaluated


def _read_json(path: Path) -> dict:
    """A JSON object from disk; {} for a missing, corrupt or non-object file (one bad run must not break a page)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        if Path(path).exists():
            logger.warning("Unreadable %s: %s", path, error)
        return {}
    return data if isinstance(data, dict) else {}


def _modified(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def list_data_files(data_dir: Path) -> list[DataFile]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    paths = sorted({path for pattern in DATA_PATTERNS for path in data_dir.glob(pattern)})
    return [DataFile(path, path.stat().st_size, _modified(path)) for path in paths]


def list_keyword_lists(data_dir: Path) -> list[DataFile]:
    """Keyword lists (one term per line) in the data folder."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    paths = sorted({path for pattern in KEYWORD_LIST_PATTERNS for path in data_dir.glob(pattern)})
    return [DataFile(path, path.stat().st_size, _modified(path)) for path in paths]


def inspect_export(path: Path) -> dict:
    """Quick overview of an export before training (no cleaning), including its raw column headers."""
    df = load_exports([path])
    languages = df["language"].replace("", "unbekannt").value_counts()
    return {
        "rows": len(df),
        "columns": export_columns(path),
        "with_description": int((df["description"].str.strip() != "").sum()),
        "with_keywords": int((df["keywords_raw"].str.strip() != "").sum()),
        "languages": {language: int(count) for language, count in languages.items()},
    }


def save_data_file(name: str, content: bytes, data_dir: Path) -> Path:
    """Store an uploaded export in `data_dir` after checking it is a WLO export (written atomically).

    :raises ValueError: if it is not a .csv/.csv.gz file with a title column.
    :raises FileExistsError: if a file of that name exists.
    """
    name = Path(name).name
    if not name.endswith((".csv", ".csv.gz")):
        raise ValueError("Nur .csv oder .csv.gz.")
    target = Path(data_dir) / name
    if target.exists():
        raise FileExistsError(f"{name} gibt es schon.")
    try:
        compression = "gzip" if name.endswith(".gz") else None  # streamed: only the header line is unpacked
        header = pd.read_csv(io.BytesIO(content), sep=";", nrows=0, compression=compression).columns
    except (OSError, EOFError, ValueError, UnicodeDecodeError) as error:  # BadGzipFile is an OSError
        raise ValueError(f"{name} ist keine lesbare CSV-Datei: {error}") from error
    if REQUIRED_COLUMN not in header:
        raise ValueError(f"{name} ist kein WLO-Export: Spalte {REQUIRED_COLUMN} fehlt.")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(name + ".part")
    partial.write_bytes(content)
    partial.replace(target)
    return target


def describe_state(status: Status | None) -> str:
    if status is None:
        return "unbekannt"
    if is_stale(status):
        return "abgebrochen (kein Lebenszeichen)"
    return _STATE_LABELS[status.state]


def _launch(run_dir: Path, root: Path) -> subprocess.Popen:
    """Start the training process detached from the UI, so closing the UI does not end the run."""
    command = [sys.executable, "-m", "m2vtrain", "run", str(run_dir / "config.json"), "--run-dir", str(run_dir)]
    # UTF-8 and unbuffered output, so console.log shows umlauts and progress on Windows too.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    options: dict = {"cwd": root, "env": env, "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    with open(run_dir / CONSOLE_LOG, "wb") as console:
        return subprocess.Popen(command, stdout=console, **options)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def start_run(config: RunConfig, root: Path) -> Path:
    """Write the run's config and an initial status, then start `python -m m2vtrain run` in `root`.

    Relative paths in the config (data files, runs, models, cache) are resolved against `root`.

    :raises OSError: if the process cannot be started (recorded as a failed run).
    """
    root = Path(root).expanduser().resolve()
    run_dir = new_run_dir(config, root)
    save_config(config, run_dir / "config.json")
    now = _now()
    # The process needs a few seconds to import torch; until it writes its own status this one shows the run.
    # pid 0 until the process exists, so a stop can never hit another process.
    initial = Status(state=RunState.RUNNING, stages=[], stage=STARTING, pid=0, started_at=now, updated_at=now, heartbeat_at=now)
    write_status(run_dir, initial)
    try:
        process = _launch(run_dir, root)
    except OSError as error:
        initial.state, initial.error, initial.finished_at = RunState.FAILED, f"Start fehlgeschlagen: {error}", _now()
        write_status(run_dir, initial)
        raise
    _PROCESSES[run_dir] = process
    initial.pid = process.pid
    write_status(run_dir, initial)
    return run_dir


def _record_early_exit(run_dir: Path, status: Status | None) -> Status | None:
    """If a process started here exited before writing its own status, record it as failed with its last output."""
    process = _PROCESSES.get(run_dir)
    if process is None or process.poll() is None:
        return status
    del _PROCESSES[run_dir]
    if status is None or status.state is not RunState.RUNNING or status.stage != STARTING:
        return status
    status.state, status.finished_at = RunState.FAILED, _now()
    status.error = tail(run_dir / CONSOLE_LOG, lines=15) or f"Prozess beendet mit Code {process.returncode}"
    write_status(run_dir, status)
    return status


def run_status(run_dir: Path) -> Status | None:
    """The run's status, with a crash before the process's first own status write recorded as failed."""
    return _record_early_exit(Path(run_dir), read_status(run_dir))


def _signal_process(pid: int, force: bool) -> bool:
    """Send a stop to `pid`; False if that was impossible (no such process, access denied)."""
    if os.name == "nt":
        # Kill the interpreter and everything it started (e.g. data loader workers).
        result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        return result.returncode in (0, 128)  # 128: process already gone
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pid, sig)  # runs started by the UI lead their own process group
    except ProcessLookupError:
        try:
            os.kill(pid, sig)  # a run started from the command line
        except ProcessLookupError:
            return True  # already gone
    except PermissionError:
        return False
    return True


def _exited(pid: int, process: subprocess.Popen | None) -> bool:
    # A child of this process stays a zombie (still "alive" for kill 0) until it is waited for.
    return process.poll() is not None if process is not None else not _posix_alive(pid)


def _posix_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 just checks existence (on Windows it would send CTRL_C)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_run(run_dir: Path, wait_seconds: float = STOP_WAIT_SECONDS) -> bool:
    """Stop a running run and record it as cancelled. Returns False if the run was not running or could not be
    stopped.

    POSIX: SIGTERM lets the run record the cancellation itself; after `wait_seconds` it is killed. Windows has no
    such signal, so the process tree is killed at once.
    """
    status = read_status(run_dir)
    if status is None or status.state is not RunState.RUNNING or is_stale(status) or not status.pid:
        return False
    if not _signal_process(status.pid, force=False):
        return False
    if os.name != "nt":
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            current = read_status(run_dir)
            if current is None or current.state is not RunState.RUNNING:
                return True
            time.sleep(0.2)
        _signal_process(status.pid, force=True)
        deadline = time.monotonic() + 5
        while not _exited(status.pid, _PROCESSES.get(Path(run_dir))) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not _exited(status.pid, _PROCESSES.get(Path(run_dir))):
            return False
    time.sleep(1.0)  # killing is asynchronous; a write still in flight must not land after mark_cancelled
    mark_cancelled(run_dir, "Vom Benutzer gestoppt")
    return True


def _load_run(path: Path) -> RunInfo:
    try:
        config = load_config(path / "config.json")
    except (OSError, ValueError):
        config = None
    return RunInfo(path.name, path, config, run_status(path))


def list_runs(runs_dir: Path) -> list[RunInfo]:
    """Run directories (those with a config.json), newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    paths = [path for path in runs_dir.iterdir() if (path / "config.json").is_file()]
    paths.sort(key=lambda path: (path / "config.json").stat().st_mtime, reverse=True)
    return [_load_run(path) for path in paths]


def results_frame(runs: Sequence[RunInfo], mode: str) -> pd.DataFrame:
    """Comparison table: one row per run and encoder (the model and its baselines) for one evaluation input mode."""
    rows = []
    for run in runs:
        if not run.has_eval:
            continue
        results = run.results()
        encoders = {"Modell": results.get("model", {}), **results.get("baselines", {})}
        settings = {column: get(run.config) for column, get in _CONFIG_COLUMNS.items()} if run.config else {}
        split = results.get("test_split", {})
        split_label = f"{split.get('fingerprint', '?')} ({split.get('documents', '?')})" if split else "?"
        for encoder, by_mode in encoders.items():
            if isinstance(by_mode, dict) and isinstance(by_mode.get(mode), dict):
                name = run.config.name if run.config else ""
                rows.append(
                    {"Lauf": run.run_id, "Name": name, "Encoder": encoder, "Testsplit": split_label, **settings, **by_mode[mode]}
                )
    return pd.DataFrame(rows)


def _dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def list_models(models_dir: Path) -> list[ModelInfo]:
    """Finished models (directories with model.safetensors), newest first."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    models = []
    for path in models_dir.iterdir():
        weights = path / "model.safetensors"
        if path.name.endswith(PARTIAL_SUFFIX) or not weights.is_file():  # still being built
            continue
        metrics = _read_json(path / "eval_results.json").get("model", {})
        metrics = metrics.get("raw", {}) if isinstance(metrics, dict) else {}
        models.append(ModelInfo(path.name, path, _dir_size(path), _modified(weights), metrics))
    return sorted(models, key=lambda model: model.modified, reverse=True)


def tail(path: Path, lines: int = 200, max_bytes: int = 200_000) -> str:
    """The last `lines` lines of a (possibly growing) log file; empty if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return ""
    with open(path, "rb") as file:
        file.seek(max(path.stat().st_size - max_bytes, 0))
        text = file.read().decode("utf-8", errors="replace")
    # tqdm progress bars redraw with carriage returns; keep only the latest state of each line.
    cleaned = [line.rsplit("\r", 1)[-1] for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(cleaned[-lines:])
