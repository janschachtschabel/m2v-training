"""Streamlit UI for m2vtrain: data files, run options, live status, run comparison and finished models.

Start from the project folder: `streamlit run ui/app.py`. Runs are separate processes (`python -m m2vtrain run`),
so they keep going when the browser tab or the UI is closed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from m2vtrain.config import RunConfig
from m2vtrain.data import COLUMNS as EXPORT_COLUMNS
from m2vtrain.export import upload_model
from m2vtrain.runs import (
    CONSOLE_LOG,
    describe_state,
    inspect_export,
    save_data_file,
    list_data_files,
    list_keyword_lists,
    list_models,
    list_runs,
    results_frame,
    run_status,
    start_run,
    stop_run,
    tail,
)
from m2vtrain.status import RunState, is_stale, mark_cancelled
from run_form import choose_template, options_form  # ui/ is on the path when Streamlit runs app.py

ROOT = Path(os.environ.get("M2V_HOME") or Path(__file__).resolve().parents[1]).expanduser().resolve()
DATA_DIR = ROOT / "data"
PAGES = ("Daten", "Neuer Lauf", "Läufe", "Vergleich", "Modelle")
MAIN_METRICS = ["keyword_ndcg@10", "keyword_lower_ndcg@10", "keyword_recall@10", "title_mrr@10", "subject_macro_f1"]
FIELD_LABELS = {
    "id": "ID", "title": "Titel", "description": "Beschreibung", "keywords_raw": "Keywords", "language": "Sprache", "subjects": "Fach",
}
DEFAULT_CONFIG = RunConfig.from_dict(
    {"name": "m2v-de-bildung", "synthetic": {"enabled": True}, "keyword_lists": {"files": ["data/keywords.txt"]}}
)


@st.cache_resource
def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


@st.cache_data(show_spinner="Datei wird gelesen …")
def cached_inspect(path: str, modified: float) -> dict:  # `modified` invalidates the cache when the file changes
    return inspect_export(Path(path))


def _megabytes(size: int) -> str:
    return f"{size / 1_000_000:,.1f} MB"


# ---------------------------------------------------------------- Daten


def page_data() -> None:
    st.header("Daten")
    st.caption(f"Exportdateien (`;`-getrennte WLO-CSV, auch `.csv.gz`) liegen in `{DATA_DIR}`.")
    uploaded = st.file_uploader("Exportdatei hinzufügen", type=["csv", "gz"])
    # The uploader keeps its file across reruns; save each upload once.
    if uploaded is not None and st.session_state.get("saved_upload") != uploaded.file_id:
        st.session_state["saved_upload"] = uploaded.file_id
        try:
            saved = save_data_file(uploaded.name, uploaded.getvalue(), DATA_DIR)
        except (ValueError, FileExistsError) as error:
            st.error(str(error))
        else:
            st.success(f"{saved.name} gespeichert.")

    files = list_data_files(DATA_DIR)
    if not files:
        st.info("Noch keine Daten. Datei hochladen oder in den Ordner `data/` kopieren.")
        return
    st.dataframe(
        pd.DataFrame(
            [{"Datei": f.path.name, "Größe": _megabytes(f.size_bytes), "Geändert": f"{f.modified:%d.%m.%Y %H:%M}"} for f in files]
        ),
        hide_index=True,
    )
    chosen = st.selectbox("Datei prüfen", [f.path.name for f in files])
    if st.button("Prüfen"):
        path = DATA_DIR / chosen
        try:
            info = cached_inspect(str(path), path.stat().st_mtime)
        except (OSError, EOFError, ValueError, UnicodeDecodeError) as error:
            st.error(f"{chosen} lässt sich nicht lesen: {error}")
            return
        columns = st.columns(3)
        columns[0].metric("Einträge", f"{info['rows']:,}")
        columns[1].metric("mit Beschreibung", f"{info['with_description']:,}")
        columns[2].metric("mit Keywords", f"{info['with_keywords']:,}")
        used = {column: FIELD_LABELS[field] for column, field in EXPORT_COLUMNS.items()}
        st.write(f"**{len(info['columns'])} Spalten**, davon nutzt das Training {sum(c in used for c in info['columns'])}:")
        st.dataframe(
            pd.DataFrame([{"Spalte": column, "genutzt als": used.get(column, "")} for column in info["columns"]]),
            hide_index=True,
        )
        with st.expander("Sprachen in der Datei"):
            st.caption(
                "Nur zur Übersicht, hier wird nichts gefiltert. Das Training nimmt die Sprachen aus "
                "„Neuer Lauf“ → Erweitert → „Sprachen“ (Standard: de) plus Einträge ohne Angabe, deren Sprache erkannt wird."
            )
            st.dataframe(
                pd.DataFrame(list(info["languages"].items()), columns=["Sprache", "Einträge"]), hide_index=True
            )


# ---------------------------------------------------------------- Neuer Lauf


def page_new_run() -> None:
    st.header("Neuer Lauf")
    data_files = [f"data/{f.path.name}" for f in list_data_files(DATA_DIR)]
    keyword_lists = [f"data/{f.path.name}" for f in list_keyword_lists(DATA_DIR)]
    if not data_files:
        st.caption("Keine Exportdateien in data/: trainiert wird dann nur mit dem synthetischen Datensatz und Keyword-Listen.")
    raw = options_form(choose_template(list_runs(ROOT / "runs"), DEFAULT_CONFIG), data_files, keyword_lists)
    if raw is None:
        return
    try:
        config = RunConfig.from_dict(raw)
    except ValueError as error:
        st.error(f"Ungültige Optionen: {error}")
        return
    try:
        run_dir = start_run(config, ROOT)
    except OSError as error:
        st.error(f"Der Lauf konnte nicht gestartet werden: {error}")
        return
    st.session_state["next_page"] = "Läufe"
    st.session_state["next_run"] = run_dir.name
    st.rerun()


# ---------------------------------------------------------------- Läufe


def _local_time(timestamp: str) -> str:
    if not timestamp:
        return "–"
    try:
        return f"{datetime.fromisoformat(timestamp).astimezone():%d.%m.%Y %H:%M:%S}"
    except ValueError:
        return timestamp


def _show_status(run_dir: Path, was_running: bool) -> None:
    status = run_status(run_dir)
    label = describe_state(status)
    if status is None:
        st.warning("Kein Status vorhanden.")
        return
    alive = status.state is RunState.RUNNING and not is_stale(status)
    if was_running and not alive:
        st.rerun()  # whole page, so the results of the finished run appear and polling stops
    count = len(status.stages)
    stage = f"Schritt {status.stage_index}/{count}: {status.stage}" if count else status.stage
    st.markdown(f"**Zustand:** {label} · {stage}")
    if alive:
        st.progress(status.progress if status.progress is not None else 0.0, text=status.message or stage)
        if st.button("Lauf stoppen", type="secondary"):
            with st.spinner("Wird gestoppt …"):
                stopped = stop_run(run_dir)
            if not stopped:
                st.error("Der Prozess ließ sich nicht stoppen (schon beendet oder keine Berechtigung).")
            else:
                st.rerun()
    elif status.state is RunState.RUNNING:  # stale: the process is gone without a final status
        st.warning("Der Prozess meldet sich nicht mehr. Die Ursache steht meist am Ende der Konsole.")
        if st.button("Als abgebrochen markieren"):
            mark_cancelled(run_dir, "Kein Lebenszeichen mehr")
            st.rerun()
    elif status.message:
        st.caption(status.message)
    if status.error:
        st.error(status.error)
    st.caption(
        f"Gestartet {_local_time(status.started_at)} · letzte Meldung {_local_time(status.heartbeat_at)}"
        f" · beendet {_local_time(status.finished_at)}"
    )
    st.code(tail(run_dir / "run.log", lines=40) or "(noch kein Log)", language=None)
    with st.expander("Konsole (Fortschrittsbalken, Fehlermeldungen)"):
        st.code(tail(run_dir / CONSOLE_LOG, lines=80) or "(leer)", language=None)


def page_runs() -> None:
    st.header("Läufe")
    runs = list_runs(ROOT / "runs")
    if not runs:
        st.info("Noch keine Läufe.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Lauf": run.run_id,
                    "Name": run.config.name if run.config else "",
                    "Zustand": describe_state(run.status),
                    "Schritt": run.status.stage if run.status else "",
                    "Gestartet": _local_time(run.status.started_at) if run.status else "",
                }
                for run in runs
            ]
        ),
        hide_index=True,
    )
    ids = [run.run_id for run in runs]
    if st.session_state.get("selected_run") not in ids:
        st.session_state.pop("selected_run", None)
    if "next_run" in st.session_state:
        st.session_state["selected_run"] = st.session_state.pop("next_run")
    run_id = st.selectbox("Lauf ansehen", ids, key="selected_run")
    run = runs[ids.index(run_id)]
    running = run.status is not None and run.status.state is RunState.RUNNING and not is_stale(run.status)
    st.fragment(run_every=3 if running else None)(_show_status)(run.path, running)
    if run.has_eval:
        st.subheader("Ergebnisse")
        for mode in run.config.eval_modes if run.config else ("raw",):
            st.caption(f"Eingabe: {mode}")
            frame = results_frame([run], mode)
            if frame.empty:
                st.warning("eval.json ist leer oder beschädigt.")
                break
            st.dataframe(frame[["Encoder", *[m for m in MAIN_METRICS if m in frame]]], hide_index=True)


# ---------------------------------------------------------------- Vergleich


def page_compare() -> None:
    st.header("Vergleich")
    runs = list_runs(ROOT / "runs")
    mode = st.radio("Eingabe bei der Evaluation", ["raw", "lemma"], horizontal=True, help="lemma: Anfragen und Dokumente als Substantiv-Grundformen")
    frame = results_frame(runs, mode)
    if frame.empty:
        st.info("Noch keine ausgewerteten Läufe.")
        return
    if not st.checkbox("Baselines (TF-IDF, Vergleichsmodelle) zeigen", True):
        frame = frame[frame["Encoder"] == "Modell"]
    metrics = st.multiselect("Metriken", [c for c in frame.columns if "@" in c or c.endswith("_f1")], [m for m in MAIN_METRICS if m in frame])
    if frame["Testsplit"].nunique() > 1:
        st.warning(
            "Die Läufe haben verschiedene Testsplits (andere Daten, Testanteil oder Seed). "
            "Direkt vergleichbar sind nur Zeilen mit gleichem Testsplit."
        )
    columns = ["Lauf", "Name", "Encoder", "Testsplit", "Lehrer", "Dim", "Eingabe Training", "Tokenlearn", "Paare", *metrics]
    view = frame[[c for c in columns if c in frame]].sort_values(metrics[0] if metrics else "Lauf", ascending=False)
    st.dataframe(view.style.highlight_max(subset=metrics, color="#c8e6c9").format(precision=3, subset=metrics), hide_index=True)


# ---------------------------------------------------------------- Modelle


def _card_body(readme: Path) -> str:
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    return text.split("---", 2)[2] if text.startswith("---") else text


def page_models() -> None:
    st.header("Modelle")
    models = list_models(ROOT / "models")
    if not models:
        st.info("Noch keine fertigen Modelle.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {"Modell": m.name, "Größe": _megabytes(m.size_bytes), "Erstellt": f"{m.modified:%d.%m.%Y %H:%M}", **{k: m.metrics.get(k) for k in MAIN_METRICS}}
                for m in models
            ]
        ),
        hide_index=True,
    )
    model = models[[m.name for m in models].index(st.selectbox("Modell", [m.name for m in models]))]
    st.markdown(f"Ordner: `{model.path}`")
    with st.expander("Model Card"):
        st.markdown(_card_body(model.path / "README.md"))

    st.subheader("Auf Hugging Face hochladen")
    st.caption("Anmeldung vorher einmal mit `hf auth login` oder über die Umgebungsvariable HF_TOKEN.")
    repo_id = st.text_input("Repository", placeholder=f"benutzername/{model.name}")
    public = st.checkbox("öffentlich (sonst privat)")
    if st.button("Hochladen", disabled="/" not in repo_id):
        with st.spinner("Wird hochgeladen …"):
            try:
                url = upload_model(model.path, repo_id.strip(), private=not public)
            except Exception as error:  # shown to the user: login, network or permission problems
                st.error(f"Upload fehlgeschlagen: {error}")
            else:
                st.success(f"Hochgeladen: {url}")
    command = ["python", "-m", "m2vtrain", "upload", str(model.path), repo_id or f"benutzername/{model.name}"]
    st.code(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command), language="bash")


# ---------------------------------------------------------------- Rahmen


def main() -> None:
    st.set_page_config(page_title="M2V-Training", layout="wide")
    if st.get_option("server.address") not in ("localhost", "127.0.0.1"):
        st.warning(
            "Die Oberfläche ist im Netzwerk erreichbar. Bitte im Projektordner starten "
            "(dann gilt .streamlit/config.toml) oder mit --server.address localhost."
        )
    if "next_page" in st.session_state:
        st.session_state["page"] = st.session_state.pop("next_page")
    with st.sidebar:
        st.title("M2V-Training")
        page = st.radio("Bereich", PAGES, key="page")
        gpu = gpu_name()
        st.caption(f"GPU: {gpu}" if gpu else "Keine GPU gefunden – Training läuft auf der CPU (langsam).")
        st.caption(f"Projektordner: {ROOT}")
    {
        "Daten": page_data,
        "Neuer Lauf": page_new_run,
        "Läufe": page_runs,
        "Vergleich": page_compare,
        "Modelle": page_models,
    }[page]()


main()
