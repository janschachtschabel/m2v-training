"""Smoke tests of the Streamlit UI with streamlit's AppTest (renders the script without a browser)."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from m2vtrain.config import RunConfig, save_config
from m2vtrain.status import StatusWriter

from tests.conftest import make_row, write_wlo_export

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")
PAGES = ["Daten", "Neuer Lauf", "Läufe", "Vergleich", "Modelle"]


def _app(home, monkeypatch) -> AppTest:
    monkeypatch.setenv("M2V_HOME", str(home))
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    return app


def _open(app: AppTest, page: str) -> AppTest:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
    return app


@pytest.fixture
def home(tmp_path):
    (tmp_path / "data").mkdir()
    write_wlo_export(tmp_path / "data" / "export.csv.gz", [make_row("Bruch", "Text", "Bruch", "de")])
    run_dir = tmp_path / "runs" / "r1"
    save_config(RunConfig(name="eins", data_files=["data/export.csv.gz"]), run_dir / "config.json")
    StatusWriter(run_dir, stages=["Daten"]).finish()
    metrics = {"keyword_ndcg@10": 0.5, "title_mrr@10": 0.4, "subject_macro_f1": 0.3}
    results = {"model": {"raw": metrics, "lemma": metrics}, "baselines": {"tfidf": {"raw": metrics, "lemma": metrics}}}
    (run_dir / "eval.json").write_text(json.dumps(results), encoding="utf-8")
    model_dir = tmp_path / "models" / "r1"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"x")
    (model_dir / "README.md").write_text("---\nlicense: apache-2.0\n---\n# eins\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_any_data(tmp_path, monkeypatch, page):
    _open(_app(tmp_path, monkeypatch), page)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_runs_and_models(home, monkeypatch, page):
    app = _open(_app(home, monkeypatch), page)
    if page in ("Vergleich", "Modelle"):
        assert app.dataframe


def test_starting_a_run_switches_to_the_runs_page(home, monkeypatch):
    import m2vtrain.runs as runs

    started = []

    def fake_start(config, root):
        started.append(config)
        run_dir = home / "runs" / "neu"
        save_config(config, run_dir / "config.json")
        return run_dir

    monkeypatch.setattr(runs, "start_run", fake_start)
    app = _open(_app(home, monkeypatch), "Neuer Lauf")
    app.text_input[0].set_value("mein-lauf")
    app.button[0].click().run()

    assert not app.exception
    assert [config.name for config in started] == ["mein-lauf"]
    # the default data basis is the openly licensed one; local exports are opt-in
    assert started[0].synthetic.enabled and started[0].data_files == ()
    assert app.session_state["page"] == "Läufe"


def test_new_runs_use_the_workspace_folders_even_from_a_foreign_template(home, monkeypatch):
    import m2vtrain.runs as runs

    foreign = home / "runs" / "fremd"
    save_config(
        RunConfig(name="fremd", data_files=["data/export.csv.gz"], runs_dir="experimente", models_dir="C:/x", test_fraction=0.6),
        foreign / "config.json",
    )
    started = []
    monkeypatch.setattr(runs, "start_run", lambda config, root: started.append(config) or foreign)
    app = _open(_app(home, monkeypatch), "Neuer Lauf")
    app.selectbox[0].set_value("wie Lauf fremd").run()
    assert not app.exception, app.exception
    app.button[0].click().run()

    assert not app.exception
    assert (started[0].runs_dir, started[0].models_dir, started[0].cache_dir) == ("runs", "models", "cache")
    assert started[0].test_fraction == 0.6


def test_pages_survive_corrupt_run_files(home, monkeypatch):
    (home / "runs" / "r1" / "eval.json").write_text("{", encoding="utf-8")
    (home / "models" / "r1" / "eval_results.json").write_text("{", encoding="utf-8")
    app = _app(home, monkeypatch)
    for page in PAGES:
        _open(app, page)


def test_checking_an_export_lists_its_columns_and_marks_the_used_ones(home, monkeypatch):
    app = _open(_app(home, monkeypatch), "Daten")
    next(button for button in app.button if button.label == "Prüfen").click().run()
    assert not app.exception, app.exception
    columns = app.dataframe[1].value
    assert "properties.cclom:title" in list(columns["Spalte"])
    used = dict(zip(columns["Spalte"], columns["genutzt als"]))
    assert used["properties.cclom:title"] == "Titel"
