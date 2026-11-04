"""Options form of the "Neuer Lauf" page: turns widget values into a run config dict."""

from __future__ import annotations

import streamlit as st

from m2vtrain.config import RunConfig
from m2vtrain.lemma import InputMode
from m2vtrain.runs import RunInfo

DEVICES = ["auto", "cuda", "cpu", "mps"]


def choose_template(runs: list[RunInfo], default: RunConfig) -> RunConfig:
    """Select box: the default options or those of an earlier run."""
    labels = {"Standardwerte": default, **{f"wie Lauf {run.run_id}": run.config for run in runs if run.config}}
    choice = st.selectbox("Vorlage", list(labels), help="Optionen eines früheren Laufs übernehmen")
    return labels[choice]


def options_form(base: RunConfig, data_files: list[str], keyword_lists: list[str] | None = None) -> dict | None:
    """Render the options form; returns the raw config dict when submitted."""
    keyword_lists = keyword_lists or []
    raw = base.to_dict()
    # Runs started here always use the workspace folders, whatever the template run used.
    raw.update(runs_dir="runs", models_dir="models", cache_dir="cache")
    missing = [f for f in base.data_files if f not in data_files and not any(c in f for c in "*?[")]
    if missing:
        st.warning(f"Datendateien der Vorlage fehlen in data/: {', '.join(missing)}")
    with st.form("run_options"):
        raw["name"] = st.text_input("Name des Laufs / Modells", base.name, help="Ordnername unter runs/ und models/, Vorschlag für den Hugging-Face-Namen")
        st.subheader("Datenbasis")
        synthetic = raw["synthetic"]
        synthetic["enabled"] = st.checkbox(
            "Synthetischer Datensatz (CC BY 4.0)",
            base.synthetic.enabled,
            help="JanSchachtschabel/synthetic-german-educational-materials-50k: Titel, Beschreibungen, Keywords, Volltexte "
            "und Fragen; frei lizenziert und dokumentierbar",
        )
        synthetic["dataset"] = st.text_input("Datensatz (Hub-Name oder lokaler Ordner)", base.synthetic.dataset)
        selected_lists = [f for f in base.keyword_lists.files if f in keyword_lists]
        raw["keyword_lists"]["files"] = st.multiselect(
            "Keyword-Listen aus data/ (*.txt)",
            keyword_lists,
            default=selected_lists,
            help="Nur Begriffe, ein Keyword pro Zeile, z. B. mit `python -m m2vtrain extract-keywords` erzeugt",
        )
        selected = [f for f in base.data_files if f in data_files]
        if not selected and not base.synthetic.enabled:
            selected = data_files
        raw["data_files"] = st.multiselect(
            "Lokale Exportdateien (optional)",
            data_files,
            default=selected,
            help="Exporte aus data/; gewählte Dateien werden zusammen gelesen, Dubletten entfernt. Für ein dokumentierbares "
            "Modell leer lassen und nur den synthetischen Datensatz und Keyword-Listen nutzen",
        )
        left, right = st.columns(2)
        raw["cross_validation"]["folds"] = left.number_input(
            "k-Fold (0 = einfacher Testteil)",
            0,
            20,
            base.cross_validation.folds,
            help="k Modelle, jedes auf dem Teil getestet, den es nicht gesehen hat; Mittelwert ± Streuung. Kostet k-mal "
            "Training, die Lehrer-Embeddings werden nur einmal berechnet",
        )
        raw["cross_validation"]["final_model"] = right.checkbox(
            "Danach finales Modell auf allen Daten", base.cross_validation.final_model
        )
        raw["cross_validation"]["shared_vocabulary"] = right.checkbox(
            "Ein Vokabular für alle Folds (schneller)",
            base.cross_validation.shared_vocabulary,
            help="Nur einmal destillieren statt pro Fold; leicht optimistisch, weil Wörter der Testdokumente als Token existieren",
        )

        st.subheader("Lehrermodell")
        teacher = raw["teacher"]
        teacher["model_name"] = st.text_input(
            "Hugging-Face-Modell oder lokaler Pfad",
            base.teacher.model_name,
            help="z. B. Alibaba-NLP/gte-multilingual-base, BAAI/bge-m3, Qwen/Qwen3-Embedding-0.6B",
        )
        left, right = st.columns(2)
        teacher["trust_remote_code"] = left.checkbox("trust_remote_code", base.teacher.trust_remote_code, help="nötig für gte-multilingual")
        teacher["document_prefix"] = right.text_input("Dokument-Präfix", base.teacher.document_prefix, help='z. B. "passage: " für e5-Modelle')
        devices = list(dict.fromkeys([*DEVICES, base.teacher.device or "auto"]))
        device = left.selectbox(
            "Gerät", devices, index=devices.index(base.teacher.device or "auto"), help="auto nimmt die GPU, wenn vorhanden, sonst die CPU"
        )
        teacher["device"] = device
        teacher["batch_size"] = right.number_input(
            "Batchgröße Lehrer", 1, 1024, base.teacher.batch_size, help="Texte pro Schritt des Lehrers; größer = etwas schneller, mehr RAM. CPU: 32–64"
        )
        poolings = ["mean", "last", "first", "pooler"]
        teacher["pooling"] = left.selectbox(
            "Pooling (Destillation)",
            poolings,
            index=poolings.index(base.teacher.pooling),
            help="mean für gte/bge/e5, last für Decoder-Modelle wie Qwen3-Embedding, first für CLS-Modelle",
        )
        teacher["max_seq_length"] = right.number_input(
            "max. Tokens Lehrer (0 = Modellgrenze)", 0, 32768, base.teacher.max_seq_length or 0, step=128
        ) or None

        st.subheader("Modell und Training")
        left, right = st.columns(2)
        dims = sorted({64, 128, 256, 384, 512, 768, base.pca_dims})
        raw["pca_dims"] = left.select_slider(
            "Dimensionen", dims, base.pca_dims, help="Länge der Vektoren; mehr = etwas genauer, aber größer und langsamer"
        )
        modes = [mode.value for mode in InputMode]
        raw["input_mode"] = right.selectbox(
            "Eingabe beim Training",
            modes,
            index=modes.index(base.input_mode),
            help="raw: Text wie er ist · lemma: nur Substantive in Grundform · mixed: zufällige Mischung, das Modell funktioniert danach mit beiden Eingaben (empfohlen)",
        )
        raw["lemma_share"] = right.slider("Anteil Grundform bei mixed", 0.0, 1.0, base.lemma_share, help="Wie oft bei mixed die Grundform-Variante gezeigt wird")
        raw["tokenlearn"]["enabled"] = left.checkbox(
            "Tokenlearn (an Lehrer-Embeddings trainieren)",
            base.tokenlearn.enabled,
            help="Der eigentliche Lernschritt: passt das Modell an die Lehrer-Embeddings der Dokumente an",
        )
        max_texts = left.number_input(
            "max. Trainingstexte (0 = alle)",
            0,
            None,
            base.tokenlearn.max_texts or 0,
            step=10_000,
            help="Bestimmt die Laufzeit am stärksten. Ohne GPU zum Testen 20.000; alle Trainingsdokumente dauern viele Stunden",
        )
        raw["tokenlearn"]["max_texts"] = max_texts or None
        raw["pairs"]["enabled"] = left.checkbox(
            "Paar-Training (Keyword→Dokument, Titel→Beschreibung)",
            base.pairs.enabled,
            help="Kontrastive Feinabstimmung nach Tokenlearn für bessere Keyword-Suche; kostet zusätzlich Zeit",
        )
        formats = ["float16", "float32", "int8"]
        raw["quantize_to"] = right.selectbox(
            "Speichern als",
            formats,
            index=formats.index(base.quantize_to),
            help="float16: empfohlen für Hugging Face, halb so groß, praktisch verlustfrei · "
            "int8: kleinste Datei, etwas Verlust · float32: zum Weitertrainieren",
        )

        with st.expander("Erweitert"):
            _advanced_options(raw, base, device)
        submitted = st.form_submit_button("Training starten", type="primary")
    return raw if submitted else None


def _advanced_options(raw: dict, base: RunConfig, device: str) -> None:
    left, right = st.columns(2)
    vocab = raw["vocab"]
    vocab["max_vocab_size"] = left.number_input("max. zusätzliches Vokabular", 0, 2_000_000, base.vocab.max_vocab_size, step=10_000)
    vocab["min_word_count"] = left.number_input("Wort min. Häufigkeit", 1, 1000, base.vocab.min_word_count)
    vocab["keyword_min_docs"] = left.number_input("Keyword min. Dokumente", 1, 1000, base.vocab.keyword_min_docs)
    vocab["add_lemmas"] = left.checkbox("Grundformen ins Vokabular", base.vocab.add_lemmas)
    vocab["add_lowercase_variants"] = left.checkbox("Kleinschreibvarianten ins Vokabular", base.vocab.add_lowercase_variants)
    vocab["include_keywords"] = left.checkbox("Keywords ins Vokabular", base.vocab.include_keywords)

    for section, label in (("tokenlearn", "Tokenlearn"), ("pairs", "Paar-Training")):
        train, defaults = raw[section]["train"], getattr(base, section).train
        train["learning_rate"] = right.number_input(f"Lernrate {label}", 1e-6, 1.0, defaults.learning_rate, format="%.0e")
        train["max_epochs"] = right.number_input(f"max. Epochen {label} (-1 = bis Early Stopping)", -1, 1000, defaults.max_epochs)
        train["device"] = device
    raw["pairs"]["max_pairs"] = right.number_input("max. Paare (0 = alle)", 0, None, base.pairs.max_pairs or 0, step=50_000) or None

    languages = left.text_input(
        "Sprachen",
        ", ".join(base.data.languages),
        help="Kommagetrennt. Einträge ohne Sprachangabe werden erkannt und mitgenommen, wenn sie passen",
    )
    raw["data"]["languages"] = [part.strip() for part in languages.split(",") if part.strip()]
    raw["test_fraction"] = left.number_input(
        "Testanteil", 0.01, 0.99, base.test_fraction, step=0.01, help="Nur für die Evaluation zurückgehaltene Dokumente"
    )
    raw["seed"] = left.number_input("Seed", 0, None, base.seed, help="Gleicher Seed, Daten und Testanteil = gleicher Testteil, nur dann sind Läufe vergleichbar")
    raw["eval"]["max_queries"] = right.number_input("max. Keyword-Anfragen in der Evaluation", 10, 100_000, base.eval.max_queries)
    raw["baseline_models"] = [
        line.strip()
        for line in right.text_area(
            "Vergleichsmodelle (eins pro Zeile)",
            "\n".join(base.baseline_models),
            help="Model2Vec-Modelle (z. B. JanSchachtschabel/m2v-gte-256-edu, minishlab/potion-multilingual-128M) oder "
            "Transformer (z. B. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, intfloat/multilingual-e5-small). "
            "Transformer brauchen ohne GPU je etwa 20 bis 40 Minuten.",
        ).splitlines()
        if line.strip()
    ]
    raw["evaluate_teacher"] = right.checkbox("Lehrer mitbewerten (langsam ohne GPU)", base.evaluate_teacher)
    raw["license"] = right.text_input("Lizenz (Model Card)", base.license)
