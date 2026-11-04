# Konfiguration

*English: [../en/configuration.md](../en/configuration.md)*

Eine Laufkonfiguration ist eine JSON-Datei. Nötig sind nur `name`, eine Datenquelle (`synthetic.enabled: true`
und/oder `data_files`) und die Werte, die vom Standard abweichen; `python -m m2vtrain init-config` schreibt eine
vollständige Datei. Jeder Lauf speichert seine komplette Konfiguration in `runs/<lauf>/config.json`;
`python -m m2vtrain run runs/<lauf>/config.json` wiederholt ihn. Unbekannte Schlüssel und falsche Typen werden mit
einer Fehlermeldung abgelehnt.

## Daten

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `name` | – | Name von Lauf- und Modellordner, Vorschlag für den Hub-Namen |
| `data_files` | `[]` | lokale Exporte (`;`-getrenntes CSV, `.csv.gz`); Platzhalter erlaubt; mit synthetischem Datensatz optional |
| `synthetic.enabled` | `false` | synthetischen Datensatz verwenden |
| `synthetic.dataset` | `JanSchachtschabel/synthetic-german-educational-materials-50k` | Hub-Name, lokaler Ordner oder Parquet-Datei |
| `synthetic.revision` | `null` | feste Hub-Revision (Commit oder Tag) für Reproduzierbarkeit |
| `synthetic.splits` | alle drei | welche Teile des Datensatzes gelesen werden (getrennt wird von `m2vtrain`) |
| `synthetic.use_fulltext`, `fulltext_chunk_chars` | `true`, 1200 | Volltexte als Tokenlearn-Texte und fürs Vokabular, in Abschnitten aus ganzen Absätzen |
| `synthetic.use_questions`, `max_questions_per_doc` | `true`, 3 | Fragen als Tokenlearn-Texte und Paare Frage → Dokument |
| `synthetic.use_answers` | `false` | Antworten als zusätzliche Tokenlearn-Texte |
| `synthetic.max_documents` | `null` | Stichprobe (Schnelltests) |
| `keyword_lists.files` | `[]` | Keyword-Listen, ein Begriff pro Zeile |
| `keyword_lists.min_count`, `max_keywords` | 1, `null` | nach der Anzahl-Spalte filtern, Zahl der Begriffe begrenzen |
| `keyword_lists.add_to_vocabulary`, `tokenlearn`, `lowercase_inputs`, `lemma_inputs`, `pairs` | alle `true` | wie die Begriffe genutzt werden, siehe [daten-und-lizenzen.md](daten-und-lizenzen.md) |
| `keyword_lists.max_docs_per_keyword` | 3 | Dokumente pro Begriff für Paare Begriff → Dokument |
| `data.languages` | `["de"]` | Sprachen, die aus lokalen Exporten übernommen werden |
| `data.*` | | weitere Bereinigung lokaler Exporte (Spracherkennung, Beschreibungslänge, Boilerplate, Stoppliste) |

## Aufteilung und Evaluation

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `test_fraction` | 0.05 | zurückgehaltener Anteil beim einfachen Testteil |
| `cross_validation.folds` | 0 | 0: einfacher Testteil; 2–20: k-Fold |
| `cross_validation.final_model` | `true` | nach den Folds ein Modell auf allen Dokumenten trainieren und speichern |
| `cross_validation.shared_vocabulary` | `false` | ein Vokabular und eine Destillation für alle Folds (schneller, leicht optimistisch) |
| `seed` | 42 | Seed für Aufteilung, Stichproben und Training |
| `eval.*` | | Anzahl Anfragen, Optionen der Fach-Klassifikation |
| `eval_modes` | `["raw", "lemma"]` | Eingabearten der Evaluation |
| `baseline_models` | `[]` | Model2Vec- oder Sentence-Transformers-Modelle zum Vergleich |
| `evaluate_teacher` | `false` | auch den Lehrer bewerten (langsam) |

## Lehrer

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `teacher.model_name` | `Alibaba-NLP/gte-multilingual-base` | Hub-Name oder lokaler Pfad |
| `teacher.trust_remote_code` | `true` bei gte | nur für Modelle mit eigenem Code |
| `teacher.pooling` | `mean` | Pooling bei der Destillation: `mean`, `last` (Decoder wie Qwen3), `first` (CLS), `pooler` |
| `teacher.document_prefix` | `""` | Text vor jedem Dokument, z. B. `"passage: "` bei e5 |
| `teacher.batch_size` | 64 | Texte pro Lehrer-Schritt |
| `teacher.max_seq_length` | `null` | Lehrer-Eingabe in Tokens begrenzen (Speicher wächst quadratisch mit der Länge) |
| `teacher.dtype` | `auto` | `float16`/`bfloat16` halbieren auf der GPU Speicher und Zeit; die CPU nutzt immer float32 |
| `teacher.device` | `null` (automatisch) | `cuda`, `cpu`, `mps` |

`max_seq_length` oder `dtype` zu ändern, ändert den Cache-Schlüssel der Lehrer-Embeddings (die Standardwerte
lassen ältere Caches gültig).

## Modell und Training

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `pca_dims` | 256 | Dimensionen des statischen Modells |
| `vocab.max_vocab_size` | 200000 | Obergrenze für zusätzliches Vokabular (Keywords bleiben immer) |
| `vocab.min_word_count`, `keyword_min_docs` | 5, 3 | Mindesthäufigkeit von Korpuswörtern und Dokument-Keywords |
| `vocab.add_lemmas`, `add_lowercase_variants`, `include_keywords` | `true` | welche Einträge dazukommen |
| `input_mode`, `lemma_share` | `mixed`, 0.5 | Eingabe des statischen Modells beim Training: `raw`, `lemma` oder gemischt |
| `tokenlearn.enabled`, `max_texts` | `true`, `null` | Tokenlearn auf (allen) Texten |
| `tokenlearn.train.*` | Lernrate 1e-3, Early Stopping 5 | `max_epochs`, `learning_rate`, `batch_size`, `device` |
| `pairs.enabled`, `max_pairs` | `false`, 500000 | kontrastives Paar-Training und seine Obergrenze (`null`: alle Paare) |
| `pairs.train.*` | Lernrate 1e-4, Early Stopping 5 | wie oben |
| `quantize_to` | `float16` | Speicherformat: `float32`, `float16`, `int8` |
| `vocabulary_quantization` | `null` | Vokabular auf so viele Vektoren clustern (kleineres Modell) |

## Model Card und Pfade

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `license` | `apache-2.0` | Lizenz in der Card; zum Lehrer passend wählen |
| `card.data_description` | `""` | ersetzt den automatischen Text über die Trainingsdaten |
| `card.show_data_facts` | `false` | Dokument- und Vokabularzahlen zeigen (nur bei öffentlichen Daten) |
| `runs_dir`, `models_dir`, `cache_dir` | `runs`, `models`, `cache` | Ausgabeordner |

## Fertige Konfigurationen

| Datei | Zweck |
|---|---|
| `configs/smoke-test-cpu.json` | kleiner 3-Fold-Lauf auf der CPU, prüft die ganze Pipeline |
| `configs/rtx5060ti-01-gte-reference.json` | bisheriger Lehrer auf der neuen Datenbasis, 3-Fold |
| `configs/rtx5060ti-02-bge-m3.json` | bge-m3, 3-Fold |
| `configs/rtx5060ti-03-arctic-embed-l-v2.json` | arctic-embed-l-v2.0, 3-Fold |
| `configs/rtx5060ti-04-qwen3-embedding-0.6b.json` | Qwen3-Embedding-0.6B, 3-Fold |
| `configs/rtx5060ti-05-qwen3-embedding-4b.json` | Qwen3-Embedding-4B, 3-Fold |
| `configs/rtx5060ti-final-5fold.json` | 5-Fold-Schätzung plus finales Modell auf allen Daten (Qwen3-Embedding-4B; durch den Gewinner ersetzen) |

Alle erwarten `data/keywords.txt`; ohne `keyword_lists` wird nur mit dem synthetischen Datensatz trainiert.
