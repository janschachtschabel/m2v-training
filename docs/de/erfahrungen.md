# Erfahrungen aus bisherigen Läufen

*English: [../en/lessons-learned.md](../en/lessons-learned.md)*

Erkenntnisse aus den Läufen, die
[`JanSchachtschabel/m2v-gte-256-edu-v2`](https://huggingface.co/JanSchachtschabel/m2v-gte-256-edu-v2) ergeben
haben (Lehrer gte-multilingual-base, 256 Dimensionen, auf einem Laptop ohne GPU trainiert). Die Zahlen stammen aus
der öffentlichen Model Card und beziehen sich auf deren eigenen Testteil; sie sind **nicht vergleichbar** mit
Ergebnissen auf dem synthetischen Datensatz.

## Was am meisten gebracht hat

- **Tokenlearn plus Paar-Training** war der große Hebel. Gegenüber dem Vorgänger (nur Destillation, gleicher
  Lehrer, gleiche Dimensionen) stieg die Keyword-Suche (nDCG@10) von 0,084 auf 0,323, Titel → Beschreibung
  (MRR@10) von 0,344 auf 0,648 und die Fach-Klassifikation (Makro-F1) von 0,372 auf 0,500. Das Modell schlug auch
  TF-IDF (0,220 / 0,577 / 0,435), potion-multilingual-128M und den Transformer
  paraphrase-multilingual-MiniLM-L12-v2.
- **Grundformen für kleingeschriebene Anfragen.** Mit kleingeschriebenen Keywords lag nDCG@10 bei Rohtext bei
  0,212, mit auf Substantiv-Grundformen reduzierten Anfragen und Dokumenten bei 0,291. Für ordentlich geschriebenen
  Text bleibt Rohtext besser. Das Vokabular enthält deshalb gebeugte Formen, Grundformen und
  Kleinschreibvarianten, und das Training mischt Rohtext und Grundformen (`input_mode: mixed`), damit das Modell
  beides kann.
- **Groß-/Kleinschreibung zählt.** Tokenizer im Stil von XLM-R unterscheiden Groß- und Kleinschreibung:
  „photosynthese“ und „Photosynthese“ sind verschiedene Tokens. Kleinschreibvarianten im Vokabular und das neue
  `keyword_lists.lowercase_inputs` setzen hier an.

## Trainingsverlauf

- Tokenlearn stoppte nach etwa 23 Epochen (Early Stopping); das Paar-Training verbesserte sich langsam weiter, war
  aber nach etwa 35 Epochen flach. Die GPU-Konfigurationen begrenzen es deshalb auf 35 (`pairs.train.max_epochs`).
- Die Lehrer-Embeddings sind der teure Schritt. Auf der Laptop-CPU schaffte der gte-Lehrer etwa 7–10 Texte pro
  Sekunde; ein voller Lauf dauerte fast einen Tag. Eine GPU ändert das um ein bis zwei Größenordnungen.
- Einzelne sehr lange Eingabetexte ließen dem Lehrer bei
  Batchgröße 64 den Speicher ausgehen. `teacher.max_seq_length` begrenzt die Eingabe jetzt; die
  GPU-Konfigurationen setzen 512 Tokens.

## Technische Fallstricke (im Code bereits gelöst)

- `gte-multilingual-base` unter transformers 5: Sein eigener Code berechnet Positions-IDs und Rotary-Puffer in
  `__init__`; transformers 5 lässt sie uninitialisiert (IndexError oder still falsche Embeddings).
  `m2vtrain.targets.restore_computed_buffers` behebt das.
- `StaticModel.push_to_hub` ersetzt die Model Card durch eine allgemeine. Stattdessen `python -m m2vtrain upload`.
- Läufe aus einem IDE-Terminal oder einer Remote-Sitzung endeten, als dieser Elternprozess neu startete. Lange
  Läufe abgekoppelt starten (`scripts/run_detached.ps1`, `nohup`).
- Ein unterbrochener Lauf setzt die Lehrer-Embeddings aus den gespeicherten Blöcken fort.
- Ergebnisse sind nur bei gleichem Split-Fingerabdruck vergleichbar (`test_split.fingerprint` in `eval.json`).

## Noch nicht ausprobierte Ideen

- ein größerer Lehrer (Konfigurationen liegen bereit: `configs/rtx5060ti-0*.json`, `configs/dual-gpu/`)
- 512 Dimensionen für den besten Lehrer (`configs/dual-gpu/final-5fold-qwen3-4b-512.json`)
- BM25 plus statische Embeddings als hybride Suche
- schwere Negativbeispiele (hard negatives) im Paar-Training
- eine kleine Menge echter Suchanfragen mit bewerteten Treffern als zusätzlicher Testteil
- mehr frei lizenzierter Text für Tokenlearn (z. B. deutsche Wikipedia) und Paare (z. B. GND-Schlagwörter)
