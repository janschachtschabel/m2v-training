# m2v-training

Schnelle, statische [Model2Vec](https://github.com/MinishLab/model2vec)-Embedding-Modelle für **deutsche
Bildungsinhalte** trainieren, die Substantive und Keywords gut erkennen, und sie sauber evaluieren.

*English version: [README.md](README.md)*

Ein Model2Vec-Modell braucht beim Einsatz weder GPU noch Transformer: Ein Text wird eingebettet, indem die
Vektoren seiner Tokens nachgeschlagen und gemittelt werden. Dieses Repository enthält das komplette
Trainingssetup: Datenaufbereitung, Vokabular, Destillation aus einem Transformer-Lehrermodell, Feinabstimmung im
Stil von [Tokenlearn](https://github.com/MinishLab/tokenlearn), kontrastives Paar-Training, eine
Fach-Evaluation (auch als k-Fold-Kreuzvalidierung), Model Cards, Upload zu Hugging Face und eine kleine
Streamlit-Oberfläche.

## Was ein Lauf macht

1. **Daten laden**: den frei lizenzierten synthetischen Datensatz, reine Keyword-Listen und optional lokale
   Metadaten-Exporte. Siehe [Datenbasis](#datenbasis).
2. **Aufteilen** ohne Datenleck: Beinahe-Dubletten und Datensätze zum selben Thema bleiben immer auf einer Seite.
   Entweder ein zurückgehaltener Testteil oder **k Folds**.
3. **Vokabular**: Wörter des Korpus, Grundformen der Substantive (Lemmata), Kleinschreibvarianten und Keywords
   kommen zum Tokenizer des Lehrers dazu. So funktioniert das Modell mit Rohtext und mit Grundform-Eingabe.
4. **Destillieren** eines statischen Modells aus dem Lehrer (Standard: `Alibaba-NLP/gte-multilingual-base`).
5. **Tokenlearn**: Die Token-Vektoren werden so trainiert, dass gemittelte Texte die Satz-Embeddings des Lehrers
   nachbilden (Dokumente, Volltext-Abschnitte, Fragen, Keyword-Begriffe samt Klein- und Grundform-Schreibweise).
6. **Paar-Training** (InfoNCE): Keyword → Dokument, Titel → Beschreibung, Frage → Dokument.
7. **Evaluieren** und **speichern** mit Model Card (Keyword-Suche nDCG@10, Titel → Beschreibung MRR@10,
   Frage → Dokument MRR@10, Fach-Klassifikation Makro-F1; jeweils mit Rohtext und mit Grundformen; TF-IDF und
   andere Modelle zum Vergleich).

## Schnellstart

```bash
git clone https://github.com/janschachtschabel/m2v-training
cd m2v-training
./scripts/setup_linux.sh            # Windows: .\scripts\setup_windows.ps1  (installiert PyTorch für CUDA 12.8+)
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1

# Keyword-Liste (nur Begriffe) aus einem lokalen Metadaten-Export, optional, aber empfohlen
python -m m2vtrain extract-keywords pfad/zum/export.csv.gz data/keywords.txt --min-docs 3

# wenige Minuten auf der CPU: prüft, ob alles von Anfang bis Ende funktioniert
python -m m2vtrain run configs/smoke-test-cpu.json

# Lehrervergleich auf der GPU, danach der finale 5-Fold-Lauf mit dem Gewinner
python -m m2vtrain run configs/rtx5060ti-02-bge-m3.json
python -m m2vtrain compare
```

Die Oberfläche macht dasselbe mit Formularen: `streamlit run ui/app.py` (im Repository-Ordner starten; sie ist
nur auf dem eigenen Rechner erreichbar).

## Datenbasis

Ziel ist eine Trainingsgrundlage, die sich **dokumentieren und veröffentlichen lässt, ohne Rechte Dritter zu
verletzen**:

| Quelle | Was genutzt wird | Lizenz / Status |
|---|---|---|
| [JanSchachtschabel/synthetic-german-educational-materials-50k](https://huggingface.co/datasets/JanSchachtschabel/synthetic-german-educational-materials-50k) | Titel, Beschreibungen, Keywords, Volltexte, Fragen, Fächer, Themen | CC BY 4.0, von einem Sprachmodell geschriebene Texte |
| Keyword-Liste (`data/keywords.txt`) | nur Begriffe, einer pro Zeile; keine Titel, Beschreibungen, IDs oder Links | lokal mit `extract-keywords` erzeugt; wird nie eingecheckt |
| Lokale Exporte (`data_files`) | optional, für private Experimente | werden nie eingecheckt und in Model Cards nie beschrieben |

Modelle, die nur mit den ersten beiden Quellen trainiert sind, lassen sich vollständig beschreiben. Details,
Begründung und was vor einer Veröffentlichung zu prüfen ist: [docs/de/daten-und-lizenzen.md](docs/de/daten-und-lizenzen.md).

## Lehrermodelle für eine RTX 5060 Ti (16 GB)

| Lehrer | Lizenz | Größe | Deutsche Suche (MIRACL-de nDCG@10) | Konfiguration |
|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-4B` | Apache 2.0 | 4,0 Mrd., ~8 GB in bf16 | 62,98 | `rtx5060ti-05-qwen3-embedding-4b.json` |
| `BAAI/bge-m3` | MIT | 568 Mio. | 57,59 | `rtx5060ti-02-bge-m3.json` |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | Apache 2.0 | 568 Mio. | – | `rtx5060ti-03-arctic-embed-l-v2.json` |
| `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0 | 0,6 Mrd. | 54,21 | `rtx5060ti-04-qwen3-embedding-0.6b.json` |
| `Alibaba-NLP/gte-multilingual-base` (bisher) | Apache 2.0 | 305 Mio. | – | `rtx5060ti-01-gte-reference.json` |

Empfehlung: gte (Referenz), bge-m3 und Qwen3-Embedding-4B per 3-Fold-Kreuzvalidierung vergleichen und danach
`rtx5060ti-final-5fold.json` mit dem Gewinner laufen lassen. Begründung, Quellen und was nicht passt oder nicht
frei lizenziert ist: [docs/de/lehrermodelle.md](docs/de/lehrermodelle.md). GPU-Einrichtung und Laufzeiten:
[docs/de/gpu-einrichtung.md](docs/de/gpu-einrichtung.md).

## Evaluation und k-Fold-Kreuzvalidierung

Mit `"cross_validation": {"folds": 5}` werden die Dokumente in 5 Folds geteilt (gruppiert nach Thema und
Beinahe-Dubletten). Fünf Modelle werden trainiert, jedes wird auf dem Fold getestet, den es nie gesehen hat. Die
Kennzahlen werden gemittelt und als Mittelwert ± Standardabweichung angegeben. Mit `"final_model": true` wird
danach ein letztes Modell auf **allen** Dokumenten trainiert und mit den kreuzvalidierten Werten veröffentlicht.
Die Lehrer-Embeddings werden nur einmal für alle Folds berechnet. Details: [docs/de/evaluierung.md](docs/de/evaluierung.md).

## Kommandozeile

```bash
python -m m2vtrain init-config meine.json --synthetic --keywords data/keywords.txt --folds 5
python -m m2vtrain run meine.json                   # Laufordner: runs/<Zeitstempel>-<Name>
python -m m2vtrain run runs/<lauf>/config.json      # einen Lauf mit gleichen Optionen wiederholen
python -m m2vtrain compare [runs/a runs/b] [--mode lemma]
python -m m2vtrain extract-keywords export.csv.gz data/keywords.txt [--min-docs 3] [--with-counts]
python -m m2vtrain upload models/<lauf> <user>/<modell> [--public]
```

Eine Konfiguration braucht nur `name`, eine Datenquelle und die Werte, die vom Standard abweichen; unbekannte
Schlüssel und falsche Typen werden abgelehnt. Alle Optionen: [docs/de/konfiguration.md](docs/de/konfiguration.md).

## Aufbau des Repositorys

```
m2vtrain/          Python-Paket (data, sources, split, vocab, targets, train, evaluate, pipeline, export, cli)
ui/                Streamlit-Oberfläche
configs/           fertige Laufkonfigurationen (CPU-Schnelltest, Lehrervergleich RTX 5060 Ti, finaler Lauf)
scripts/           Einrichtung (Windows/Linux), GPU-Check, abgekoppelte Läufe, Warteschlangen
docs/en, docs/de   Dokumentation auf Englisch und Deutsch
tests/             Offline-Tests mit einem winzigen lokalen Lehrermodell (keine Downloads)
```

Zur Laufzeit entstehen die Ordner `data/`, `runs/`, `models/` und `cache/` im Repository-Ordner (oder unter
`M2V_HOME`); git ignoriert sie.

## Ein trainiertes Modell nutzen

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("JanSchachtschabel/m2v-gte-256-edu-v2")
embeddings = model.encode(["Bruchrechnung für die 6. Klasse", "Photosynthese"])
```

Für kleingeschriebene Suchanfragen Anfragen und Dokumente vorher auf Substantiv-Grundformen reduzieren:

```python
from m2vtrain.lemma import to_lemma_text
model.encode([to_lemma_text("die schülerinnen rechnen mit brüchen")])   # -> "Schülerin Bruch"
```

## Weitere Dokumentation

- [Datenbasis und Lizenzen](docs/de/daten-und-lizenzen.md)
- [Lehrermodelle](docs/de/lehrermodelle.md)
- [Evaluation und k-Fold](docs/de/evaluierung.md)
- [GPU-Einrichtung (RTX 5060 Ti)](docs/de/gpu-einrichtung.md)
- [Konfiguration](docs/de/konfiguration.md)
- [Erfahrungen aus bisherigen Läufen](docs/de/erfahrungen.md)

## Tests

```bash
python -m pytest -q
```

Die Tests brauchen kein Internet; sie bauen ein winziges Zufalls-Lehrermodell und einen winzigen synthetischen
Datensatz lokal.

## Lizenz

Code: Apache 2.0 (siehe [LICENSE](LICENSE)). Trainierte Modelle übernehmen Pflichten von Lehrermodell und Daten:
`license` in der Konfiguration passend setzen (z. B. `apache-2.0` bei Apache-Lehrern, `mit` bei bge-m3) und die
Namensnennung des synthetischen Datensatzes (CC BY 4.0) beibehalten, die die erzeugte Model Card enthält.
