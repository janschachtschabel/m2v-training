# Datenbasis und Lizenzen

*English: [../en/data-and-licensing.md](../en/data-and-licensing.md)*

Das Trainingssetup kann drei Arten von Quellen kombinieren. Für ein Modell, dessen Trainingsdaten sich
dokumentieren und veröffentlichen lassen, nur die ersten beiden verwenden.

## 1. Synthetischer Datensatz (CC BY 4.0)

[`JanSchachtschabel/synthetic-german-educational-materials-50k`](https://huggingface.co/datasets/JanSchachtschabel/synthetic-german-educational-materials-50k)
enthält 50.000 deutsche Bildungsmaterialien, die ein Sprachmodell für alle Bildungsbereiche geschrieben hat
(Schule, Hochschule, berufliche Bildung, Erwachsenenbildung, frühkindliche Bildung). Kein Text ist aus bestehenden
Materialien kopiert. Pro Datensatz nutzt die Pipeline:

| Feld | Verwendung |
|---|---|
| `title`, `description` | Dokumenttext (Tokenlearn, Paare Titel → Beschreibung, Evaluation) |
| `keywords` | Vokabular, Paare Keyword → Dokument, Evaluation der Keyword-Suche |
| `fulltext` (Markdown) | ohne Markdown, in Abschnitte von etwa 1.200 Zeichen geteilt, Tokenlearn-Texte und Vokabular |
| `qa_pairs.question` | Tokenlearn-Texte und Paare Frage → Dokument (standardmäßig bis zu 3 pro Datensatz); die Evaluation nutzt die Fragen der zurückgehaltenen Datensätze |
| `qa_pairs.answer` | nur mit `synthetic.use_answers: true` |
| `subject` | Fach-Klassifikation in der Evaluation |
| `topic_uri` / `topic` | Gruppierung: Datensätze zum selben Thema landen immer im selben Fold |

Die Aufteilung des Datensatzes in train/validation/test wird nicht übernommen: `m2vtrain` liest alle Teile und
trennt selbst, gruppiert nach Thema. So kann die k-Fold-Kreuzvalidierung jeden Datensatz einmal zum Testen und
sonst zum Training nutzen.

Geladen wird vom Hub (`synthetic.dataset`, optional fest auf `synthetic.revision`) oder aus einem lokalen Ordner
bzw. einer Parquet-Datei. **Namensnennung:** CC BY 4.0 verlangt die Nennung der Quelle; die erzeugte Model Card
verlinkt Datensatz und Lizenz automatisch.

Grenzen: Die Texte sind synthetisch, nicht fachlich geprüft, stammen von einem Modell mit einem Prompt-Set und
beschreiben das deutsche Bildungssystem. Kennzahlen auf diesen Daten messen, wie gut ein Modell mit dieser Art
Text umgeht, nicht die Qualität bei echten Suchanfragen.

## 2. Keyword-Listen (nur Begriffe)

Eine Keyword-Liste ist eine Textdatei mit einem Begriff pro Zeile (optional `Begriff<TAB>Anzahl`; Zeilen mit `#`
am Anfang sind Kommentare). Sie enthält **keine Titel, Beschreibungen, IDs, URLs oder Verweise auf Ressourcen**;
nichts darin führt zu einzelnen Datensätzen zurück.

Erzeugt wird sie lokal aus einem Metadaten-Export (z. B. einem edu-sharing-/WLO-CSV-Export):

```bash
python -m m2vtrain extract-keywords data/export.csv.gz data/keywords.txt --min-docs 3
```

- Keywords werden wie im Training bereinigt (Stoppliste allgemeiner Plattform- und Format-Tags, Längengrenzen,
  keine URLs oder reinen Zahlen) und ohne Rücksicht auf Groß-/Kleinschreibung unter der häufigsten Schreibweise
  zusammengefasst.
- `--min-docs 3` behält nur Begriffe, die mindestens drei Dokumente verwenden; Tippfehler und Einzel-Tags fallen
  weg.
- Ohne `--with-counts` enthält die Datei nur die Begriffe und verrät nichts über die Größe der Quelldaten.

So werden die Begriffe im Training genutzt (`keyword_lists.*`):

| Option | Standard | Wirkung |
|---|---|---|
| `add_to_vocabulary` | an | jeder Begriff (und seine Kleinschreibung) wird ein Token und nicht in Wortteile zerlegt |
| `tokenlearn` | an | das statische Modell lernt das Lehrer-Embedding jedes Begriffs |
| `lowercase_inputs`, `lemma_inputs` | an | Kleinschreibung und Grundform werden auf dasselbe Ziel trainiert, damit kleingeschriebene und gebeugte Anfragen funktionieren |
| `pairs` | an | Begriff → Dokumente des Korpus, die ihn als ganzes Wort enthalten (bis zu `max_docs_per_keyword`) |

Die Begriffe sind in jedem Fold Trainingsdaten und nie Suchanfragen der Evaluation. Die Keyword-Anfragen der
Evaluation sind die Keywords der zurückgehaltenen synthetischen Datensätze.

## 3. Lokale Exporte (optional, privat)

`data_files` nimmt weiterhin `;`-getrennte CSV-Exporte (auch `.csv.gz`) mit `properties.cclom:*`-Spalten. Nur für
private Experimente verwenden. Das Repository enthält sie nie (`data/` wird von git ignoriert), die veröffentlichte
`training_config.json` ersetzt ihre Namen durch `<local file>`, und die Model Card sagt nur „further German
educational texts“, ohne Herkunft, Umfang oder Art der Daten.

## Was nie ins Repository kommt

- Datendateien, Keyword-Listen, Laufordner, Modelle und Lehrer-Caches (`data/`, `runs/`, `models/`, `cache/` sind
  in `.gitignore`)
- Hugging-Face-Tokens: Anmeldung mit `hf auth login` oder `HF_TOKEN` als Umgebungsvariable, nie in einer
  Konfiguration
- Anzahl oder Beschreibung privater Daten in Doku, Model Cards oder Commit-Nachrichten

## Model Cards

Standardmäßig nennt die Card nur frei lizenzierte Quellen und keine Dokumentzahlen (`card.show_data_facts`
schaltet Zahlen ein, nur für Modelle aus öffentlichen Daten). `card.data_description` ersetzt den automatischen
Text ganz. `eval_results.json` neben dem Modell lässt Laufdaten und Testgröße weg, solange `show_data_facts` aus
ist.

## Vor einer Veröffentlichung

1. Lizenz des Lehrermodells prüfen (siehe [lehrermodelle.md](lehrermodelle.md)) und `license` passend setzen.
2. Die Namensnennung des synthetischen Datensatzes (CC BY 4.0) in der Card lassen.
3. Die Bedingungen der Metadatenquelle prüfen, aus der die Keyword-Liste stammt. Einzelne Schlagwörter und kurze
   Fachbegriffe sind in der Regel nicht als Werke geschützt, Datenbankrechte oder Nutzungsbedingungen der Quelle
   können aber für systematische Auszüge gelten. Mit dem Modell wird die Liste selbst nicht veröffentlicht; für
   eine Veröffentlichung der Liste wäre diese Prüfung nötig. (Praktischer Hinweis, keine Rechtsberatung.)
4. Die erzeugte Card einmal lesen: Sie darf keine lokalen Dateinamen, Zahlen oder die Art privater Daten nennen.
