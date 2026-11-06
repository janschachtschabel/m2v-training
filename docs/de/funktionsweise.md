# Funktionsweise

*English: [../en/how-it-works.md](../en/how-it-works.md)*

Ein Model2Vec-Modell ist eine Tabelle mit einem Vektor pro Token. Um einen Text einzubetten, zerlegt der
Tokenizer ihn in Tokens, das Modell schlägt deren Vektoren nach und mittelt sie (pro Token gewichtet). Es gibt
keine Attention und keinen Schichtenstapel, das Einbetten kostet also fast nichts. Die Qualität steckt allein in
den Vektoren der Tabelle, und diese Kette ist dazu da, die Tabelle für deutsche Bildungsinhalte gut zu füllen.

Ein Lauf geht durch die Schritte unten. Mit k-Fold-Kreuzvalidierung laufen die Schritte 3 bis 8 einmal pro Fold
und, wenn gewünscht, noch einmal auf allen Daten für das finale Modell.

## 1. Daten laden

| Quelle | Wird zu |
|---|---|
| Synthetischer Datensatz (CC BY 4.0) | ein Dokument pro Datensatz: Titel, Beschreibung, Keywords, Fach, Thema. Volltexte werden in Abschnitte aus ganzen Absätzen geteilt (bis 1.200 Zeichen, ohne Markdown). Bis zu 3 Fragen pro Datensatz. |
| Keyword-Liste | reine Begriffe, einer pro Zeile. Sie sind nie Dokumente und nie Testanfragen, sondern speisen nur Vokabular, Tokenlearn und Paare Begriff → Dokument. |
| Lokale Exporte (optional) | Dokumente aus `;`-getrennten CSV-Metadaten-Exporten, bereinigt (Sprache, Boilerplate, Beschreibungslänge, Keyword-Stoppliste). |

Details und Lizenzen: [daten-und-lizenzen.md](daten-und-lizenzen.md).

## 2. Ohne Datenleck aufteilen

Beinahe-Dubletten (gleicher normalisierter Titel oder eine lange, identische Beschreibung) und Dokumente zum
selben Thema werden zu einer Gruppe zusammengefasst. Eine Gruppe bleibt immer auf einer Seite: Beim einfachen
Testteil wird `test_fraction` der Gruppen zurückgehalten; bei k Folds bekommt jede Gruppe ihren Fold aus einem
Hash mit Seed, die Aufteilung hängt also nicht von der Reihenfolge der Zeilen ab. Ohne das würde ein Modell auf
Beinahe-Kopien seiner Trainingsdaten getestet und besser aussehen, als es ist. Mehr in
[evaluierung.md](evaluierung.md).

## 3. Vokabular aufbauen

Der Tokenizer des Lehrers zerlegt seltene deutsche Wörter in Stücke (zum Beispiel „Photo“, „syn“, „these“), und
der Mittelwert der Stücke ist ein schlechter Vektor für das Wort. Deshalb kommen ganze Wörter als eigene Tokens
dazu:

- **Keywords**: Begriffe der Keyword-Liste und Dokument-Keywords, die mindestens `vocab.keyword_min_docs`
  Dokumente nutzen; bleiben immer erhalten
- **Wörter des Korpus**, die mindestens `vocab.min_word_count`-mal vorkommen
- **Grundformen der Substantive** dieser Wörter („Brüchen“ -> „Bruch“), damit lemmatisierte Eingabe ein Token
  findet
- **Kleinschreibvarianten** („photosynthese“), weil der Tokenizer Groß- und Kleinschreibung unterscheidet und
  Menschen klein tippen

`vocab.max_vocab_size` begrenzt die Einträge aus dem Korpus; Keywords zählen nicht mit.

## 4. Aus dem Lehrer destillieren

model2vec schickt jeden Vokabular-Eintrag einmal durch den Lehrer (Pooling `mean`, bei Decoder-Lehrern wie
Qwen3-Embedding `last`), reduziert die Vektoren per PCA auf `pca_dims` Dimensionen und gewichtet sehr häufige
Tokens herunter (SIF-Gewichtung). Das Ergebnis ist schon ein brauchbares statisches Modell, aber jedes Token wurde
allein gesehen, nie im Zusammenhang.

## 5. Lehrer-Embeddings ganzer Texte

Der Lehrer bettet die Trainingstexte vollständig ein: Dokumente („Titel. Beschreibung. Keywords“),
Volltext-Abschnitte, Fragen und die Keyword-Begriffe. Diese Satz-Embeddings sind die Ziele für den nächsten
Schritt. Sie sind der teuerste Teil eines Laufs, deshalb werden sie einmal für alle Folds berechnet, in
`cache/teacher/` in Blöcken von 4.096 Texten gespeichert (ein unterbrochener Lauf macht beim letzten fertigen
Block weiter) und mit `teacher.devices` auf mehrere GPUs verteilt. Vor dem Training werden die Ziele per PCA auf
`pca_dims` reduziert, angepasst auf die Trainingstexte des Folds.

## 6. Tokenlearn

Die Token-Vektoren werden so trainiert, dass der Mittelwert der Tokens eines Textes das Lehrer-Embedding dieses
Textes nachbildet (Kosinus-Verlust, Early Stopping auf einem Validierungsteil). Vor dem Training werden alle
Token-Vektoren in die Basis der Ziele gedreht (orthogonale Procrustes-Rotation), denn beim Training bewegen sich
nur Tokens, die in den Trainingstexten vorkommen, und die übrigen dürfen nicht in der alten Basis bleiben. Dazu
werden Gewichte pro Token gelernt, damit aussagekräftige Wörter mehr zählen als Füllwörter.

`input_mode` legt fest, was das statische Modell sieht: Rohtext (`raw`), Grundform-Text (`lemma`, nur
Substantiv-Grundformen) oder `mixed` (jeder Text auf eine der beiden Arten, `lemma_share` davon als Grundformen).
Der Lehrer sieht immer den Rohtext. Mit `mixed` lernt das Modell, „Schülerin Bruch“ an dieselbe Stelle zu legen
wie „Die Schülerinnen rechnen mit Brüchen“. Keyword-Begriffe werden zusätzlich in Klein- und Grundform-Schreibweise
trainiert (`keyword_lists.lowercase_inputs`, `lemma_inputs`).

Dieser Schritt war in früheren Läufen der größte einzelne Gewinn ([erfahrungen.md](erfahrungen.md)).

## 7. Paar-Training

Kontrastives Training (InfoNCE mit den anderen Beispielen eines Batches als Negativbeispielen) auf Paaren, die
nah beieinander landen sollen:

- Keyword -> Dokument (Dokument-Keywords und Begriffe der Keyword-Liste, die in einem Dokument vorkommen, bis zu
  `keyword_lists.max_docs_per_keyword` Dokumente pro Begriff)
- Titel -> Beschreibung
- Frage -> Dokument (synthetische Datensätze)

Das schärft genau die Aufgaben, für die die Modelle gedacht sind: Material zu einem Begriff oder einer Frage
finden.

## 8. Evaluieren

Das Modell wird auf den zurückgehaltenen Dokumenten (oder dem zurückgehaltenen Fold) getestet: Keyword-Suche,
kleingeschriebene Keyword-Suche, Titel -> Beschreibung, Frage -> Dokument und Fach-Klassifikation, jeweils mit
Rohtext und mit Grundformen, neben TF-IDF und den Vergleichsmodellen der Konfiguration. Dokumente werden ohne ihre
Keywords eingebettet; eine Keyword-Suche gelingt also nur, wenn das Modell das Konzept im Text erkennt. Kennzahlen
und ihre Grenzen: [evaluierung.md](evaluierung.md).

## 9. Speichern und veröffentlichen

Das Modell wird in `float16` gespeichert (oder `int8`; `vocabulary_quantization` kann die Tabelle weiter
verkleinern), zusammen mit einer Model Card (`README.md`), `training_config.json` und `eval_results.json`.
Standardmäßig enthalten diese Dateien keine lokalen Dateinamen und keine Dokument- oder Vokabularzahlen
(`card.show_data_facts`), und die Card nennt die CC-BY-4.0-Quelle der synthetischen Daten.
`python -m m2vtrain upload` veröffentlicht das Modell, standardmäßig privat.

## Was ein Lauf schreibt

| Pfad | Inhalt |
|---|---|
| `runs/<lauf>/config.json` | die vollständige Konfiguration; `run runs/<lauf>/config.json` wiederholt den Lauf |
| `runs/<lauf>/status.json` | Zustand, aktueller Schritt, Fortschritt, Lebenszeichen (gelesen von `status`, der Oberfläche und den Skripten) |
| `runs/<lauf>/run.log` | das Log des Laufs |
| `runs/<lauf>/eval.json` | alle Kennzahlen, Vergleichsmodelle, Split-Fingerabdruck; bei k Folds auch Mittelwert ± Streuung und `folds/fold-<i>/eval.json` |
| `models/<name>/` | das trainierte Modell mit Model Card, bereit für `upload` (einfacher Testteil oder finales Modell) |
| `cache/teacher/` | gespeicherte Lehrer-Embeddings, geteilt von allen Läufen mit gleichem Lehrer und gleichen Texten |

## Warum so

- **Warum kein Transformer im Einsatz?** Er ist pro Anfrage deutlich genauer, braucht aber für große Bestände und
  Suche während des Tippens eine GPU oder viel CPU-Zeit. Das statische Modell gibt etwas Qualität für Tempo und
  Einfachheit auf; die Evaluation zeigt, wie viel.
- **Warum nicht einfach TF-IDF?** TF-IDF vergleicht Zeichenketten. Es weiß nicht, dass „Bruchrechnung“ und
  „Brüche addieren“ zusammengehören. Für die Keyword-Suche ist es trotzdem ein starker Vergleich, deshalb weist
  jeder Lauf es aus.
- **Warum destillieren und dann trainieren statt nur destillieren?** Die Destillation sieht jedes Token allein.
  Tokenlearn und Paar-Training bringen den Vektoren bei, wie sich Wörter in echten Texten und Anfragen verhalten.
- **Warum synthetische Daten?** Sie lassen sich veröffentlichen und dokumentieren, und ihre Fragen und Volltexte
  liefern Trainingssignal, das Metadaten allein nicht haben. Echte Metadaten bleiben für private Experimente
  nützlich.
