# Evaluation und k-Fold-Kreuzvalidierung

*English: [../en/evaluation.md](../en/evaluation.md)*

## Kennzahlen

Jede Kennzahl wird zweimal berechnet: mit **Rohtext** (`raw`) und mit **Grundformen** (`lemma`: Anfragen und
Dokumente auf Substantiv-Grundformen reduziert, z. B. „Die Schülerinnen rechnen mit Brüchen“ → „Schülerin
Bruch“). Dokumente werden als „Titel. Beschreibung“ eingebettet, **ohne** ihre Keywords; das Modell muss das
Konzept also im Text erkennen.

| Kennzahl | Aufgabe |
|---|---|
| `keyword_ndcg@10`, `keyword_recall@10` | jedes Keyword der Testdokumente (von mindestens 2 verwendet) ist eine Anfrage; relevant sind die Testdokumente mit diesem Keyword |
| `keyword_lower_*` | dasselbe mit kleingeschriebenen Anfragen, wie Nutzer tippen |
| `title_mrr@10`, `title_recall@10` | zum Titel die passende Beschreibung finden (mehrdeutige Titel/Beschreibungen bleiben außen vor) |
| `question_mrr@10`, `question_recall@10` | ein Dokument über eine seiner Fragen finden (nur synthetische Datensätze; eine Frage pro Datensatz, per Seed gewählt) |
| `subject_macro_f1` | logistische Regression auf den eingefrorenen Embeddings, trainiert auf dem Trainingsteil, Dokumente mit genau einem Fach |

Jeder Lauf bewertet zusätzlich **TF-IDF** (auf den Trainingsdokumenten angepasst, pro Eingabeart), die Modelle aus
`baseline_models` (Model2Vec- oder Sentence-Transformers-Modelle) und optional den Lehrer (`evaluate_teacher`).

## Einfacher Testteil

`cross_validation.folds: 0` (Standard) hält `test_fraction` der Dokumente zurück. Beinahe-Dubletten (gleicher
normalisierter Titel oder ausreichend lange gleiche Beschreibung) und Datensätze mit gleicher `group` (das Thema
eines synthetischen Datensatzes) landen immer auf derselben Seite.

## k-Fold-Kreuzvalidierung

```json
"cross_validation": {"folds": 5, "final_model": true, "shared_vocabulary": false}
```

1. Jedes Dokument bekommt eine Fold-Nummer aus einem Hash seiner Gruppe mit Seed. Die Aufteilung ist dadurch
   reproduzierbar, unabhängig von der Zeilenreihenfolge und hält Themen und Beinahe-Dubletten zusammen.
2. Die Tokenlearn-Texte aller Dokumente und die Begriffe der Keyword-Listen werden **einmal** vom Lehrer
   eingebettet (Zwischenspeicher in `cache/teacher/`).
3. Für jeden Fold *i*: Vokabular und Destillation aus den Dokumenten der anderen Folds, Tokenlearn auf deren
   Texten und den Keyword-Begriffen (die PCA der Ziele wird genau auf diesen Texten angepasst), Paar-Training auf
   deren Paaren, dann die Evaluation auf Fold *i*. Ergebnisse: `runs/<lauf>/folds/fold-<i>/eval.json`.
4. `runs/<lauf>/eval.json` enthält den Mittelwert pro Modell, Eingabeart und Kennzahl (gleiches Format wie eine
   einfache Evaluation, Oberfläche und `compare` funktionieren also unverändert), dazu `cv.std` und die Werte jedes
   Folds in `cv.values`.
5. Mit `final_model: true` wird danach ein letztes Modell auf **allen** Dokumenten trainiert und gespeichert; seine
   Card zeigt die kreuzvalidierten Kennzahlen als Mittelwert ± Standardabweichung. Das ist das übliche Vorgehen
   „mit Kreuzvalidierung schätzen, dann auf allen Daten trainieren“: Die veröffentlichte Zahl beschreibt das
   Trainingsrezept; das finale Modell hat mehr Daten gesehen und sollte mindestens so gut sein.

Jedes Dokument wird genau einmal getestet und k−1-mal zum Training genutzt. Bei k = 5 trainiert jedes Modell auf
80 % der Daten, die Schätzung liegt also nah am finalen Modell.

`shared_vocabulary: true` baut das Vokabular aus allen Dokumenten und destilliert nur einmal. Das spart k−1
Destillationen (auf der GPU Minuten, auf der CPU bei 250k Vokabular je etwa eine halbe Stunde), ist aber leicht
optimistisch: Wörter und Keywords des Test-Folds existieren als Tokens, auch wenn ihre Vektoren nur auf Texten der
anderen Folds trainiert werden. Gut für schnelle Experimente (der CPU-Schnelltest nutzt es), nicht für
veröffentlichte Zahlen.

Was außerhalb der Folds bleibt:

- **Keyword-Begriffe** sind in jedem Fold Trainingsdaten und nie Anfragen. Das ist gewollt: Das finale Modell
  kennt sie auch. Die Keyword-Suche wird mit den Keywords zurückgehaltener synthetischer Datensätze gemessen; steht
  ein Begriff auch in der Keyword-Liste, hat das Modell vom Lehrer gelernt, was er bedeutet, nicht welche
  Dokumente ihn tragen.
- **Lehrer-Embeddings** werden für alle Texte auf einmal berechnet; sie hängen nur vom Text ab, nicht von der
  Aufteilung.

## Läufe vergleichen

```bash
python -m m2vtrain compare                 # alle Läufe in runs/
python -m m2vtrain compare runs/a runs/b --mode lemma
```

Direkt vergleichbar sind nur Zeilen mit gleichem **Split-Fingerabdruck**: gleiche Daten, `seed`, `test_fraction`
bzw. Anzahl Folds. Bei kreuzvalidierten Läufen sind Unterschiede unter etwa der doppelten Standardabweichung als
Rauschen zu behandeln. Die Seite „Vergleich“ der Oberfläche zeigt dieselben Mittelwerte.

## Was die Zahlen sagen und was nicht

Die synthetischen Datensätze sind sauber, einheitlich und von einem Modell geschrieben. Die Werte darauf sind
höher und streuen weniger als auf echten, verrauschten Metadaten; sie ordnen Lehrer und Rezepte gegeneinander,
sagen die Qualität bei echten Suchanfragen aber nicht genau voraus. Eine kleine Menge echter Anfragen mit
bewerteten Treffern (schon 100) wäre die wertvollste Ergänzung dieser Evaluation.
