# Lehrermodelle

*English: [../en/teacher-models.md](../en/teacher-models.md)*

Das Lehrermodell ist der Transformer, dessen Wissen das statische Modell übernimmt: einmal über die Destillation
(jeder Vokabular-Eintrag läuft durch den Lehrer) und, wichtiger, über Tokenlearn (das statische Modell lernt, die
Satz-Embeddings des Lehrers nachzubilden). Ein im Deutschen stärkerer Lehrer hebt also die Obergrenze.

Anforderungen für den neuen Rechner (NVIDIA RTX 5060 Ti, 16 GB):

- **freie Lizenz** (Apache 2.0 oder MIT; keine Non-Commercial- oder eigenen Modelllizenzen)
- **mehrsprachig mit starkem Deutsch**
- **passt in 16 GB** und lässt Platz für Aktivierungen bei Batchgrößen, die die GPU auslasten

## Kandidaten

Deutsche Suche: MIRACL Deutsch nDCG@10; mehrsprachig gesamt: MMTEB-Mittel. Beides wie in
[diesem Vergleich](https://wz-it.com/en/blog/best-embedding-models-german/) angegeben (MTEB-Ergebnisse, September
2026); „–“ heißt: kein geprüfter Wert.

| Modell | Lizenz | Parameter | Dim | Gewichte (bf16/fp16) | MIRACL-de | MMTEB | Pooling Destillation | Konfiguration |
|---|---|---|---|---|---|---|---|---|
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Apache 2.0 | 4,0 Mrd. | 2560 | ~8 GB | **62,98** | **69,45** | `last` | `rtx5060ti-05-qwen3-embedding-4b.json` |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | MIT | 568 Mio. | 1024 | ~1,1 GB | 57,59 | 59,56 | `mean` | `rtx5060ti-02-bge-m3.json` |
| [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | Apache 2.0 | 568 Mio. | 1024 | ~1,1 GB | – | – | `mean` | `rtx5060ti-03-arctic-embed-l-v2.json` |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache 2.0 | 0,6 Mrd. | 1024 | ~1,2 GB | 54,21 | 64,33 | `last` | `rtx5060ti-04-qwen3-embedding-0.6b.json` |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) (bisher) | Apache 2.0 | 305 Mio. | 768 | ~0,6 GB | – | – | `mean` | `rtx5060ti-01-gte-reference.json` |

### Empfehlung

1. **Qwen3-Embedding-4B** ist das stärkste frei lizenzierte Modell für deutsche Suche, das auf die Karte passt:
   Rund 8 GB Gewichte in bfloat16 lassen Platz für Batches von 16 Texten mit 512 Tokens. Es ist ein
   Decoder-Modell: gepoolt wird das letzte Token, die Destillation nutzt `pooling: "last"`. Es ist auch der
   langsamste Lehrer (Laufzeiten in [gpu-einrichtung.md](gpu-einrichtung.md)), aber die Lehrer-Embeddings werden
   zwischengespeichert und nur einmal für alle Folds berechnet.
2. **bge-m3** ist die robuste Encoder-Alternative: deutlich schneller, mit Model2Vec erprobt (MinishLabs
   `potion-multilingual-128M` ist daraus destilliert, ebenso das frühere `JanSchachtschabel/m2v-bge-m3-edu`).
3. **gte-multilingual-base** bleibt als Referenz im Vergleich, damit sich der Effekt des Lehrers vom Effekt der
   neuen Datenbasis trennen lässt.

01, 02 und 05 laufen lassen (3-Fold, ohne finales Modell), mit `python -m m2vtrain compare` vergleichen, dann den
Gewinner in `rtx5060ti-final-5fold.json` eintragen (dort steht Qwen3-Embedding-4B) und diesen Lauf starten. 03
und 04 bei Zeit ergänzen. Ein Lehrer mit dem besten MTEB-Wert ergibt nicht automatisch das beste *statische*
Modell; deshalb wird gemessen statt angenommen. Gerade Decoder-Lehrer sind für die Model2Vec-Destillation weniger
erprobt; ihre Qualität auf Satzebene überträgt vor allem Tokenlearn.

### Nicht empfohlen

| Modell | Grund |
|---|---|
| Qwen3-Embedding-8B (Apache 2.0) | ~15 GB Gewichte in bf16 lassen auf 16 GB keinen Platz für Aktivierungen |
| jina-embeddings-v3 / v5 | CC BY-NC 4.0, nicht frei für kommerzielle Nutzung |
| jina-embeddings-v4 | Qwen-Research-Lizenz |
| EmbeddingGemma-300m | Gemma-Nutzungsbedingungen, keine offene Lizenz |
| EmbeddingGemma 2 (seit Oktober 2026 Apache 2.0) | vielversprechend und klein (270 Mio. Textteil), aber multimodal, sehr neu und noch ohne deutsche Suchwerte; später testen |
| granite-embedding-311m-multilingual-r2 (Apache 2.0) | frei, aber im Deutschen schwächer (MIRACL-de 50,97) |
| multilingual-e5-large-instruct (MIT) | in deutscher Suche schwächer (MIRACL-de 43,37), braucht Präfixe |

## Einstellungen pro Lehrer

| Einstellung | gte-multilingual | bge-m3 / arctic-l-v2 | Qwen3-Embedding |
|---|---|---|---|
| `teacher.trust_remote_code` | `true` | `false` | `false` |
| `teacher.pooling` (Destillation) | `mean` | `mean` | `last` |
| `teacher.dtype` | `float16` | `float16` | `bfloat16` |
| `teacher.batch_size` (16 GB, 512 Tokens) | 128 | 64 | 16 (4B), 64 (0.6B) |
| `teacher.document_prefix` | keins | keins | keins (Qwens Instruktion gilt nur für Anfragen) |
| `license` des trainierten Modells | `apache-2.0` | `mit` / `apache-2.0` | `apache-2.0` |

`pca_dims` bleibt in allen Konfigurationen bei 256, damit die Modelle mit den veröffentlichten vergleichbar sind;
512 Dimensionen sind ein günstiges Folge-Experiment für den Gewinner.

## Arbeitsspeicher für Lehrer-Embeddings

Alle Lehrer-Embeddings eines Laufs liegen während des Trainings als float32 im RAM: grob
`Texte × Lehrer-Dimension × 4 Byte`. Mit dem synthetischen Datensatz (Dokumenttext, etwa drei
Volltext-Abschnitte und drei Fragen pro Datensatz) und einer Keyword-Liste von einigen zehntausend Begriffen sind
das einige hunderttausend Texte: etwa 1,5 GB bei einem 1024-dimensionalen Lehrer und etwa 4 GB bei
Qwen3-Embedding-4B (2560 Dimensionen), dazu eine vorübergehende Kopie während der PCA. Bei wenig RAM
`tokenlearn.max_texts` oder `synthetic.max_questions_per_doc` senken.
