# GPU-Einrichtung (NVIDIA RTX 5060 Ti, 16 GB)

*English: [../en/gpu-setup.md](../en/gpu-setup.md)*

## Installation

Die RTX-50-Serie (Blackwell, Compute Capability 12.0 / `sm_120`) braucht einen aktuellen NVIDIA-Treiber und ein
PyTorch-Build für **CUDA 12.8 oder neuer**. Ältere Builds laden zwar, scheitern aber mit „no kernel image is
available“.

Windows (PowerShell, Python 3.12 installiert):

```powershell
git clone https://github.com/janschachtschabel/m2v-training
cd m2v-training
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # einmalig, falls Skripte blockiert sind
.\scripts\setup_windows.ps1                          # venv, torch (cu128), Paket, GPU-Check
```

Linux:

```bash
./scripts/setup_linux.sh python3.12 https://download.pytorch.org/whl/cu128
```

`scripts/check_gpu.py` zeigt GPU, Speicher und Compute Capability und testet eine bfloat16-Matrixmultiplikation.
Meldet es einen fehlenden `sm_120`-Kernel, torch von einem neueren CUDA-Index installieren (z. B. `cu129` oder
`cu130`).

Hugging Face: einmal `hf auth login` (nötig für Uploads; öffentliche Modelle und der Datensatz laden auch ohne).
Die Keyword-Liste nach `data/keywords.txt` auf den neuen Rechner kopieren; sie ist nicht im Repository.

## Starten

```bash
python -m m2vtrain run configs/smoke-test-cpu.json            # Funktionstest, wenige Minuten
python -m m2vtrain run configs/rtx5060ti-01-gte-reference.json
python -m m2vtrain run configs/rtx5060ti-02-bge-m3.json
python -m m2vtrain run configs/rtx5060ti-05-qwen3-embedding-4b.json
python -m m2vtrain compare
# den besten Lehrer in configs/rtx5060ti-final-5fold.json eintragen, dann:
python -m m2vtrain run configs/rtx5060ti-final-5fold.json
python -m m2vtrain upload models/<lauf> JanSchachtschabel/<name>   # standardmäßig privat
```

Lange Läufe nicht als Kindprozess eines IDE-Terminals oder einer Remote-Sitzung starten: Wird dieser Elternprozess
geschlossen oder neu gestartet, endet auch der Lauf. Stattdessen:

- Windows: `.\scripts\run_detached.ps1 configs\rtx5060ti-02-bge-m3.json` (Start über WMI, Log in
  `runs\<config>-console.log`), oder `.\scripts\run_queue.ps1 configs\a.json configs\b.json` für mehrere
  Konfigurationen
- Linux: `nohup ./scripts/run_queue.sh configs/rtx5060ti-0*.json > runs/queue.log 2>&1 &`

Ein gestoppter oder abgestürzter Lauf setzt die Lehrer-Embeddings beim erneuten Start mit gleicher Konfiguration
dort fort, wo er aufgehört hat (Blöcke von 4.096 Texten in `cache/teacher/`).

## Was die GPU nutzt

| Schritt | GPU | Hinweis |
|---|---|---|
| Destillation | ja (`teacher.device`) | ein Durchlauf pro Vokabular-Eintrag; Minuten |
| Lehrer-Embeddings | ja | bei großen Lehrern der längste Schritt; zwischengespeichert, einmal für alle Folds |
| Tokenlearn, Paar-Training | ja (`tokenlearn.train.device`, `pairs.train.device`) | das statische Modell ist klein; das Laden der Daten dominiert |
| Evaluation | teilweise | Model2Vec-Modelle und TF-IDF laufen in Sekunden auf der CPU; Transformer-Vergleichsmodelle nutzen die GPU |

## Grobe Laufzeiten

Schätzungen zur Planung, keine Messungen auf dieser Karte; die ersten Läufe zeigen die echten Werte. Auf Jans
Laptop-CPU schaffte der gte-Lehrer etwa 7–10 Texte pro Sekunde. Auf einer Blackwell-Karte mit 16 GB grob:

| Lehrer | Lehrer-Embeddings (einige hunderttausend Texte, ≤512 Tokens) | pro Fold: Destillation + Tokenlearn + Paare |
|---|---|---|
| gte-multilingual-base (fp16) | deutlich unter einer Stunde | unter einer Stunde |
| bge-m3 / arctic-l-v2 (fp16) | etwa eine Stunde | unter einer Stunde |
| Qwen3-Embedding-4B (bf16, Batch 16) | mehrere Stunden | etwa eine Stunde |

Ein 3-Fold-Lehrervergleich passt also in eine Nacht pro Lehrer, der finale 5-Fold-Lauf mit Qwen3-4B in ein bis
zwei Tage. Wird der Speicher knapp, zuerst `teacher.batch_size` senken, dann `teacher.max_seq_length`.

## Speicher

- GPU: Qwen3-Embedding-4B braucht in bf16 etwa 8 GB für die Gewichte; Batch 16 mit 512 Tokens passt in den Rest.
  Alle anderen Kandidaten brauchen deutlich weniger.
- RAM: Lehrer-Embeddings liegen als float32 vor (siehe [lehrermodelle.md](lehrermodelle.md#arbeitsspeicher-für-lehrer-embeddings)):
  etwa 1,5 GB bei 1024-dimensionalen Lehrern, etwa 4 GB bei Qwen3-4B, dazu eine vorübergehende Kopie während der
  PCA. Mit 32 GB RAM ist das bequem, 16 GB reichen mit `tokenlearn.max_texts` oder weniger Fragen pro Datensatz.
