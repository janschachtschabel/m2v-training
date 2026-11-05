# Zwei GPUs und Fernsteuerung von einem anderen Rechner

*English: [../en/dual-gpu.md](../en/dual-gpu.md)*

Diese Seite gilt für einen Trainings-PC mit **zwei NVIDIA-GPUs mit je 16 GB**, den du von einem anderen Rechner
über SSH steuerst. Installation und Speicherhinweise für eine einzelne Karte stehen in
[gpu-einrichtung.md](gpu-einrichtung.md); sie gelten für jede der beiden Karten.

## Wie die zwei GPUs genutzt werden

| Phase | Was läuft | GPU-Nutzung |
|---|---|---|
| Lehrervergleich (3-Fold) | zwei Warteschlangen parallel, jede fest auf einer GPU (`CUDA_VISIBLE_DEVICES`) | GPU 0: gte-Referenz, bge-m3, arctic-embed-l-v2; GPU 1: Qwen3-Embedding-4B, Qwen3-Embedding-0.6B |
| Finaler 5-Fold-Lauf | ein Lauf; seine Lehrer-Embeddings werden auf beide GPUs verteilt (`teacher.devices`) | beide GPUs für die Lehrer-Embeddings (der längste Schritt), GPU 0 für Destillation und Training |

Die Aufteilung des Vergleichs verteilt die Arbeit ungefähr gleich: Qwen3-4B allein braucht etwa so lange wie die
drei kleineren Lehrer zusammen. Model2Vec-Modelle sind klein, Destillation, Tokenlearn und Paar-Training
profitieren deshalb nicht von einer zweiten GPU. Die Lehrer-Embeddings schon: Mit `teacher.devices` brauchen sie
etwa die halbe Zeit.

`teacher.devices: ["cuda:0", "cuda:1"]` startet pro GPU einen Arbeitsprozess mit eigener Lehrer-Kopie und teilt
jeden Block von 4.096 Texten gleichmäßig auf. Die Ergebnisse sind dieselben wie auf einer GPU (bis auf
Rundung) und landen im selben Cache; ein unterbrochener Lauf kann also mit einer oder zwei GPUs weitermachen.

Fertige Konfigurationen:

- `configs/rtx5060ti-0*.json`: der Vergleich; sie nutzen `"device": "cuda"`, also die GPU, auf die die
  Warteschlange festgelegt ist.
- `configs/dual-gpu/final-5fold-qwen3-4b.json`: finaler Lauf mit Qwen3-Embedding-4B, 5 Folds plus finales Modell
  auf allen Daten. Ist Qwen3-4B nicht der Gewinner des Vergleichs, den Gewinner unter `teacher` eintragen.
- `configs/dual-gpu/final-5fold-qwen3-4b-512.json`: dasselbe mit 512 Dimensionen. Nach dem 256er-Lauf gestartet,
  nutzt er dessen gespeicherte Lehrer-Embeddings und kostet nur die Trainingszeit.

## Speicher

- GPU: Jede Warteschlange und jeder Arbeitsprozess hält einen Lehrer; Qwen3-4B braucht in bf16 etwa 8 GB pro
  Karte, die anderen deutlich weniger.
- RAM: Beim Vergleich sind zwei Läufe gleichzeitig im Speicher (Lehrer-Vektoren von Qwen3-4B etwa 4 GB, von einem
  1024-dimensionalen Lehrer etwa 1,5 GB, dazu eine Kopie während der PCA und das Laden des Lehrers). Mit
  **32 GB RAM** ist das bequem. Bei 16 GB die Warteschlangen nacheinander starten (`start <config> 0`, später
  `start <config> 1`) oder `tokenlearn.max_texts` setzen.
- Beide Läufe teilen sich `cache/`; die Dateien sind nach Lehrer und Texten benannt und kommen sich nicht in die
  Quere.

## Einmalige Einrichtung auf dem GPU-PC

1. Installieren wie in [gpu-einrichtung.md](gpu-einrichtung.md) (`scripts/setup_linux.sh` oder
   `scripts\setup_windows.ps1`) und `python scripts/check_gpu.py` aufrufen: Es muss `cuda:0` und `cuda:1`
   anzeigen.
2. SSH einschalten:
   - Windows: Einstellungen > System > Optionale Features > „OpenSSH-Server“ hinzufügen, dann in einer
     Administrator-PowerShell `Start-Service sshd; Set-Service sshd -StartupType Automatic`. Die Standard-Shell
     bleibt `cmd.exe`; davon gehen die Skripte aus.
   - Linux: `sudo apt install openssh-server`.
   - Mit SSH-Schlüssel anmelden, nicht mit Passwort (`ssh-keygen` auf dem anderen Rechner, den öffentlichen
     Schlüssel auf dem GPU-PC in `~/.ssh/authorized_keys` eintragen; bei einem Administratorkonto unter Windows
     in `C:\ProgramData\ssh\administrators_authorized_keys`).
   - SSH nur im eigenen Netz oder per VPN erreichbar machen, Port 22 nicht ins Internet freigeben.
3. Hugging-Face-Anmeldung auf dem GPU-PC (nur für Uploads nötig): `hf auth login`.
4. **Daten kopieren.** Die Keyword-Liste und lokale Exporte sind nie im Repository und gehen nie über git. Direkt
   nach `<Repository>/data/` kopieren, etwa mit `remote.sh push-data data/keywords.txt` (siehe unten), per
   USB-Stick oder Netzlaufwerk. `start-compare` und `start-final` starten nicht, wenn eine Konfiguration
   `data/keywords.txt` braucht und die Datei fehlt. `.gitignore` schließt `data/`, `runs/`, `models/` und
   `cache/` aus.

## Steuerung vom anderen Rechner

Auf dem anderen Rechner braucht es nur einen Klon des Repositorys (für die Skripte) und den SSH-Client.

```bash
# Linux, macOS, WSL, Git Bash
export M2V_HOST=jan@gpu-pc
export M2V_SERVER_OS=windows                 # oder linux
export M2V_REPO=C:/Users/jan/m2v-training    # Repository auf dem GPU-PC (Linux z. B. ~/m2v-training)
./scripts/remote.sh check
```

```powershell
# Windows PowerShell
$env:M2V_HOST = "jan@gpu-pc"
$env:M2V_SERVER_OS = "windows"               # oder "linux"
$env:M2V_REPO = "C:/Users/jan/m2v-training"  # Pfad ohne Leerzeichen
.\scripts\remote.ps1 check
```

| Aktion | Was sie auf dem GPU-PC tut |
|---|---|
| `push-data <datei>` | kopiert eine lokale Datendatei nach `<Repository>/data/` (scp) |
| `update` | `git pull --ff-only` und Paket neu installieren |
| `check` | GPUs, CUDA-Build von torch, Keyword-Liste vorhanden |
| `start-compare` | Lehrervergleich, eine Warteschlange pro GPU |
| `start-final [config]` | finaler Lauf auf beiden GPUs (Standard `configs/dual-gpu/final-5fold-qwen3-4b.json`) |
| `start <config> [gpu]` | beliebige Konfiguration, wahlweise fest auf GPU 0 oder 1 |
| `status` | Läufe mit Zustand und Schritt, GPU-Last und -Speicher, letzte Zeilen der Warteschlangen |
| `log [zeilen]` | Ende des Logs des neuesten Laufs |
| `results [--mode lemma]` | Markdown-Tabelle zum Vergleich der fertigen Läufe |
| `stop` | beendet alle Warteschlangen (vor ihrer nächsten Konfiguration) und alle laufenden Läufe |
| `ui` | startet die Streamlit-Oberfläche auf dem GPU-PC, nur auf localhost erreichbar |
| `tunnel` | SSH-Tunnel; danach auf dem anderen Rechner http://localhost:8501 öffnen |
| `fetch <pfad>` | kopiert z. B. `models/<name>` vom GPU-PC in den aktuellen Ordner |

Ein typischer Ablauf:

```bash
./scripts/remote.sh push-data data/keywords.txt
./scripts/remote.sh update
./scripts/remote.sh start-compare        # eine Nacht
./scripts/remote.sh status               # jederzeit
./scripts/remote.sh results              # am nächsten Morgen: Gewinner wählen
# Gewinner-Lehrer in configs/dual-gpu/final-5fold-qwen3-4b.json eintragen (committen, pushen, dann 'update')
./scripts/remote.sh start-final
./scripts/remote.sh start-final configs/dual-gpu/final-5fold-qwen3-4b-512.json   # nach dem ersten
ssh $M2V_HOST   # dann auf dem GPU-PC: python -m m2vtrain upload models/<name> JanSchachtschabel/<name>
```

Jeder Auftrag wird abgekoppelt gestartet (Windows über WMI, Linux mit `setsid nohup`). Das Schließen der
SSH-Sitzung oder ein Neustart des anderen Rechners beendet also keinen Lauf. Ein abgestürzter oder gestoppter
Lauf setzt die Lehrer-Embeddings beim erneuten Start mit gleicher Konfiguration aus den gespeicherten Blöcken
fort.

Die Skripte auf dem GPU-PC lassen sich dort auch direkt aufrufen: `scripts/server.sh <aktion>` (Linux) oder
`.\scripts\server.ps1 <aktion>` (Windows); `python -m m2vtrain status` und `python -m m2vtrain stop`
funktionieren überall.

## Hinweise

- `update` holt den Branch, der auf dem GPU-PC ausgecheckt ist. Konfigurationen auf dem anderen Rechner ändern,
  pushen, dann `update`; oder direkt auf dem GPU-PC ändern (dann verweigert `git pull` womöglich, bis die
  Änderung committet oder verworfen ist).
- Die Ausgabe der Warteschlangen steht in `runs/queue-<name>.log`, das Log des Laufs in `runs/<lauf>/run.log`,
  sein Zustand in `runs/<lauf>/status.json` (auch über `status` und die Oberfläche).
- Model Cards verbergen standardmäßig die Namen lokaler Dateien und die Zahl der Dokumente
  (`card.show_data_facts: false`). Ein mit der Keyword-Liste trainiertes Modell lässt sich also hochladen, ohne
  diese Liste zu beschreiben. Siehe [daten-und-lizenzen.md](daten-und-lizenzen.md).
