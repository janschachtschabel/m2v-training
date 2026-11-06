# Two GPUs and remote control from another computer

*Deutsch: [../de/zwei-gpus.md](../de/zwei-gpus.md)*

This page is for a training PC with **two NVIDIA GPUs with 16 GB each** that you control from another computer
over SSH. Installation and memory notes for a single card are in [gpu-setup.md](gpu-setup.md); they apply to
each card.

## How the two GPUs are used

| Phase | What runs | GPU use |
|---|---|---|
| Teacher comparison (3-fold) | two queues in parallel, each pinned to one GPU (`CUDA_VISIBLE_DEVICES`) | GPU 0: gte reference, bge-m3, arctic-embed-l-v2; GPU 1: Qwen3-Embedding-4B, Qwen3-Embedding-0.6B |
| Final 5-fold run | one run; its teacher embeddings are split across both GPUs (`teacher.devices`) | both GPUs for the teacher embeddings (the longest step), GPU 0 for distillation and training |

The split of the comparison roughly balances the work: Qwen3-4B alone takes about as long as the three smaller
teachers together. Model2Vec models are small, so distillation, tokenlearn and pair training do not gain from a
second GPU; the teacher embeddings do, and with `teacher.devices` they take about half the time.

`teacher.devices: ["cuda:0", "cuda:1"]` starts one worker process per GPU, each with its own copy of the teacher,
and splits every block of 4,096 texts evenly between them. Results are the same as on one GPU (up to rounding)
and use the same cache, so an interrupted run can resume on one GPU or two.

Ready-made configs:

- `configs/rtx5060ti-0*.json`: the comparison; they use `"device": "cuda"`, which is whichever GPU the queue is
  pinned to.
- `configs/dual-gpu/final-5fold-qwen3-4b.json`: final run, Qwen3-Embedding-4B, 5 folds plus a final model on all
  data. Put the winner of the comparison in `teacher` if it is not Qwen3-4B.
- `configs/dual-gpu/final-5fold-qwen3-4b-512.json`: the same with 512 dimensions; started after the 256 run, it
  reuses the cached teacher embeddings and takes only the training time.

## Memory

- GPU: each queue or worker holds one teacher; Qwen3-4B in bf16 needs about 8 GB per card, the others far less.
- RAM: during the comparison two runs are in memory at once (teacher vectors of Qwen3-4B about 4 GB, of a
  1024-dimensional teacher about 1.5 GB, plus a copy during PCA, plus loading the teacher). **32 GB RAM** is
  comfortable; with 16 GB start the queues one after the other (`start <config> 0`, then `start <config> 1`
  later) or set `tokenlearn.max_texts`.
- Both runs share `cache/`; the cache files are named by teacher and texts, so they do not collide.

## One-time setup on the GPU PC

1. Install as in [gpu-setup.md](gpu-setup.md) (`scripts/setup_linux.sh` or `scripts\setup_windows.ps1`) and run
   `python scripts/check_gpu.py`: it must list `cuda:0` and `cuda:1`.
2. Enable SSH:
   - Windows: Settings > System > Optional features > add "OpenSSH Server", then in an admin PowerShell
     `Start-Service sshd; Set-Service sshd -StartupType Automatic`. Default shell stays `cmd.exe`; the scripts
     expect that.
   - Linux: `sudo apt install openssh-server`.
   - Log in with an SSH key, not a password (`ssh-keygen` on the other computer, then add the public key to
     `~/.ssh/authorized_keys` on the GPU PC; for an administrator account on Windows:
     `C:\ProgramData\ssh\administrators_authorized_keys`).
   - Keep SSH inside your network or VPN; do not forward port 22 from the internet.
3. Hugging Face login on the GPU PC (only needed for uploads): `hf auth login`.
4. **Copy the data.** The keyword list and any local exports are never in the repository and never go through
   git. Copy them directly, for example with `remote.sh push-data data/keywords.txt` (see below), a USB stick or
   a network share, to `<repository>/data/`. `start-compare` and `start-final` refuse to start if a config needs
   `data/keywords.txt` and the file is missing. `.gitignore` excludes `data/`, `runs/`, `models/` and `cache/`.

## Control from the other computer

On the other computer you need only a clone of the repository (for the scripts) and the SSH client.

```bash
# Linux, macOS, WSL, Git Bash
export M2V_HOST=jan@gpu-pc
export M2V_SERVER_OS=windows                 # or linux
export M2V_REPO=C:/Users/jan/m2v-training    # repository on the GPU PC (Linux e.g. ~/m2v-training)
./scripts/remote.sh check
```

```powershell
# Windows PowerShell
$env:M2V_HOST = "jan@gpu-pc"
$env:M2V_SERVER_OS = "windows"               # or "linux"
$env:M2V_REPO = "C:/Users/jan/m2v-training"  # no spaces in the path
.\scripts\remote.ps1 check
```

| Action | What it does on the GPU PC |
|---|---|
| `push-data <file>` | copies a local data file into `<repository>/data/` (scp) |
| `update` | `git pull --ff-only` and reinstall the package |
| `check` | GPUs, torch CUDA build, keyword list present |
| `start-compare` | teacher comparison, one queue per GPU |
| `start-final [config]` | final run on both GPUs (default `configs/dual-gpu/final-5fold-qwen3-4b.json`) |
| `start <config> [gpu]` | any config, optionally pinned to GPU 0 or 1 |
| `status` | runs with state and stage, GPU load and memory, last queue lines |
| `log [lines]` | end of the newest run log |
| `results [--mode lemma]` | Markdown table comparing the finished runs |
| `stop` | stops all queues (before their next config) and all running runs |
| `ui` | starts the Streamlit UI on the GPU PC, listening on localhost only |
| `tunnel` | SSH tunnel; then open http://localhost:8501 on the other computer |
| `fetch <path>` | copies e.g. `models/<name>` from the GPU PC into the current folder |

A typical session:

```bash
./scripts/remote.sh push-data data/keywords.txt
./scripts/remote.sh update
./scripts/remote.sh start-compare        # one night
./scripts/remote.sh status               # any time
./scripts/remote.sh results              # next morning: pick the winner
# put the winning teacher into configs/dual-gpu/final-5fold-qwen3-4b.json (commit and push, then 'update')
./scripts/remote.sh start-final
./scripts/remote.sh start-final configs/dual-gpu/final-5fold-qwen3-4b-512.json   # after the first one
ssh $M2V_HOST   # then on the GPU PC: python -m m2vtrain upload models/<name> JanSchachtschabel/<name>
```

Every job is started detached (Windows: through WMI; Linux: `setsid nohup`), so closing the SSH session or
restarting the other computer does not end a run. A run that crashed or was stopped resumes its teacher
embeddings from the cached blocks when it is started again with the same config.

The scripts on the GPU PC can also be called directly there: `scripts/server.sh <action>` (Linux) or
`.\scripts\server.ps1 <action>` (Windows); `python -m m2vtrain status` and `python -m m2vtrain stop` work
anywhere.

## Notes

- `update` pulls the branch that is checked out on the GPU PC. Change configs on the other computer, push them,
  then `update`; or edit them on the GPU PC (then `git pull` may refuse until you commit or reset the change).
- Queue output goes to `runs/queue-<name>.log`, the run's own log to `runs/<run>/run.log`, its state to
  `runs/<run>/status.json` (also shown by `status` and the UI).
- Model cards hide the names of local files and the number of documents by default
  (`card.show_data_facts: false`), so a model trained with the keyword list can be uploaded without describing
  that list. See [data-and-licensing.md](data-and-licensing.md).
