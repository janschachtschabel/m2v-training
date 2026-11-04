# GPU setup (NVIDIA RTX 5060 Ti, 16 GB)

*Deutsch: [../de/gpu-einrichtung.md](../de/gpu-einrichtung.md)*

## Installation

The RTX 50 series (Blackwell, compute capability 12.0 / `sm_120`) needs a recent NVIDIA driver and a PyTorch
build for **CUDA 12.8 or newer**. Older builds load but fail with "no kernel image is available".

Windows (PowerShell, Python 3.12 installed):

```powershell
git clone https://github.com/janschachtschabel/m2v-training
cd m2v-training
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, if scripts are blocked
.\scripts\setup_windows.ps1                          # venv, torch (cu128), package, GPU check
```

Linux:

```bash
./scripts/setup_linux.sh python3.12 https://download.pytorch.org/whl/cu128
```

`scripts/check_gpu.py` prints the GPU, its memory and compute capability, and tests a bfloat16 matrix
multiplication. If it reports a missing `sm_120` kernel, install torch from a newer CUDA index (for example
`cu129` or `cu130`).

Hugging Face: `hf auth login` once (needed for uploads; downloads of public models and the dataset work without).
Copy your keyword list to `data/keywords.txt` on the new machine; it is not in the repository.

## Running

```bash
python -m m2vtrain run configs/smoke-test-cpu.json            # functional check, minutes
python -m m2vtrain run configs/rtx5060ti-01-gte-reference.json
python -m m2vtrain run configs/rtx5060ti-02-bge-m3.json
python -m m2vtrain run configs/rtx5060ti-05-qwen3-embedding-4b.json
python -m m2vtrain compare
# put the winning teacher into configs/rtx5060ti-final-5fold.json, then:
python -m m2vtrain run configs/rtx5060ti-final-5fold.json
python -m m2vtrain upload models/<run> JanSchachtschabel/<name>   # private by default
```

Long runs should not be children of an IDE terminal or remote session: closing or restarting that parent ends
the run. Use

- Windows: `.\scripts\run_detached.ps1 configs\rtx5060ti-02-bge-m3.json` (starts through WMI, log in
  `runs\<config>-console.log`), or `.\scripts\run_queue.ps1 configs\a.json configs\b.json` for several configs
- Linux: `nohup ./scripts/run_queue.sh configs/rtx5060ti-0*.json > runs/queue.log 2>&1 &`

A stopped or crashed run resumes its teacher embeddings where it stopped (chunks of 4,096 texts in
`cache/teacher/`) when it is started again with the same config.

## What uses the GPU

| Step | GPU | Notes |
|---|---|---|
| distillation | yes (`teacher.device`) | one forward pass per vocabulary entry; minutes |
| teacher embeddings | yes | the longest step for large teachers; cached, once for all folds |
| tokenlearn, pair training | yes (`tokenlearn.train.device`, `pairs.train.device`) | the static model is small; data loading dominates |
| evaluation | partly | Model2Vec models and TF-IDF run on the CPU in seconds; transformer baselines use the GPU |

## Rough run times

These are estimates for planning, not measurements on this card; the first runs will tell. On Jan's laptop CPU
the gte teacher encoded about 7–10 texts per second. On a 16 GB Blackwell card, expect roughly:

| Teacher | Teacher embeddings (a few hundred thousand texts, ≤512 tokens) | per fold: distill + tokenlearn + pairs |
|---|---|---|
| gte-multilingual-base (fp16) | well under an hour | under an hour |
| bge-m3 / arctic-l-v2 (fp16) | about an hour | under an hour |
| Qwen3-Embedding-4B (bf16, batch 16) | several hours | about an hour |

A 3-fold teacher comparison therefore fits into a night per teacher; the final 5-fold run with Qwen3-4B into one
to two days. If memory runs out, lower `teacher.batch_size` first, then `teacher.max_seq_length`.

## Memory

- GPU: Qwen3-Embedding-4B needs about 8 GB for weights in bf16; batch 16 at 512 tokens fits into the rest.
  All other candidates use far less.
- RAM: teacher embeddings are held in float32 (see [teacher-models.md](teacher-models.md#memory-for-teacher-embeddings)):
  about 1.5 GB for 1024-dimensional teachers, about 4 GB for Qwen3-4B, plus a temporary copy during PCA. 32 GB of
  RAM are comfortable, 16 GB work with `tokenlearn.max_texts` or fewer questions per record.
