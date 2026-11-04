# m2v-training

Train fast, static [Model2Vec](https://github.com/MinishLab/model2vec) embedding models for **German education
content** that recognise nouns and keywords well, and evaluate them cleanly.

*Deutsche Fassung: [README.de.md](README.de.md)*

A Model2Vec model needs no GPU and no transformer at inference time: a text is embedded by looking up and
averaging token vectors. This repository contains the complete training setup: data preparation, vocabulary,
distillation from a transformer teacher, [Tokenlearn](https://github.com/MinishLab/tokenlearn)-style
fine-tuning, contrastive pair training, a domain evaluation (also as k-fold cross-validation), model cards and
Hugging Face upload, plus a small Streamlit UI.

## What a run does

1. **Load data**: the openly licensed synthetic dataset, plain keyword lists and (optionally) local metadata
   exports. See [Data basis](#data-basis).
2. **Split** leakage-free: near-duplicates and records of the same topic always stay on one side. Either one
   held-out test split or **k folds**.
3. **Vocabulary**: corpus words, their noun base forms (lemmas), lowercase variants and keywords are added to the
   teacher's tokenizer, so the model works on raw text and on noun/base-form input.
4. **Distill** a static model from the teacher (default `Alibaba-NLP/gte-multilingual-base`).
5. **Tokenlearn**: train token vectors so that averaged texts reproduce the teacher's sentence embeddings
   (documents, full-text chunks, questions, keyword terms and their lowercase/base-form spellings).
6. **Pair training** (InfoNCE): keyword -> document, title -> description, question -> document.
7. **Evaluate** and **save** with a model card (keyword search nDCG@10, title -> description MRR@10,
   question -> document MRR@10, subject classification macro-F1; raw and noun/base-form input; TF-IDF and other
   models for comparison).

## Quick start

```bash
git clone https://github.com/janschachtschabel/m2v-training
cd m2v-training
./scripts/setup_linux.sh            # Windows: .\scripts\setup_windows.ps1  (installs PyTorch for CUDA 12.8+)
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1

# keyword list (terms only) from a local metadata export, optional but recommended
python -m m2vtrain extract-keywords path/to/export.csv.gz data/keywords.txt --min-docs 3

# a few minutes on a CPU: checks that everything works end to end
python -m m2vtrain run configs/smoke-test-cpu.json

# teacher comparison on the GPU, then the final 5-fold run with the winner
python -m m2vtrain run configs/rtx5060ti-02-bge-m3.json
python -m m2vtrain compare
```

The UI does the same with forms: `streamlit run ui/app.py` (start it in the repository folder; it only listens
on localhost).

## Data basis

The goal is a training basis that can be **documented and published without violating anyone's rights**:

| Source | What is used | License / status |
|---|---|---|
| [JanSchachtschabel/synthetic-german-educational-materials-50k](https://huggingface.co/datasets/JanSchachtschabel/synthetic-german-educational-materials-50k) | titles, descriptions, keywords, full texts, questions, subjects, topics | CC BY 4.0, LLM-written texts |
| Keyword list (`data/keywords.txt`) | terms only, one per line; no titles, descriptions, ids or links | extracted locally with `extract-keywords`; never committed |
| Local exports (`data_files`) | optional, for private experiments | never committed; never described in model cards |

Models trained only on the first two sources can be described fully. Details, the reasoning and what to check
before publishing: [docs/en/data-and-licensing.md](docs/en/data-and-licensing.md).

## Teacher models for an RTX 5060 Ti (16 GB)

| Teacher | License | Size | German retrieval (MIRACL-de nDCG@10) | Config |
|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-4B` | Apache 2.0 | 4.0B, ~8 GB in bf16 | 62.98 | `rtx5060ti-05-qwen3-embedding-4b.json` |
| `BAAI/bge-m3` | MIT | 568M | 57.59 | `rtx5060ti-02-bge-m3.json` |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | Apache 2.0 | 568M | – | `rtx5060ti-03-arctic-embed-l-v2.json` |
| `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0 | 0.6B | 54.21 | `rtx5060ti-04-qwen3-embedding-0.6b.json` |
| `Alibaba-NLP/gte-multilingual-base` (current) | Apache 2.0 | 305M | – | `rtx5060ti-01-gte-reference.json` |

Recommendation: compare gte (reference), bge-m3 and Qwen3-Embedding-4B with 3-fold cross-validation, then run
`rtx5060ti-final-5fold.json` with the winner. Reasoning, sources and what does not fit or is not freely licensed:
[docs/en/teacher-models.md](docs/en/teacher-models.md). GPU setup and run times:
[docs/en/gpu-setup.md](docs/en/gpu-setup.md).

## Evaluation and k-fold cross-validation

With `"cross_validation": {"folds": 5}` the documents are split into 5 folds (grouped by topic and
near-duplicates). Five models are trained, each tested on the fold it never saw; the metrics are averaged and
reported as mean ± standard deviation. With `"final_model": true` a last model is trained on **all** documents and
published with the cross-validated metrics. Teacher embeddings are computed only once for all folds.
Details: [docs/en/evaluation.md](docs/en/evaluation.md).

## Command line

```bash
python -m m2vtrain init-config my.json --synthetic --keywords data/keywords.txt --folds 5
python -m m2vtrain run my.json                      # run folder: runs/<timestamp>-<name>
python -m m2vtrain run runs/<run>/config.json       # repeat a run with the same options
python -m m2vtrain compare [runs/a runs/b] [--mode lemma]
python -m m2vtrain extract-keywords export.csv.gz data/keywords.txt [--min-docs 3] [--with-counts]
python -m m2vtrain upload models/<run> <user>/<model> [--public]
```

A config needs only `name`, a data source and the values that differ from the defaults; unknown keys and wrong
types are rejected. All options: [docs/en/configuration.md](docs/en/configuration.md).

## Repository layout

```
m2vtrain/          Python package (data, sources, split, vocab, targets, train, evaluate, pipeline, export, cli)
ui/                Streamlit UI (German)
configs/           ready-made run configs (CPU smoke test, RTX 5060 Ti teacher comparison, final run)
scripts/           setup (Windows/Linux), GPU check, detached runs, run queues
docs/en, docs/de   documentation in English and German
tests/             offline tests with a tiny local teacher (no downloads)
```

At runtime the folders `data/`, `runs/`, `models/` and `cache/` are created in the repository folder (or under
`M2V_HOME`); they are git-ignored.

## Using a trained model

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("JanSchachtschabel/m2v-gte-256-edu-v2")
embeddings = model.encode(["Bruchrechnung für die 6. Klasse", "Photosynthese"])
```

For lowercase search queries, reduce queries and documents to noun base forms first:

```python
from m2vtrain.lemma import to_lemma_text
model.encode([to_lemma_text("die schülerinnen rechnen mit brüchen")])   # -> "Schülerin Bruch"
```

## Further documentation

- [Data basis and licensing](docs/en/data-and-licensing.md)
- [Teacher models](docs/en/teacher-models.md)
- [Evaluation and k-fold](docs/en/evaluation.md)
- [GPU setup (RTX 5060 Ti)](docs/en/gpu-setup.md)
- [Configuration reference](docs/en/configuration.md)
- [Lessons learned from earlier runs](docs/en/lessons-learned.md)

## Tests

```bash
python -m pytest -q
```

The tests need no internet; they build a tiny random teacher and a tiny synthetic dataset locally.

## License

Code: Apache 2.0 (see [LICENSE](LICENSE)). Trained models inherit obligations from their teacher and data:
set `license` in the config accordingly (for example `apache-2.0` for Apache teachers, `mit` for bge-m3) and keep
the CC BY 4.0 attribution of the synthetic dataset, which the generated model card includes.
