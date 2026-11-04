# Data basis and licensing

*Deutsch: [../de/daten-und-lizenzen.md](../de/daten-und-lizenzen.md)*

The training setup can mix three kinds of sources. For a model whose training data can be documented and
published, use only the first two.

## 1. Synthetic dataset (CC BY 4.0)

[`JanSchachtschabel/synthetic-german-educational-materials-50k`](https://huggingface.co/datasets/JanSchachtschabel/synthetic-german-educational-materials-50k)
holds 50,000 German educational materials written by a language model for all education sectors (school, higher
education, vocational training, adult education, early childhood). None of the texts is copied from existing
materials. Per record the pipeline uses:

| Field | Use |
|---|---|
| `title`, `description` | document text (tokenlearn, title -> description pairs, evaluation) |
| `keywords` | vocabulary, keyword -> document pairs, keyword search evaluation |
| `fulltext` (Markdown) | stripped of Markdown, cut into chunks of about 1,200 characters, tokenlearn texts and vocabulary |
| `qa_pairs.question` | tokenlearn texts and question -> document pairs (up to 3 per record by default); the evaluation uses the questions of held-out records |
| `qa_pairs.answer` | only with `synthetic.use_answers: true` |
| `subject` | subject classification probe in the evaluation |
| `topic_uri` / `topic` | grouping: records about the same topic always land in the same fold |

The dataset's own train/validation/test split is not used: `m2vtrain` reads all splits and separates them
itself, grouped by topic, so k-fold cross-validation can use every record for training once and for testing once.

The dataset is loaded from the Hub (`synthetic.dataset`, optionally pinned with `synthetic.revision`) or from a
local folder or parquet file. **Attribution:** CC BY 4.0 requires naming the source; the generated model card
links the dataset and its license automatically.

Limitations to keep in mind: the texts are synthetic, not reviewed by experts, written by one model with one
prompt set, and describe the German education system. Evaluation numbers on this data measure how well a model
handles this kind of text, not real user queries.

## 2. Keyword lists (terms only)

A keyword list is a plain text file with one term per line (optionally `term<TAB>count`; lines starting with `#`
are comments). It contains **no titles, descriptions, identifiers, URLs or links to resources**, so nothing in it
points back to individual records.

Create it locally from a metadata export (for example an edu-sharing/WLO CSV export):

```bash
python -m m2vtrain extract-keywords data/export.csv.gz data/keywords.txt --min-docs 3
```

- Keywords are cleaned like in training (stoplist of generic platform/format tags, length limits, no URLs or
  pure numbers) and grouped case-insensitively under their most frequent spelling.
- `--min-docs 3` keeps only terms used by at least three documents, which drops typos and one-off tags.
- Without `--with-counts` the file holds only the terms and reveals nothing about the size of the source data.

How the terms are used in training (`keyword_lists.*`):

| Option | Default | Effect |
|---|---|---|
| `add_to_vocabulary` | on | every term (and its lowercase form) becomes a token, so it is not split into subwords |
| `tokenlearn` | on | the static model learns the teacher embedding of each term |
| `lowercase_inputs`, `lemma_inputs` | on | the lowercase spelling and the noun base form are trained towards the same target, which makes lowercase and inflected queries work |
| `pairs` | on | term -> documents of the corpus that mention it as whole words (up to `max_docs_per_keyword`) |

The terms are training data of every fold; they are never used as evaluation queries. The evaluation's keyword
queries are the keywords of the held-out synthetic records.

## 3. Local exports (optional, private)

`data_files` still accepts `;`-separated CSV exports (also `.csv.gz`) with `properties.cclom:*` columns. Use them
only for private experiments. The repository never contains them (`data/` is git-ignored), the published
`training_config.json` replaces their names with `<local file>`, and the model card only says "further German
educational texts", without origin, size or kind of data.

## What never goes into the repository

- data files, keyword lists, run folders, models and teacher caches (`data/`, `runs/`, `models/`, `cache/` are
  git-ignored)
- Hugging Face tokens: log in with `hf auth login` or set `HF_TOKEN` in the environment, never in a config
- document counts or descriptions of private data in docs, model cards or commit messages

## Model cards

By default the card names only openly licensed sources and shows no document counts (`card.show_data_facts`
switches counts on, for models trained on public data only). `card.data_description` replaces the automatic text
entirely. `eval_results.json` next to the model omits run facts and the test split size unless
`show_data_facts` is on.

## Before publishing a model

1. Check the license of the teacher (see [teacher-models.md](teacher-models.md)) and set `license` accordingly.
2. Keep the CC BY 4.0 attribution of the synthetic dataset in the card.
3. Check the terms of the metadata source the keyword list was extracted from. Single keywords and short
   technical terms are generally not protected as works, but database rights or terms of use of the source can
   still apply to systematic extracts. Publishing the model does not publish the list itself; publishing the list
   would need this check. (This is a practical note, not legal advice.)
4. Read the generated card once: it must not mention local file names, counts or the nature of private data.
