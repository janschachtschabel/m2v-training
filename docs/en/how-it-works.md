# How it works

*Deutsch: [../de/funktionsweise.md](../de/funktionsweise.md)*

A Model2Vec model is a table with one vector per token. To embed a text, the tokenizer splits it into tokens, the
model looks up their vectors and averages them (weighted per token). There is no attention and no layer stack, so
encoding costs almost nothing. The quality comes entirely from the vectors in the table, and this pipeline is
about filling that table well for German educational content.

A run goes through the steps below. With k-fold cross-validation, steps 3 to 8 run once per fold and, if
requested, once more on all data for the final model.

## 1. Load data

| Source | Becomes |
|---|---|
| Synthetic dataset (CC BY 4.0) | one document per record: title, description, keywords, subject, topic. Full texts are split into chunks of whole paragraphs (up to 1,200 characters, Markdown removed). Up to 3 questions per record. |
| Keyword list | plain terms, one per line. They are never documents and never test queries; they only feed the vocabulary, tokenlearn and term → document pairs. |
| Local exports (optional) | documents from `;`-separated CSV metadata exports, cleaned (language, boilerplate, description length, keyword stoplist). |

Details and licensing: [data-and-licensing.md](data-and-licensing.md).

## 2. Split without leakage

Documents that are near-duplicates (same normalised title, or a long identical description) or belong to the
same topic are joined into one group. A group always stays on one side: with a single split, `test_fraction` of
the groups is held out; with k folds, each group gets its fold from a seeded hash, so the partition does not
depend on row order. Without this, a model would be tested on near-copies of its training data and look better
than it is. More in [evaluation.md](evaluation.md).

## 3. Build the vocabulary

The teacher's tokenizer splits rare German words into pieces (for example "Photo", "syn", "these"), and the average of
the pieces is a poor vector for the word. So whole words are added as their own tokens:

- **keywords**: terms of the keyword list and document keywords used by at least `vocab.keyword_min_docs`
  documents; always kept
- **corpus words** used at least `vocab.min_word_count` times
- **noun base forms** of these words ("Brüchen" -> "Bruch"), so lemmatised input finds a token
- **lowercase variants** ("photosynthese"), because the tokenizer is case-sensitive and users type in lowercase

`vocab.max_vocab_size` caps the corpus-derived entries; keywords do not count against it.

## 4. Distil from the teacher

model2vec runs every vocabulary entry through the teacher once (pooling `mean`, or `last` for decoder teachers
such as Qwen3-Embedding), reduces the vectors to `pca_dims` dimensions with PCA and down-weights very frequent
tokens (SIF weighting). The result is already a usable static model, but each token was seen in isolation, never
in context.

## 5. Teacher embeddings of whole texts

The teacher embeds the training texts in full: documents ("title. description. keywords"), full-text chunks,
questions and the keyword terms. These sentence embeddings are the targets for the next step. They are the most
expensive part of a run, so they are computed once for all folds, stored in `cache/teacher/` in blocks of 4,096
texts (an interrupted run resumes from the last finished block) and, with `teacher.devices`, spread over several
GPUs. Before training, the targets are reduced to `pca_dims` with a PCA fitted on the training texts of the fold.

## 6. Tokenlearn

The token vectors are trained so that the average of a text's tokens reproduces the teacher's embedding of that
text (cosine loss, early stopping on a validation part). Before training, all token vectors are rotated into the
basis of the targets (orthogonal Procrustes), because only tokens that occur in the training texts move during
training and the others must not stay in the old basis. Per-token weights are learned as well, so informative
words count more than filler words.

`input_mode` decides what the static model sees: `raw` text, `lemma` text (only noun base forms) or `mixed`
(each text either way, `lemma_share` of them as lemmas). The teacher always sees the raw text. With `mixed` the
model learns to map "Schülerin Bruch" to the same place as "Die Schülerinnen rechnen mit Brüchen". Keyword terms
are also trained in lowercase and base-form spelling (`keyword_lists.lowercase_inputs`, `lemma_inputs`).

This step was the largest single gain in earlier runs ([lessons-learned.md](lessons-learned.md)).

## 7. Pair training

Contrastive training (InfoNCE with in-batch negatives) on pairs that should land close together:

- keyword -> document (document keywords, and keyword-list terms that occur in a document, up to
  `keyword_lists.max_docs_per_keyword` documents per term)
- title -> description
- question -> document (synthetic records)

This sharpens exactly the tasks the models are used for: finding material for a term or a question.

## 8. Evaluate

The model is tested on the held-out documents (or the held-out fold): keyword search, lowercase keyword search,
title -> description, question -> document and subject classification, each with raw and with base-form input,
next to TF-IDF and the comparison models of the config. Documents are embedded without their keywords, so a
keyword search succeeds only if the model recognises the concept in the text. Metrics and their limits:
[evaluation.md](evaluation.md).

## 9. Save and publish

The model is stored in `float16` (or `int8`; `vocabulary_quantization` can shrink the table further) together
with a model card (`README.md`), `training_config.json` and `eval_results.json`. By default these files contain no
local file names and no document or vocabulary counts (`card.show_data_facts`), and the card names the CC BY 4.0
source of the synthetic data. `python -m m2vtrain upload` publishes it, private by default.

## What a run writes

| Path | Contents |
|---|---|
| `runs/<run>/config.json` | the complete config; `run runs/<run>/config.json` repeats the run |
| `runs/<run>/status.json` | state, current stage, progress, heartbeat (read by `status`, the UI and the scripts) |
| `runs/<run>/run.log` | the run's log |
| `runs/<run>/eval.json` | all metrics, baselines, split fingerprint; with k folds also mean ± std and `folds/fold-<i>/eval.json` |
| `models/<name>/` | the trained model with model card, ready for `upload` (single split or final model) |
| `cache/teacher/` | cached teacher embeddings, shared by all runs with the same teacher and texts |

## Why this design

- **Why not a transformer at inference?** It is far more accurate per query but needs a GPU or a lot of CPU time
  for large collections and search-as-you-type. The static model gives up some quality for speed and simplicity;
  the evaluation shows how much.
- **Why not just TF-IDF?** TF-IDF matches strings. It does not know that "Bruchrechnung" and "Brüche addieren"
  belong together. It is still a strong baseline for keyword search, which is why every run reports it.
- **Why distil and then train, instead of only distilling?** Distillation sees each token alone. Tokenlearn and
  pair training teach the vectors how words behave in real texts and queries.
- **Why synthetic data?** It can be published and documented, and its questions and full texts add training
  signal that metadata alone does not have. Real metadata stays useful for private experiments.
