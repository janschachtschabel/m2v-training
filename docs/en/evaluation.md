# Evaluation and k-fold cross-validation

*Deutsch: [../de/evaluierung.md](../de/evaluierung.md)*

## Metrics

Every metric is computed twice: with **raw** input and with **lemma** input (queries and documents reduced to
noun base forms, e.g. "Die Schülerinnen rechnen mit Brüchen" -> "Schülerin Bruch"). Documents are embedded as
"title. description", **without** their keywords, so the model has to recognise the concept in the text.

| Metric | Task |
|---|---|
| `keyword_ndcg@10`, `keyword_recall@10` | each keyword of the test documents (used by at least 2 of them) is a query; relevant are the test documents tagged with it |
| `keyword_lower_*` | the same with lowercase queries, as users type them |
| `title_mrr@10`, `title_recall@10` | find a document's description from its title (ambiguous titles/descriptions are left out) |
| `question_mrr@10`, `question_recall@10` | find a document from one of its questions (synthetic records only; one seeded question per record) |
| `subject_macro_f1` | logistic-regression probe on frozen embeddings, trained on the training part, single-subject documents |

Each run also scores **TF-IDF** (fitted on the training documents, per input mode) and the models listed in
`baseline_models` (Model2Vec models or Sentence Transformers), and optionally the teacher (`evaluate_teacher`).

## Single split

`cross_validation.folds: 0` (default) holds out `test_fraction` of the documents. Near-duplicates (same
normalized title or a long enough identical description) and records with the same `group` (the topic of a
synthetic record) always land on the same side.

## k-fold cross-validation

```json
"cross_validation": {"folds": 5, "final_model": true, "shared_vocabulary": false}
```

1. Every document gets a fold number from a seeded hash of its group, so the partition is deterministic,
   independent of row order, and keeps topics and near-duplicates together.
2. The tokenlearn texts of all documents and the keyword-list terms are encoded by the teacher **once**
   (cached in `cache/teacher/`).
3. For each fold *i*: vocabulary and distillation from the other folds' documents, tokenlearn on their texts and
   the keyword terms (PCA of the targets fitted on exactly these texts), pair training on their pairs, then the
   evaluation on fold *i*. Results: `runs/<run>/folds/fold-<i>/eval.json`.
4. `runs/<run>/eval.json` holds the mean per encoder, input mode and metric (same layout as a single
   evaluation, so the UI and `compare` work unchanged), plus `cv.std` and every fold's value in `cv.values`.
5. With `final_model: true` a last model is trained on **all** documents and saved; its card shows the
   cross-validated metrics as mean ± standard deviation. This is the usual "estimate with CV, refit on all data"
   procedure: the published number describes the training recipe; the final model has seen more data and is
   expected to be at least as good.

Every document is tested exactly once and used for training k-1 times. With k=5 each model trains on 80 % of the
data, so the estimate is close to the final model.

`shared_vocabulary: true` builds the vocabulary from all documents and distills only once. That saves (k-1)
distillations (minutes on a GPU, about half an hour each on a CPU with a 250k vocabulary) but is slightly
optimistic: words and keywords of the test fold exist as tokens, even though their vectors are trained only on
other folds' texts. Use it for quick experiments (the CPU smoke test does), not for the numbers you publish.

What stays outside the folds:

- **Keyword-list terms** are training data in every fold and never queries. That is intended: the final model
  knows them too. The keyword search evaluation uses the keywords of held-out synthetic records; if a term is
  also in the keyword list, the model has learned what the term means from the teacher, not which documents carry
  it.
- **Teacher embeddings** are computed for all texts at once; they depend only on the text, not on the split.

## Comparing runs

```bash
python -m m2vtrain compare                 # all runs in runs/
python -m m2vtrain compare runs/a runs/b --mode lemma
```

Only rows with the same **split fingerprint** are directly comparable: same data, `seed`, `test_fraction` or
number of folds. For cross-validated runs, differences smaller than about twice the standard deviation should
be treated as noise. The UI's "Vergleich" page shows the same means.

## What the numbers do and do not say

The synthetic records are clean, consistent and written by one model. Scores on them are higher and the spread
smaller than on real, noisy metadata; they rank teachers and recipes against each other but do not predict the
quality on real user queries exactly. A small set of real queries with judged results (even 100) would be the
most valuable addition to this evaluation.
