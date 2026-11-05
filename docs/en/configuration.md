# Configuration reference

*Deutsch: [../de/konfiguration.md](../de/konfiguration.md)*

A run config is a JSON file. Only `name`, a data source (`synthetic.enabled: true` and/or `data_files`) and the
values that differ from the defaults are needed; `python -m m2vtrain init-config` writes a complete file. Every run
saves its full config to `runs/<run>/config.json`, so `python -m m2vtrain run runs/<run>/config.json` repeats it.
Unknown keys and wrong types are rejected with an error message.

## Data

| Key | Default | Meaning |
|---|---|---|
| `name` | – | run and model folder name, suggested Hub name |
| `data_files` | `[]` | local exports (`;`-separated CSV, `.csv.gz`); glob patterns allowed; optional with the synthetic dataset |
| `synthetic.enabled` | `false` | use the synthetic dataset |
| `synthetic.dataset` | `JanSchachtschabel/synthetic-german-educational-materials-50k` | Hub id, local folder or parquet file |
| `synthetic.revision` | `null` | pin a Hub revision (commit or tag) for reproducibility |
| `synthetic.splits` | all three | which of the dataset's splits to read (separation is done by `m2vtrain`) |
| `synthetic.use_fulltext`, `fulltext_chunk_chars` | `true`, 1200 | full texts as tokenlearn texts and vocabulary, in chunks of whole paragraphs |
| `synthetic.use_questions`, `max_questions_per_doc` | `true`, 3 | questions as tokenlearn texts and question -> document pairs |
| `synthetic.use_answers` | `false` | answers as extra tokenlearn texts |
| `synthetic.max_documents` | `null` | sample a subset (smoke tests) |
| `keyword_lists.files` | `[]` | keyword lists, one term per line |
| `keyword_lists.min_count`, `max_keywords` | 1, `null` | filter by the count column, cap the number of terms |
| `keyword_lists.add_to_vocabulary`, `tokenlearn`, `lowercase_inputs`, `lemma_inputs`, `pairs` | all `true` | how the terms are used, see [data-and-licensing.md](data-and-licensing.md) |
| `keyword_lists.max_docs_per_keyword` | 3 | documents per term for term -> document pairs |
| `data.languages` | `["de"]` | languages kept from local exports |
| `data.*` | | further cleaning of local exports (language detection, description length, boilerplate, stoplist) |

## Split and evaluation

| Key | Default | Meaning |
|---|---|---|
| `test_fraction` | 0.05 | held-out share for a single split |
| `cross_validation.folds` | 0 | 0: single split; 2–20: k-fold |
| `cross_validation.final_model` | `true` | after the folds, train and save a model on all documents |
| `cross_validation.shared_vocabulary` | `false` | one vocabulary and distillation for all folds (faster, slightly optimistic) |
| `seed` | 42 | split, sampling and training seed |
| `eval.*` | | numbers of queries, probe options |
| `eval_modes` | `["raw", "lemma"]` | input modes of the evaluation |
| `baseline_models` | `[]` | Model2Vec or Sentence Transformers models evaluated next to the new one |
| `evaluate_teacher` | `false` | also evaluate the teacher (slow) |

## Teacher

| Key | Default | Meaning |
|---|---|---|
| `teacher.model_name` | `Alibaba-NLP/gte-multilingual-base` | Hub id or local path |
| `teacher.trust_remote_code` | `true` for gte | only for models with their own code |
| `teacher.pooling` | `mean` | pooling for distillation: `mean`, `last` (decoders such as Qwen3), `first` (CLS), `pooler` |
| `teacher.document_prefix` | `""` | text put before every document, e.g. `"passage: "` for e5 |
| `teacher.batch_size` | 64 | texts per teacher step |
| `teacher.max_seq_length` | `null` | cap the teacher input in tokens (memory grows quadratically with length) |
| `teacher.dtype` | `auto` | `float16`/`bfloat16` on a GPU halve memory and time; the CPU always uses float32 |
| `teacher.device` | `null` (auto) | `cuda`, `cuda:0`, `cpu`, `mps` |
| `teacher.devices` | `[]` | several GPUs for the teacher embeddings, e.g. `["cuda:0", "cuda:1"]`: one teacher copy per GPU, texts split evenly; does not change the cache key ([dual-gpu.md](dual-gpu.md)) |

Changing `max_seq_length` or `dtype` changes the cache key of the teacher embeddings (default values keep older
caches valid).

## Model and training

| Key | Default | Meaning |
|---|---|---|
| `pca_dims` | 256 | dimensions of the static model |
| `vocab.max_vocab_size` | 200000 | cap for added vocabulary (keywords are always kept) |
| `vocab.min_word_count`, `keyword_min_docs` | 5, 3 | minimum frequency of corpus words and document keywords |
| `vocab.add_lemmas`, `add_lowercase_variants`, `include_keywords` | `true` | which entries are added |
| `input_mode`, `lemma_share` | `mixed`, 0.5 | static model input during training: `raw`, `lemma` or a mix |
| `tokenlearn.enabled`, `max_texts` | `true`, `null` | tokenlearn on (all) texts |
| `tokenlearn.train.*` | lr 1e-3, early stopping 5 | `max_epochs`, `learning_rate`, `batch_size`, `device` |
| `pairs.enabled`, `max_pairs` | `false`, 500000 | contrastive pair training and its cap (`null`: all pairs) |
| `pairs.train.*` | lr 1e-4, early stopping 5 | as above |
| `quantize_to` | `float16` | stored precision: `float32`, `float16`, `int8` |
| `vocabulary_quantization` | `null` | cluster the vocabulary to this many vectors (smaller model) |

## Model card and paths

| Key | Default | Meaning |
|---|---|---|
| `license` | `apache-2.0` | license in the card; match the teacher |
| `card.data_description` | `""` | replaces the automatic text about the training data |
| `card.show_data_facts` | `false` | show document and vocabulary counts (only for public data) |
| `runs_dir`, `models_dir`, `cache_dir` | `runs`, `models`, `cache` | output folders |

## Ready-made configs

| File | Purpose |
|---|---|
| `configs/smoke-test-cpu.json` | small 3-fold run on the CPU, checks the whole pipeline |
| `configs/rtx5060ti-01-gte-reference.json` | current teacher on the new data basis, 3-fold |
| `configs/rtx5060ti-02-bge-m3.json` | bge-m3, 3-fold |
| `configs/rtx5060ti-03-arctic-embed-l-v2.json` | arctic-embed-l-v2.0, 3-fold |
| `configs/rtx5060ti-04-qwen3-embedding-0.6b.json` | Qwen3-Embedding-0.6B, 3-fold |
| `configs/rtx5060ti-05-qwen3-embedding-4b.json` | Qwen3-Embedding-4B, 3-fold |
| `configs/rtx5060ti-final-5fold.json` | 5-fold estimate plus final model on all data (Qwen3-Embedding-4B; swap in the winner) |
| `configs/dual-gpu/final-5fold-qwen3-4b.json` | the same, teacher embeddings on both GPUs of a two-card PC |
| `configs/dual-gpu/final-5fold-qwen3-4b-512.json` | the same with 512 dimensions (reuses the cached teacher embeddings of the 256 run) |

All of them expect `data/keywords.txt`; remove `keyword_lists` to train on the synthetic dataset alone.
