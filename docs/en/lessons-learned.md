# Lessons learned from earlier runs

*Deutsch: [../de/erfahrungen.md](../de/erfahrungen.md)*

Findings from the runs that produced
[`JanSchachtschabel/m2v-gte-256-edu-v2`](https://huggingface.co/JanSchachtschabel/m2v-gte-256-edu-v2)
(teacher gte-multilingual-base, 256 dimensions, trained on a CPU laptop). The numbers come from its public model
card and refer to that model's own test set; they are **not comparable** with results on the synthetic dataset.

## What helped most

- **Tokenlearn plus pair training** was the big lever. Against its predecessor (distillation only, same teacher
  and dimensions) keyword search nDCG@10 rose from 0.084 to 0.323, title -> description MRR@10 from 0.344 to
  0.648 and subject macro-F1 from 0.372 to 0.500. The model also beat TF-IDF (0.220 / 0.577 / 0.435),
  potion-multilingual-128M and the transformer paraphrase-multilingual-MiniLM-L12-v2.
- **Noun base forms for lowercase queries.** With lowercase keywords, nDCG@10 was 0.212 on raw input and 0.291
  when queries and documents were reduced to noun lemmas. For well-formed text raw input stays better. The
  vocabulary therefore holds inflected forms, base forms and lowercase variants, and training mixes raw and lemma
  input (`input_mode: mixed`), so the model works both ways.
- **Case matters.** XLM-R-style tokenizers are case-sensitive: "photosynthese" and "Photosynthese" are different
  tokens. Lowercase variants in the vocabulary and the new `keyword_lists.lowercase_inputs` address this.

## Training behaviour

- Tokenlearn stopped early after about 23 epochs; pair training kept improving slowly but was flat after about
  35 epochs, so the GPU configs cap it at 35 (`pairs.train.max_epochs`).
- The teacher embeddings are the expensive step. On the laptop CPU the gte teacher managed about 7–10 texts per
  second; a full run took most of a day. A GPU changes this by one to two orders of magnitude.
- A few very long input texts made the teacher run out of memory
  at batch size 64. `teacher.max_seq_length` now caps the input; 512 tokens are set in the GPU configs.

## Technical pitfalls (already handled in the code)

- `gte-multilingual-base` under transformers 5: its remote code computes position ids and rotary buffers in
  `__init__`; transformers 5 leaves them uninitialized (IndexError or silently wrong embeddings).
  `m2vtrain.targets.restore_computed_buffers` fixes this.
- `StaticModel.push_to_hub` replaces the model card with a generic one. Use `python -m m2vtrain upload`.
- Runs started from an IDE terminal or a remote session died when that parent process was restarted. Start long
  runs detached (`scripts/run_detached.ps1`, `nohup`).
- An interrupted run resumes its teacher embeddings from the cached chunks.
- Results are only comparable with the same split fingerprint (`test_split.fingerprint` in `eval.json`).

## Ideas not yet tried

- a larger teacher (this repository's RTX 5060 Ti configs)
- 512 dimensions for the best teacher
- BM25 + static embeddings as hybrid search
- hard negatives in pair training
- a small set of real user queries with judged results as an extra test set
- more openly licensed text for tokenlearn (e.g. German Wikipedia) and pairs (e.g. GND subject headings)
