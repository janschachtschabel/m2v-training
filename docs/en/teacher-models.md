# Teacher models

*Deutsch: [../de/lehrermodelle.md](../de/lehrermodelle.md)*

The teacher is the transformer whose knowledge the static model inherits: once through distillation (each
vocabulary entry is passed through the teacher) and, more importantly, through tokenlearn (the static model learns
to reproduce the teacher's sentence embeddings). A stronger German teacher therefore raises the ceiling.

Requirements for the next machine (NVIDIA RTX 5060 Ti, 16 GB):

- **free license** (Apache 2.0 or MIT; no non-commercial or custom model licenses)
- **multilingual with strong German**
- **fits into 16 GB** with room for activations at batch sizes that keep the GPU busy

## Candidates

German retrieval: MIRACL German nDCG@10; overall multilingual: MMTEB mean. Both as reported in
[this comparison](https://wz-it.com/en/blog/best-embedding-models-german/) (MTEB results, September 2026); "–"
means no verified figure.

| Model | License | Parameters | Dim | Weights (bf16/fp16) | MIRACL-de | MMTEB | Distillation pooling | Config |
|---|---|---|---|---|---|---|---|---|
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Apache 2.0 | 4.0B | 2560 | ~8 GB | **62.98** | **69.45** | `last` | `rtx5060ti-05-qwen3-embedding-4b.json` |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | MIT | 568M | 1024 | ~1.1 GB | 57.59 | 59.56 | `mean` | `rtx5060ti-02-bge-m3.json` |
| [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | Apache 2.0 | 568M | 1024 | ~1.1 GB | – | – | `mean` | `rtx5060ti-03-arctic-embed-l-v2.json` |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache 2.0 | 0.6B | 1024 | ~1.2 GB | 54.21 | 64.33 | `last` | `rtx5060ti-04-qwen3-embedding-0.6b.json` |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) (current) | Apache 2.0 | 305M | 768 | ~0.6 GB | – | – | `mean` | `rtx5060ti-01-gte-reference.json` |

### Recommendation

1. **Qwen3-Embedding-4B** is the strongest freely licensed model for German retrieval that fits the card: about
   8 GB of weights in bfloat16 leave room for batches of 16 texts at 512 tokens. It is a decoder model: pooling
   is the last token, distillation uses `pooling: "last"`. It is also the slowest teacher (see run times in
   [gpu-setup.md](gpu-setup.md)), but teacher embeddings are cached and computed once for all folds.
2. **bge-m3** is the robust encoder alternative: much faster, proven with Model2Vec (MinishLab's
   `potion-multilingual-128M` is distilled from it, as was the earlier `JanSchachtschabel/m2v-bge-m3-edu`).
3. **gte-multilingual-base** stays in the comparison as reference, so the effect of the teacher can be separated
   from the effect of the new data basis.

Run 01, 02 and 05 (3-fold, no final model), compare with `python -m m2vtrain compare`, then put the winner into
`rtx5060ti-final-5fold.json` (it uses Qwen3-Embedding-4B) and run that. Add 03 and 04 if time allows. A teacher
that scores best on MTEB does not automatically give the best *static* model, which is why the choice is
measured rather than assumed. Decoder teachers in particular are less established for Model2Vec distillation;
tokenlearn is what transfers their sentence-level quality.

### Not recommended

| Model | Reason |
|---|---|
| Qwen3-Embedding-8B (Apache 2.0) | ~15 GB of weights in bf16 leave no room for activations on 16 GB |
| jina-embeddings-v3 / v5 | CC BY-NC 4.0, not free for commercial use |
| jina-embeddings-v4 | Qwen research license |
| EmbeddingGemma-300m | Gemma terms of use, not an open license |
| EmbeddingGemma 2 (Apache 2.0 since October 2026) | promising and small (270M text part), but multimodal, very new and without German retrieval figures yet; worth a later test |
| granite-embedding-311m-multilingual-r2 (Apache 2.0) | free, but weaker in German (MIRACL-de 50.97) |
| multilingual-e5-large-instruct (MIT) | weaker in German retrieval (MIRACL-de 43.37) and needs prefixes |

## Settings per teacher

| Setting | gte-multilingual | bge-m3 / arctic-l-v2 | Qwen3-Embedding |
|---|---|---|---|
| `teacher.trust_remote_code` | `true` | `false` | `false` |
| `teacher.pooling` (distillation) | `mean` | `mean` | `last` |
| `teacher.dtype` | `float16` | `float16` | `bfloat16` |
| `teacher.batch_size` (16 GB, 512 tokens) | 128 | 64 | 16 (4B), 64 (0.6B) |
| `teacher.document_prefix` | none | none | none (Qwen's instruction is only for queries) |
| `license` of the trained model | `apache-2.0` | `mit` / `apache-2.0` | `apache-2.0` |

`pca_dims` stays 256 in all configs so the models are comparable with the published ones; 512 dimensions are a
cheap follow-up experiment for the winner.

## Memory for teacher embeddings

All teacher embeddings of a run are held in RAM as float32 during training: roughly
`texts x teacher dimension x 4 bytes`. With the synthetic dataset (document text, about three full-text chunks
and three questions per record) plus a keyword list of a few tens of thousands of terms, expect a few hundred
thousand texts: about 1.5 GB for a 1024-dimensional teacher and about 4 GB for Qwen3-Embedding-4B (2560
dimensions), plus a temporary copy during PCA. Reduce `tokenlearn.max_texts` or
`synthetic.max_questions_per_doc` if the machine has little RAM.
