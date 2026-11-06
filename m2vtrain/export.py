"""Model card for a finished model, in the Hugging Face Hub format (YAML front matter + Markdown)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from m2vtrain.config import RunConfig
from m2vtrain.sources import SYNTHETIC_DATASET

MODEL_ID_PLACEHOLDER = "<user>/<model>"
_HUB_NAME = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
# Metrics shown in the card's tables, in this order; eval.json keeps everything.
CARD_METRICS = (
    "keyword_ndcg@10",
    "keyword_recall@10",
    "keyword_lower_ndcg@10",
    "title_mrr@10",
    "title_recall@10",
    "question_mrr@10",
    "subject_macro_f1",
)
PRIVATE_FILE = "<local file>"


def _is_hub_name(name: str) -> bool:
    """'org/model' names can be linked as base_model; local paths (also Windows ones) cannot."""
    return bool(_HUB_NAME.fullmatch(name)) and not Path(name).exists()


def _public_name(name: str) -> str:
    """Hub names as they are, local paths reduced to their last part (no local paths in published files)."""
    return name if _is_hub_name(name) else Path(name.replace("\\", "/")).name


def public_config(config: RunConfig) -> dict:
    """The run config without local paths or names of local data files, for publishing next to the model."""
    raw = config.to_dict()
    for key in ("runs_dir", "models_dir", "cache_dir"):
        raw.pop(key)
    raw["data_files"] = [PRIVATE_FILE for _ in config.data_files]
    raw["keyword_lists"]["files"] = [PRIVATE_FILE for _ in config.keyword_lists.files]
    if not _is_hub_name(config.synthetic.dataset):
        raw["synthetic"]["dataset"] = _public_name(config.synthetic.dataset)
    raw["baseline_models"] = [_public_name(name) for name in config.baseline_models]
    raw["teacher"]["model_name"] = _public_name(config.teacher.model_name)
    if config.data.stoplist_path is not None:
        raw["data"]["stoplist_path"] = _public_name(str(config.data.stoplist_path))
    return raw


def public_results(results: Mapping, show_data_facts: bool = False) -> dict:
    """Results for publishing: baseline names without local paths; without `show_data_facts` no document counts."""
    public = {**results, "baselines": {_public_name(name): by_mode for name, by_mode in results.get("baselines", {}).items()}}
    if "cv" in public:
        cv = public["cv"]
        public["cv"] = {
            **cv,
            "std": {**cv["std"], "baselines": {_public_name(n): v for n, v in cv["std"].get("baselines", {}).items()}},
            "values": {**cv["values"], "baselines": {_public_name(n): v for n, v in cv["values"].get("baselines", {}).items()}},
        }
    if not show_data_facts:
        public.pop("facts", None)
        if "test_split" in public:
            public["test_split"] = {"fingerprint": public["test_split"].get("fingerprint", "")}
    return public


def data_description(config: RunConfig) -> str:
    """The card's text about the training data: only openly licensed sources are named, local exports never."""
    if config.card.data_description:
        return config.card.data_description
    parts = []
    if config.synthetic.enabled:
        name = config.synthetic.dataset
        if name == SYNTHETIC_DATASET:
            parts.append(
                f"the synthetic dataset [{name}](https://huggingface.co/datasets/{name}) "
                "(CC BY 4.0; LLM-written German education texts)"
            )
        else:
            parts.append(f"the synthetic dataset `{_public_name(name)}`")
    if config.keyword_lists.files:
        parts.append("a list of German education keywords (terms only, no documents)")
    if config.data_files:
        parts.append("further German educational texts")
    if not parts:
        return ""
    return "Training data: " + "; ".join(parts) + "."


def _front_matter(config: RunConfig) -> str:
    from huggingface_hub import ModelCardData

    datasets = [config.synthetic.dataset] if config.synthetic.enabled and _is_hub_name(config.synthetic.dataset) else None
    data = ModelCardData(
        library_name="model2vec",
        license=config.license,
        language=list(config.data.languages),
        datasets=datasets,
        base_model=config.teacher.model_name if _is_hub_name(config.teacher.model_name) else None,
        tags=["embeddings", "static-embeddings", "sentence-transformers", "model2vec", "education"],
        pipeline_tag="sentence-similarity",
    )
    return f"---\n{data.to_yaml()}\n---"


def _format(value, std=None) -> str:
    if isinstance(value, float):
        return f"{value:.3f} ± {std:.3f}" if isinstance(std, float) else f"{value:.3f}"
    return "–" if value is None else str(value)


def metrics_table(results: Mapping) -> str:
    """One row per (encoder, input mode); columns are the CARD_METRICS the results have. Cross-validated results
    show mean ± standard deviation over the folds."""
    std = results.get("cv", {}).get("std", {})
    rows = [("this model", mode, metrics, std.get("model", {}).get(mode, {})) for mode, metrics in results.get("model", {}).items()]
    for name, by_mode in results.get("baselines", {}).items():
        rows += [(name, mode, metrics, std.get("baselines", {}).get(name, {}).get(mode, {})) for mode, metrics in by_mode.items()]
    present = [metric for metric in CARD_METRICS if any(metric in metrics for _, _, metrics, _ in rows)] or list(CARD_METRICS)
    header = "| Model | Input | " + " | ".join(present) + " |"
    separator = "|---|---|" + "---|" * len(present)
    body = [
        f"| {name} | {mode} | " + " | ".join(_format(metrics.get(metric), stds.get(metric)) for metric in present) + " |"
        for name, mode, metrics, stds in rows
    ]
    return "\n".join([header, separator, *body])


def _vocabulary_note(config: RunConfig) -> str:
    vocab = config.vocab
    forms = ["inflected forms"]
    if vocab.add_lemmas:
        forms.append("base forms")
    if vocab.add_lowercase_variants:
        forms.append("lowercase variants")
    sources = "corpus words and keywords" if vocab.include_keywords else "corpus words"
    note = f"The added vocabulary holds {', '.join(forms)} of {sources}."
    if vocab.add_lemmas:
        note += ' The model can be used on raw text and on noun/base-form input (e.g. "Brüche kürzen" -> "Bruch").'
    return note


def _input_mode_note(config: RunConfig) -> str:
    if config.input_mode == "mixed":
        return f"mixed (share of noun/base-form texts: {config.lemma_share})"
    return {"raw": "raw text", "lemma": "noun base forms"}[config.input_mode]


def write_model_card(model_dir: Path, config: RunConfig, results: Mapping, facts: Mapping[str, object]) -> Path:
    """Write README.md (replacing model2vec's generic card) plus the run config and the raw evaluation results.

    `facts` are run statistics shown under "Training", e.g. document and vocabulary counts.
    """
    model_dir = Path(model_dir)
    results = public_results(results, config.card.show_data_facts)
    teacher = _public_name(config.teacher.model_name)
    facts = facts if config.card.show_data_facts else {}
    fact_lines = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    cv = results.get("cv")
    if cv:
        evaluation_note = (
            f"{cv['folds']}-fold cross-validation: {cv['folds']} models were trained with the same recipe, each "
            "evaluated on the fold it did not see (records of one topic and near-duplicates always share a fold). "
            "Values are mean ± standard deviation over the folds. The published model was then trained on all data."
        )
    else:
        evaluation_note = "Held-out test split, documents kept apart from training by title and description."
    card = f"""{_front_matter(config)}

# {config.name}

A [Model2Vec](https://github.com/MinishLab/model2vec) static embedding model for German education content,
distilled from `{teacher}` and fine-tuned on German education texts.

{data_description(config)}

## Usage

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("{MODEL_ID_PLACEHOLDER}")
embeddings = model.encode(["Bruchrechnung für die 6. Klasse", "Photosynthese"])
```

{_vocabulary_note(config)}

## Training

- Teacher: `{teacher}`
- Dimensions: {config.pca_dims}, quantization: {config.quantize_to}
- Training input mode: {_input_mode_note(config)}
- Tokenlearn: {"yes" if config.tokenlearn.enabled else "no"}, pair fine-tuning: {"yes" if config.pairs.enabled else "no"}
{fact_lines}

## Evaluation

{evaluation_note} "raw" feeds texts unchanged, "lemma" reduces queries and documents to noun base forms first.

{metrics_table(results)}

- keyword_*: each keyword is a query, relevant are the documents tagged with it (documents without keywords)
- title_*: find a document's description from its title
- question_*: find a document from a question about it (where the data has questions)
- subject_macro_f1: logistic-regression probe on the embeddings, single-subject documents
"""
    path = model_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    (model_dir / "training_config.json").write_text(
        json.dumps(public_config(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (model_dir / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def upload_model(model_dir: Path, repo_id: str, private: bool = True, token: str | None = None, api=None) -> str:
    """Upload a finished model directory as it is (model card included) to the Hugging Face Hub.

    model2vec's `push_to_hub` re-saves the model and would replace the card, so the folder is uploaded instead.
    The token defaults to `HF_TOKEN` or a `hf auth login` session. An existing repository keeps its visibility.
    Returns the model's Hub URL.

    :raises FileNotFoundError: if `model_dir` holds no model.safetensors.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"No model.safetensors in {model_dir}")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    message = f"Upload {model_dir.name}"
    api.upload_folder(
        folder_path=str(model_dir), repo_id=repo_id, repo_type="model", commit_message=message, ignore_patterns=["README.md"]
    )
    readme = model_dir / "README.md"
    if readme.is_file():
        # The repo id goes only into the uploaded card; the stored one keeps its placeholder for later uploads.
        card = readme.read_text(encoding="utf-8").replace(MODEL_ID_PLACEHOLDER, repo_id)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, repo_type="model", commit_message=message
        )
    return f"https://huggingface.co/{repo_id}"
