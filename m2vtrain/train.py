"""Model building steps: distillation, tokenlearn, pair fine-tuning and quantization (model2vec 0.10)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from model2vec import StaticModel
from model2vec.distill import distill_from_model
from model2vec.model import quantize_model
from model2vec.train import StaticModelForPairSimilarity, StaticModelForSimilarity

from m2vtrain.targets import TeacherOptions, load_transformer


@dataclass(frozen=True)
class TrainOptions:
    learning_rate: float = 1e-3
    max_epochs: int = -1  # -1: until early stopping
    early_stopping_patience: int | None = 5
    batch_size: int | None = None  # None: model2vec picks one
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.max_epochs < 0 and self.early_stopping_patience is None:
            raise ValueError("max_epochs=-1 needs early_stopping_patience, otherwise training never stops.")


def distill_model(
    teacher: TeacherOptions, vocabulary: Sequence[str], pca_dims: int, sif_coefficient: float | None = 1e-4
) -> StaticModel:
    """Distill a float32 static model from the teacher, adding `vocabulary` to its tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher.model_name, trust_remote_code=teacher.trust_remote_code, use_fast=True)
    return distill_from_model(
        model=load_transformer(teacher),  # not distill(model_name=...): it loads remote-code teachers broken
        tokenizer=tokenizer,
        vocabulary=list(vocabulary),
        pca_dims=pca_dims,
        sif_coefficient=sif_coefficient,
        pooling=teacher.pooling,
        quantize_to="float32",
        device=teacher.torch_device,
    )


def _fit_kwargs(options: TrainOptions, seed: int) -> dict:
    return {
        "random_seed": seed,
        "learning_rate": options.learning_rate,
        "max_epochs": options.max_epochs,
        "early_stopping_patience": options.early_stopping_patience,
        "batch_size": options.batch_size,
        "device": options.device,
    }


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def align_to_targets(model: StaticModel, texts: Sequence[str], targets: np.ndarray) -> StaticModel:
    """Rotate all token vectors into the target basis (orthogonal Procrustes).

    The distilled vectors live in a PCA basis of token outputs, the targets in a PCA basis of document
    embeddings. Training only moves tokens that occur in the training texts, so without this rotation unseen
    tokens would stay in the old basis. A rotation keeps vector norms, i.e. the SIF weighting.
    """
    u, _, vt = np.linalg.svd(_unit(model.encode(list(texts))).T @ _unit(targets))
    return StaticModel(
        vectors=(model.embedding @ (u @ vt)).astype(np.float32),
        tokenizer=model.tokenizer,
        config=model.config,
        normalize=model.normalize,
        base_model_name=model.base_model_name,
        language=model.language,
    )


def tokenlearn(
    model: StaticModel, texts: Sequence[str], targets: np.ndarray, options: TrainOptions, seed: int = 42
) -> StaticModel:
    """Align the model to the target basis, then train token vectors and per-token weights so that mean-pooled
    texts match the teacher targets.

    No head (`n_layers=0`), so the result is again a plain static model. `freeze_weights=False` learns
    per-token weights on top of the SIF-scaled vectors.

    :raises ValueError: if the target dimension differs from the model dimension (a head would be needed).
    """
    if targets.shape[1] != model.dim:
        raise ValueError(f"Targets have {targets.shape[1]} dimensions, the model has {model.dim} dimensions.")
    aligned = align_to_targets(model, texts, targets)
    trainer = StaticModelForSimilarity.from_static_model(
        model=aligned, n_layers=0, out_dim=aligned.dim, freeze_weights=False
    )
    trainer.fit(list(texts), targets, **_fit_kwargs(options, seed))
    return trainer.to_static_model()


def pair_finetune(
    model: StaticModel, anchors: Sequence[str], positives: Sequence[str], options: TrainOptions, seed: int = 42
) -> StaticModel:
    """Contrastive (InfoNCE, in-batch negatives) fine-tuning on (anchor, positive) pairs."""
    trainer = StaticModelForPairSimilarity.from_static_model(model=model, n_layers=0, freeze_weights=False)
    trainer.fit(list(anchors), list(positives), **_fit_kwargs(options, seed))
    return trainer.to_static_model()


def finalize(model: StaticModel, quantize_to: str = "float16", vocabulary_quantization: int | None = None) -> StaticModel:
    """Quantize embeddings (and optionally cluster the vocabulary) for the published model."""
    return quantize_model(model, vocabulary_quantization=vocabulary_quantization, quantize_to=quantize_to)
