"""Teacher sentence embeddings used as tokenlearn targets (cached on disk) and their PCA reduction."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

Progress = Callable[[int, int], None]
DEFAULT_CHUNK_SIZE = 4096
_CACHE_FORMAT = "v3"  # v3: remote-code teachers with restored buffers, float32 on the CPU
POOLING_MODES = ("mean", "last", "first", "pooler")
DTYPES = ("auto", "float32", "float16", "bfloat16")


@dataclass(frozen=True)
class TeacherOptions:
    """The sentence-transformer teacher.

    `pooling` is only used for distillation (model2vec's per-token forward pass: "mean", "last" for decoder
    models such as Qwen3-Embedding, "first" for CLS models, "pooler"); target embeddings use the teacher's own
    Sentence Transformers pooling. `document_prefix` is prepended to every text, e.g. "passage: " for e5 models.
    `max_seq_length` caps the teacher's input in tokens (None: the model's own limit; long inputs cost memory
    quadratically). `device` None or "auto" picks the best available device. `dtype` "auto" means float32 on
    the CPU and the stored precision on a GPU; "bfloat16"/"float16" halve memory and time on a GPU.
    """

    model_name: str
    pooling: str = "mean"
    trust_remote_code: bool = False
    document_prefix: str = ""
    batch_size: int = 64
    device: str | None = None
    max_seq_length: int | None = None
    dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"teacher.dtype must be one of {list(DTYPES)}, got {self.dtype!r}.")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"teacher.pooling must be one of {list(POOLING_MODES)}, got {self.pooling!r}.")
        if self.max_seq_length is not None and self.max_seq_length < 8:
            raise ValueError(f"teacher.max_seq_length must be at least 8 (or null), got {self.max_seq_length}.")
        if self.batch_size < 1:
            raise ValueError(f"teacher.batch_size must be at least 1, got {self.batch_size}.")

    @property
    def torch_device(self) -> str | None:
        return None if self.device in (None, "auto") else self.device


def _local_fingerprint(model_name: str) -> str:
    """Size and mtime of a local teacher's files, so retrained local weights don't reuse stale targets.

    simplify: Hub models are keyed by name only; a new Hub revision of the same name reuses the cache.
    """
    path = Path(model_name)
    if not path.is_dir():
        return ""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return ";".join(f"{p.relative_to(path)}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files)


def _cache_key(texts: Sequence[str], options: TeacherOptions) -> str:
    digest = hashlib.sha256()
    header = [_CACHE_FORMAT, options.model_name, _local_fingerprint(options.model_name), options.document_prefix]
    if options.max_seq_length is not None:  # only when set, so caches of earlier runs stay valid
        header.append(f"max_seq_length={options.max_seq_length}")
    if options.dtype != "auto":
        header.append(f"dtype={options.dtype}")
    digest.update("\x00".join(header).encode("utf-8"))
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _load_dtype(device: str | None, dtype: str = "auto"):
    """float32 on the CPU (half precision is slow there; gte-multilingual is stored in float16), else `dtype`
    ("auto": the stored precision)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.float32
    return "auto" if dtype == "auto" else getattr(torch, dtype)


def restore_computed_buffers(model) -> None:
    """Recompute the buffers a remote-code model computes in `__init__` instead of storing them.

    transformers 5 builds models on the meta device and fills in only stored weights. Its own model classes
    re-initialize non-persistent buffers, remote-code models such as gte-multilingual do not: their position ids
    and rotary caches keep uninitialized memory (IndexError or silently wrong embeddings). A fresh instance from
    the config computes them exactly as the model's `__init__` does. Its random weights (as large as the model's)
    are freed right after.
    """
    if type(model).__module__.startswith("transformers."):
        return
    import torch

    with torch.device("cpu"):
        fresh = dict(type(model)(model.config).named_modules())
    for name, module in model.named_modules():
        for buffer_name in module._non_persistent_buffers_set:
            current, computed = getattr(module, buffer_name), getattr(fresh[name], buffer_name)
            if current is None or computed is None:
                continue
            restored = computed.to(device=current.device, dtype=current.dtype)
            module.register_buffer(buffer_name, restored, persistent=False)


def load_transformer(options: TeacherOptions):
    """The teacher's transformer (for distillation) with restored buffers, float32 on the CPU."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        options.model_name, trust_remote_code=options.trust_remote_code, dtype=_load_dtype(options.torch_device, options.dtype)
    )
    restore_computed_buffers(model)
    return model


def load_sentence_transformer(name: str, device: str | None = None, trust_remote_code: bool = False, dtype: str = "auto"):
    """A SentenceTransformer with restored buffers (see `restore_computed_buffers`), float32 on the CPU."""
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedModel

    model = SentenceTransformer(
        name, device=device, trust_remote_code=trust_remote_code, model_kwargs={"dtype": _load_dtype(device, dtype)}
    )
    transformer = next((module for module in model.modules() if isinstance(module, PreTrainedModel)), None)
    if transformer is not None:  # None for e.g. sentence-transformers' own static embedding models
        restore_computed_buffers(transformer)
    return model


def load_teacher(options: TeacherOptions):
    """The teacher as a SentenceTransformer, for target embeddings."""
    model = load_sentence_transformer(options.model_name, options.torch_device, options.trust_remote_code, options.dtype)
    if options.max_seq_length is not None:
        model.max_seq_length = options.max_seq_length
    return model


def encode_with_teacher(
    texts: Sequence[str],
    options: TeacherOptions,
    cache_dir: Path,
    progress: Progress | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Normalized float32 teacher embeddings for `texts`, cached in `cache_dir/<hash>.npy`.

    Finished chunks are kept as `<hash>.chunk<i>.npy` until the whole result is cached, so an interrupted
    run resumes where it stopped.

    :raises ValueError: if `texts` is empty.
    """
    if not texts:
        raise ValueError("Cannot encode: no texts given.")
    cache_dir = Path(cache_dir)
    key = _cache_key(texts, options)
    cache_file = cache_dir / f"{key}.npy"
    if cache_file.exists():
        logger.info("Using cached teacher embeddings %s", cache_file.name)
        if progress:
            progress(len(texts), len(texts))
        return np.load(cache_file)

    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = None
    out: np.ndarray | None = None
    chunk_files = []
    for index, start in enumerate(range(0, len(texts), chunk_size)):
        chunk_file = cache_dir / f"{key}.chunk{index}.npy"
        chunk_files.append(chunk_file)
        if chunk_file.exists():
            vectors = np.load(chunk_file)
        else:
            if teacher is None:
                teacher = load_teacher(options)
            chunk = [options.document_prefix + text for text in texts[start : start + chunk_size]]
            vectors = teacher.encode(chunk, batch_size=options.batch_size, normalize_embeddings=True)
            partial_chunk = chunk_file.with_name(f"{chunk_file.stem}.partial.npy")
            np.save(partial_chunk, np.asarray(vectors, dtype=np.float32))
            partial_chunk.replace(chunk_file)  # a killed run never leaves a truncated chunk behind
        if out is None:
            out = np.empty((len(texts), vectors.shape[1]), dtype=np.float32)
        out[start : start + len(vectors)] = vectors
        if progress:
            progress(start + len(vectors), len(texts))

    partial = cache_dir / f"{key}.partial.npy"
    np.save(partial, out)
    partial.replace(cache_file)  # atomic, so an interrupted run never leaves a truncated cache file
    for chunk_file in chunk_files:
        chunk_file.unlink(missing_ok=True)
    return out


def reduce_targets(vectors: np.ndarray, dims: int, rows: Sequence[int] | None = None) -> np.ndarray:
    """Center and PCA-project teacher vectors to the static model's dimensionality (as tokenlearn does).

    PCA is applied even when `dims` equals the teacher dimension, so targets are centered in every run. With
    `rows`, the result has one target per entry of `rows` (repeats allowed, e.g. spelling variants of one
    keyword); the PCA is fitted on the distinct rows only, i.e. on this fold's training texts.

    :raises ValueError: if more dimensions are requested than the teacher has.
    """
    if dims > vectors.shape[1]:
        raise ValueError(f"Cannot reduce {vectors.shape[1]} teacher dimensions to {dims} dimensions.")
    if rows is None:
        return PCA(n_components=dims, random_state=0).fit_transform(vectors).astype(np.float32, copy=False)
    rows = np.asarray(rows, dtype=np.int64)
    distinct, inverse = np.unique(rows, return_inverse=True)
    reduced = PCA(n_components=dims, random_state=0).fit_transform(vectors[distinct]).astype(np.float32, copy=False)
    return reduced[inverse]
