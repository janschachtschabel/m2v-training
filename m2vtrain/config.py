"""Run configuration: every option of a training run in one JSON file, so each run is reproducible from the CLI."""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass
from pathlib import Path

from m2vtrain.data import DataOptions
from m2vtrain.evaluate import EvalOptions
from m2vtrain.lemma import InputMode
from m2vtrain.sources import KeywordListOptions, SyntheticOptions
from m2vtrain.targets import TeacherOptions
from m2vtrain.train import TrainOptions
from m2vtrain.vocab import VocabOptions

DEFAULT_TEACHER = "Alibaba-NLP/gte-multilingual-base"
QUANTIZE_TYPES = ("float32", "float16", "int8")
EVAL_MODES = (InputMode.RAW.value, InputMode.LEMMA.value)


def _check_limit(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1 (or null for no limit), got {value}.")


@dataclass(frozen=True)
class TokenlearnConfig:
    """Train the distilled model towards teacher embeddings of the training documents."""

    enabled: bool = True
    max_texts: int | None = None  # None: all training documents
    train: TrainOptions = TrainOptions()

    def __post_init__(self) -> None:
        _check_limit("tokenlearn.max_texts", self.max_texts)


@dataclass(frozen=True)
class PairsConfig:
    """Contrastive fine-tuning on keyword->document and title->description pairs."""

    enabled: bool = False
    max_pairs: int | None = 500_000
    train: TrainOptions = TrainOptions(learning_rate=1e-4)

    def __post_init__(self) -> None:
        _check_limit("pairs.max_pairs", self.max_pairs)


@dataclass(frozen=True)
class CrossValidationConfig:
    """k-fold cross-validation instead of a single train/test split.

    With `folds` >= 2 the documents are split into k folds (near-duplicates and records of one topic always in
    the same fold); k models are trained, each evaluated on its held-out fold, and the metrics are averaged.
    With `final_model` a last model is then trained on all documents and saved; its card reports the
    cross-validated metrics. `folds` 0 keeps the single split given by `test_fraction`.
    """

    folds: int = 0
    final_model: bool = True
    # True: one vocabulary (from all documents) and one distillation for all folds. Much faster (distilling a
    # large vocabulary takes minutes on a GPU, half an hour on a CPU), slightly optimistic: test documents' words
    # and keywords exist as tokens, although their vectors are only trained on other folds' texts.
    shared_vocabulary: bool = False

    def __post_init__(self) -> None:
        if self.folds == 1 or self.folds < 0 or self.folds > 20:
            raise ValueError(f"cross_validation.folds must be 0 (single split) or 2..20, got {self.folds}.")


@dataclass(frozen=True)
class CardConfig:
    """What the model card says about the training data.

    By default it names only openly licensed sources (the synthetic dataset, "a keyword list") and no document
    counts; local exports are never described. `data_description` replaces the automatic text.
    """

    data_description: str = ""
    show_data_facts: bool = False


@dataclass(frozen=True)
class RunConfig:
    """All options of one training run.

    `data_files` entries may be glob patterns ("data/*.csv.gz"); they may be empty when the synthetic dataset is
    enabled. `baseline_models` are StaticModel names or paths evaluated next to the new model;
    `evaluate_teacher` also scores the teacher (slow without a GPU).
    """

    name: str
    data_files: tuple[str, ...] = ()
    synthetic: SyntheticOptions = SyntheticOptions()
    keyword_lists: KeywordListOptions = KeywordListOptions()
    data: DataOptions = DataOptions()
    test_fraction: float = 0.05
    cross_validation: CrossValidationConfig = CrossValidationConfig()
    seed: int = 42
    teacher: TeacherOptions = TeacherOptions(model_name=DEFAULT_TEACHER, trust_remote_code=True)
    vocab: VocabOptions = VocabOptions()
    pca_dims: int = 256
    sif_coefficient: float | None = 1e-4
    input_mode: str = InputMode.MIXED.value
    lemma_share: float = 0.5
    tokenlearn: TokenlearnConfig = TokenlearnConfig()
    pairs: PairsConfig = PairsConfig()
    quantize_to: str = "float16"
    vocabulary_quantization: int | None = None
    eval: EvalOptions = EvalOptions()
    eval_modes: tuple[str, ...] = EVAL_MODES
    baseline_models: tuple[str, ...] = ()
    evaluate_teacher: bool = False
    license: str = "apache-2.0"
    card: CardConfig = CardConfig()
    runs_dir: str = "runs"
    models_dir: str = "models"
    cache_dir: str = "cache"

    def __post_init__(self) -> None:
        # JSON gives lists; tuples keep the frozen config hashable and comparable after a round trip.
        for name in ("data_files", "eval_modes", "baseline_models"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        if not self.name.strip():
            raise ValueError("name must not be empty.")
        if not self.data_files and not self.synthetic.enabled:
            raise ValueError("data_files must list at least one export file, or synthetic.enabled must be true.")
        if self.input_mode not in {mode.value for mode in InputMode}:
            raise ValueError(f"input_mode must be one of raw, lemma, mixed, got {self.input_mode!r}.")
        if not self.eval_modes or not set(self.eval_modes) <= set(EVAL_MODES):
            raise ValueError(f"eval_modes must be a non-empty subset of {list(EVAL_MODES)}, got {list(self.eval_modes)}.")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}.")
        if not 0.0 <= self.lemma_share <= 1.0:
            raise ValueError(f"lemma_share must be in [0, 1], got {self.lemma_share}.")
        if self.pca_dims < 1:
            raise ValueError(f"pca_dims must be positive, got {self.pca_dims}.")
        _check_limit("vocabulary_quantization", self.vocabulary_quantization)
        if self.quantize_to not in QUANTIZE_TYPES:
            raise ValueError(f"quantize_to must be one of {list(QUANTIZE_TYPES)}, got {self.quantize_to!r}.")

    @classmethod
    def from_dict(cls, raw: dict) -> RunConfig:
        """Build a config from (possibly partial) JSON data; missing keys keep their defaults.

        :raises ValueError: on unknown keys or invalid values.
        """
        return _build(cls, raw, "")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def _union_members(hint) -> tuple:
    return typing.get_args(hint) if typing.get_origin(hint) in (typing.Union, types.UnionType) else (hint,)


def _matches(kind, value) -> bool:
    if kind is bool:
        return isinstance(value, bool)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in (str, Path):
        return isinstance(value, str)
    if typing.get_origin(kind) is tuple:
        return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)
    return False


def _convert(hint, value, key: str):
    """Type-check a JSON value against a field's type hint and convert lists and paths."""
    members = _union_members(hint)
    if value is None:
        if type(None) in members:
            return None
        raise ValueError(f"{key} must not be null.")
    kind = next((member for member in members if _matches(member, value)), None)
    if kind is None:
        raise ValueError(f"{key} has the wrong type: {value!r}")
    if typing.get_origin(kind) is tuple:
        return tuple(value)
    if kind is float:
        return float(value)  # JSON 1 for 1.0; keeps types stable for the UI widgets
    return Path(value) if kind is Path else value


def _build(cls, raw, where: str, default=None):
    """Build dataclass `cls` from JSON data, starting from `default` (the field's default instance) if given."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where.rstrip('.') or 'config'} must be a JSON object.")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(where + key for key in unknown)}")
    kwargs = {}
    for name, value in raw.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            nested_default = getattr(default, name) if default is not None else fields[name].default
            if nested_default is dataclasses.MISSING:
                nested_default = None
            kwargs[name] = _build(hint, value, f"{where}{name}.", nested_default)
        else:
            kwargs[name] = _convert(hint, value, where + name)
    try:
        return dataclasses.replace(default, **kwargs) if default is not None else cls(**kwargs)
    except TypeError as error:  # e.g. a missing required key such as "name"
        raise ValueError(f"{where.rstrip('.') or 'config'}: {error}") from error


def save_config(config: RunConfig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_config(path: Path) -> RunConfig:
    """:raises ValueError: if the file is not valid JSON or holds unknown keys or invalid values."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    return RunConfig.from_dict(raw)
