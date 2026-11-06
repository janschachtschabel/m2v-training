"""One training run end to end: data -> vocabulary -> distillation -> tokenlearn -> pairs -> export -> evaluation."""

from __future__ import annotations

import copy
import functools
import gc
import glob
import hashlib
import json
import logging
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from model2vec import StaticModel

from m2vtrain.config import RunConfig, save_config
from m2vtrain.corpus import TokenlearnPool, build_pairs, build_tokenlearn_pool, document_text, keyword_list_pairs
from m2vtrain.data import load_exports, prepare
from m2vtrain.evaluate import Encoder, evaluate, is_model2vec, load_baseline, tfidf_encoder, with_input_mode
from m2vtrain.export import write_model_card
from m2vtrain.fsutil import replace_with_retry
from m2vtrain.keywords import load_stoplist
from m2vtrain.lemma import InputMode, transform_inputs
from m2vtrain.sources import load_keyword_lists, load_synthetic, with_document_columns
from m2vtrain.split import assign_folds, split_train_test
from m2vtrain.status import StatusWriter
from m2vtrain.targets import encode_with_teacher, load_teacher, reduce_targets
from m2vtrain.train import distill_model, finalize, pair_finetune, tokenlearn
from m2vtrain.vocab import build_vocabulary

logger = logging.getLogger(__name__)

PARTIAL_SUFFIX = ".partial"
_RENAME_SECONDS = 60.0  # Windows Defender may hold a freshly written model file for a while
STAGE_DATA = "Daten vorbereiten"
STAGE_VOCAB = "Vokabular"
STAGE_DISTILL = "Destillation"
STAGE_TARGETS = "Lehrer-Embeddings"
STAGE_TOKENLEARN = "Tokenlearn"
STAGE_PAIRS = "Paar-Training"
STAGE_SAVE = "Speichern"
STAGE_EVAL = "Evaluation"
FINAL_PREFIX = "Finales Modell"


def is_cross_validation(config: RunConfig) -> bool:
    return config.cross_validation.folds >= 2


def produces_model(config: RunConfig) -> bool:
    """False for a cross-validation run without final model: it only writes eval.json."""
    return not is_cross_validation(config) or config.cross_validation.final_model


def stages_for(config: RunConfig) -> list[str]:
    shared = config.cross_validation.shared_vocabulary
    fit = [] if shared and is_cross_validation(config) else [STAGE_VOCAB, STAGE_DISTILL]
    if config.tokenlearn.enabled:
        fit.append(STAGE_TOKENLEARN)
    if config.pairs.enabled:
        fit.append(STAGE_PAIRS)
    if not is_cross_validation(config):
        stages = [STAGE_DATA, STAGE_VOCAB, STAGE_DISTILL]
        if config.tokenlearn.enabled:
            stages += [STAGE_TARGETS, STAGE_TOKENLEARN]
        if config.pairs.enabled:
            stages.append(STAGE_PAIRS)
        return stages + [STAGE_SAVE, STAGE_EVAL]
    k = config.cross_validation.folds
    stages = [STAGE_DATA] + ([STAGE_TARGETS] if config.tokenlearn.enabled else [])
    if shared:
        stages += [STAGE_VOCAB, STAGE_DISTILL]
    for fold in range(k):
        stages += [f"Fold {fold + 1}/{k}: {stage}" for stage in [*fit, STAGE_EVAL]]
    if config.cross_validation.final_model:
        stages += [f"{FINAL_PREFIX}: {stage}" for stage in fit] + [STAGE_SAVE]
    return stages


def resolve_data_files(patterns: tuple[str, ...]) -> list[Path]:
    """Expand glob patterns; plain paths must exist.

    :raises FileNotFoundError: if a plain path is missing or nothing matches at all.
    """
    files: list[Path] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            files += sorted(Path(match) for match in glob.glob(pattern))
        elif Path(pattern).is_file():
            files.append(Path(pattern))
        else:
            raise FileNotFoundError(f"Data file not found: {pattern}")
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No data files match {list(patterns)}")
    return files


def model_dir_for(config: RunConfig, run_dir: Path) -> Path:
    return Path(config.models_dir) / Path(run_dir).name


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _documents(df: pd.DataFrame) -> list[str]:
    return [document_text(row, with_keywords=False) for _, row in df.iterrows()]


def _load_teacher_encoder(config: RunConfig) -> Encoder:
    teacher = load_teacher(config.teacher)
    prefix, batch_size = config.teacher.document_prefix, config.teacher.batch_size
    return lambda texts: teacher.encode([prefix + text for text in texts], batch_size=batch_size, normalize_embeddings=True)


def _encoder_loaders(config: RunConfig, model: StaticModel, train_df: pd.DataFrame) -> dict[str, Callable]:
    """Per encoder name a function that loads it: the new model, TF-IDF per input mode, baselines, the teacher.

    Loaded one at a time during the evaluation, so at most one transformer is in memory.
    """
    documents = _documents(train_df)
    device, batch_size = config.teacher.torch_device, config.teacher.batch_size
    # "tfidf" loads the training documents; it is fitted per input mode in _evaluate_all.
    loaders: dict[str, Callable] = {"model": lambda: model.encode, "tfidf": lambda: documents}
    loaders.update({name: functools.partial(load_baseline, name, device, batch_size) for name in config.baseline_models})
    if config.evaluate_teacher:
        loaders["teacher"] = functools.partial(_load_teacher_encoder, config)
    return loaders


def _evaluate_all(
    config: RunConfig,
    model: StaticModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    progress: Callable,
) -> dict:
    loaders = _encoder_loaders(config, model, train_df)
    total = len(loaders) * len(config.eval_modes)
    results: dict = {"model": {}, "baselines": {}}
    done = 0
    for name, load in loaders.items():
        loaded = load()
        for mode in map(InputMode, config.eval_modes):
            progress(done, total, f"{name} ({mode.value})")
            started = time.perf_counter()
            if name == "tfidf":  # fitted on the training documents in the same input form it is evaluated on
                encode = with_input_mode(tfidf_encoder(transform_inputs(loaded, mode)), mode)
            else:
                encode = with_input_mode(loaded, mode)
            metrics = evaluate(encode, train_df, test_df, config.eval)
            logger.info("Evaluated %s (%s) in %.1f s: %s", name, mode.value, time.perf_counter() - started, metrics)
            target = results["model"] if name == "model" else results["baselines"].setdefault(name, {})
            target[mode.value] = metrics
            done += 1
        del loaded, encode
        gc.collect()  # frees a transformer baseline before the next one is loaded
    progress(done, total, "")
    return results


def _split_summary(test_df: pd.DataFrame) -> dict:
    """Identifies the test split, so results from different splits are never compared by accident."""
    keys = sorted(f"{doc_id}\t{title}" for doc_id, title in zip(test_df["id"], test_df["title"]))  # ids may be empty
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return {"documents": len(test_df), "fingerprint": digest}


def _cv_summary(df: pd.DataFrame, folds: list[int], k: int) -> dict:
    """Identifies a k-fold partition (documents and their fold), like `_split_summary` for a single split."""
    keys = sorted(f"{fold}\t{doc_id}\t{title}" for fold, doc_id, title in zip(folds, df["id"], df["title"]))
    digest = hashlib.sha256(f"cv{k}\n".encode("utf-8") + "\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return {"documents": len(df), "folds": k, "fingerprint": digest}


@dataclass
class Corpus:
    """All documents of a run (DOCUMENT_COLUMNS, index 0..n-1) and the terms of its keyword lists."""

    documents: pd.DataFrame
    keyword_terms: list[str]
    facts: dict


def load_corpus(config: RunConfig) -> Corpus:
    """Local exports, the synthetic dataset and keyword lists, as configured.

    :raises ValueError: if no document is left after cleaning.
    """
    stoplist = load_stoplist(config.data.stoplist_path)
    frames, facts = [], {}
    if config.data_files:
        files = resolve_data_files(config.data_files)
        logger.info("Loading %s", ", ".join(map(str, files)))
        exports = with_document_columns(prepare(load_exports(files), config.data), "export")
        frames.append(exports)
        facts.update(data_files=", ".join(path.name for path in files), export_documents=len(exports))
    if config.synthetic.enabled:
        synthetic = load_synthetic(config.synthetic, stoplist)
        frames.append(synthetic)
        facts["synthetic_documents"] = len(synthetic)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No documents left after cleaning; check the data files and the language filter.")
    terms = load_keyword_lists(config.keyword_lists, stoplist) if config.keyword_lists.files else []
    if terms:
        facts["keyword_list_terms"] = len(terms)
    return Corpus(documents=df, keyword_terms=terms, facts=facts)


def _chunk_chars(config: RunConfig) -> int | None:
    return config.synthetic.fulltext_chunk_chars if config.synthetic.use_fulltext else None


def _questions(config: RunConfig) -> bool:
    return config.synthetic.use_questions


def build_pool(config: RunConfig, df: pd.DataFrame, keyword_terms: list[str]) -> TokenlearnPool:
    """Tokenlearn texts of `df` (sampled to `tokenlearn.max_texts`) plus the keyword-list terms."""
    options = config.keyword_lists
    return build_tokenlearn_pool(
        df,
        config.tokenlearn.max_texts,
        config.seed,
        chunk_chars=_chunk_chars(config),
        questions=_questions(config),
        keyword_terms=keyword_terms if options.tokenlearn else (),
        lowercase_variants=options.lowercase_inputs,
        lemma_variants=options.lemma_inputs,
    )


def _encode_pool(config: RunConfig, pool: TokenlearnPool, writer: StatusWriter) -> np.ndarray:
    return encode_with_teacher(
        pool.teacher_texts,
        config.teacher,
        Path(config.cache_dir) / "teacher",
        progress=lambda done, total: writer.progress(done, total, f"{done:,} / {total:,} Texte"),
    )


def _vocabulary_texts(df: pd.DataFrame) -> list[str]:
    """Texts whose words go into the vocabulary: documents and, where present, their full texts."""
    texts = _documents(df)
    texts += [text for text in df["fulltext"] if text] if "fulltext" in df else []
    return texts


def _all_pairs(config: RunConfig, train_df: pd.DataFrame, keyword_terms: list[str]) -> tuple[list[str], list[str]]:
    """Document pairs plus keyword-list pairs, capped together at `pairs.max_pairs`."""
    with_terms = bool(keyword_terms) and config.keyword_lists.pairs
    if not with_terms:  # exactly the pairs of earlier versions for runs without keyword lists
        return build_pairs(train_df, config.pairs.max_pairs, config.seed, _questions(config))
    anchors, positives = build_pairs(train_df, None, config.seed, _questions(config))
    extra_anchors, extra_positives = keyword_list_pairs(
        train_df, keyword_terms, config.keyword_lists.max_docs_per_keyword, config.seed
    )
    pairs = list(zip(anchors + extra_anchors, positives + extra_positives))
    if config.pairs.max_pairs is not None and len(pairs) > config.pairs.max_pairs:
        pairs = random.Random(config.seed).sample(pairs, config.pairs.max_pairs)
    return [anchor for anchor, _ in pairs], [positive for _, positive in pairs]


def _fit(
    config: RunConfig,
    train_df: pd.DataFrame,
    keyword_terms: list[str],
    writer: StatusWriter,
    prefix: str = "",
    pool: TokenlearnPool | None = None,
    vectors: np.ndarray | None = None,
    run_dir: Path | None = None,
    base_model: StaticModel | None = None,
) -> tuple[StaticModel, dict]:
    """Vocabulary, distillation, tokenlearn and pair training on `train_df`. Returns the model and run facts.

    Without `pool`/`vectors` (single split) the pool is built from `train_df` and its teacher embeddings are
    computed here; with them (cross-validation) only the examples of `train_df` and the keyword list are used.
    With `base_model` (a shared distilled model) vocabulary and distillation are skipped.
    """
    mode = InputMode(config.input_mode)
    if base_model is None:
        model, facts = _distill(config, train_df, keyword_terms, writer, prefix, run_dir)
    else:
        model, facts = copy.deepcopy(base_model), {}

    if config.tokenlearn.enabled:
        if pool is None:
            pool = build_pool(config, train_df, keyword_terms)
        if vectors is None:
            writer.stage(prefix + STAGE_TARGETS)
            vectors = _encode_pool(config, pool, writer)
        examples = pool.select(set(train_df.index))
        targets = reduce_targets(vectors, model.dim, rows=[example.teacher_index for example in examples])

        writer.stage(prefix + STAGE_TOKENLEARN)
        writer.progress(0, 0, f"{len(examples):,} Texte, Eingabe: {mode.value}")
        # Targets always come from the full original text; only the static model's input changes with the mode.
        transformed = iter(
            transform_inputs([e.text for e in examples if e.transform], mode, config.lemma_share, config.seed)
        )
        inputs = [next(transformed) if example.transform else example.text for example in examples]
        model = tokenlearn(model, inputs, targets, config.tokenlearn.train, config.seed)
        facts["tokenlearn_texts"] = len(inputs)
        del targets

    if config.pairs.enabled:
        writer.stage(prefix + STAGE_PAIRS)
        anchors, positives = _all_pairs(config, train_df, keyword_terms)
        writer.progress(0, 0, f"{len(anchors):,} Paare")
        anchors = transform_inputs(anchors, mode, config.lemma_share, config.seed)
        positives = transform_inputs(positives, mode, config.lemma_share, config.seed + 1)
        model = pair_finetune(model, anchors, positives, config.pairs.train, config.seed)
        facts["pairs"] = len(anchors)
    return model, facts


def _distill(
    config: RunConfig,
    df: pd.DataFrame,
    keyword_terms: list[str],
    writer: StatusWriter,
    prefix: str = "",
    run_dir: Path | None = None,
) -> tuple[StaticModel, dict]:
    """Build the vocabulary from `df` (and the keyword lists) and distill the teacher with it."""
    facts: dict[str, object] = {}
    writer.stage(prefix + STAGE_VOCAB)
    extra = keyword_terms if config.keyword_lists.add_to_vocabulary else []
    vocabulary = build_vocabulary(_vocabulary_texts(df), df["keywords"], config.vocab, extra_keywords=extra)
    if run_dir is not None:
        (run_dir / "vocabulary.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    facts["added_vocabulary"] = len(vocabulary)
    logger.info("%sVocabulary: %d entries", prefix, len(vocabulary))

    writer.stage(prefix + STAGE_DISTILL)
    writer.progress(0, 0, "Lehrer laden, Vokabular einfügen und destillieren (kann einige Minuten dauern)")
    model = distill_model(config.teacher, vocabulary, config.pca_dims, config.sif_coefficient)
    facts["tokens"] = len(model.tokens)
    logger.info("%sDistilled model: %d tokens, %d dimensions", prefix, len(model.tokens), model.dim)
    return model, facts


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)  # a killed run never leaves a truncated file


def _check_baselines(config: RunConfig) -> None:
    # Checked now, so a wrong name fails in the first seconds instead of after hours of training. Loaded only for
    # the evaluation: transformer baselines would hold RAM or GPU memory through the whole training.
    for name in config.baseline_models:
        is_model2vec(name)  # only for its errors: a missing folder or Hub model raises


def _train(config: RunConfig, run_dir: Path, model_dir: Path, writer: StatusWriter) -> None:
    """Single train/test split; the model is built in `model_dir` (a staging folder) together with its card."""
    _seed_everything(config.seed)
    writer.stage(STAGE_DATA)
    _check_baselines(config)
    corpus = load_corpus(config)
    train_df, test_df = split_train_test(corpus.documents, config.test_fraction, config.seed)
    facts: dict[str, object] = {**corpus.facts, "train_documents": len(train_df), "test_documents": len(test_df)}
    logger.info("%d documents: %d train, %d test", len(corpus.documents), len(train_df), len(test_df))

    model, fit_facts = _fit(config, train_df, corpus.keyword_terms, writer, run_dir=run_dir)
    facts.update(fit_facts)

    writer.stage(STAGE_SAVE)
    model = finalize(model, config.quantize_to, config.vocabulary_quantization)
    model.save_pretrained(model_dir)
    logger.info("Saved model to %s", model_dir)

    writer.stage(STAGE_EVAL)
    results = _evaluate_all(config, model, train_df, test_df, writer.progress)
    results.update(facts=facts, test_split=_split_summary(test_df), eval_options=asdict(config.eval))
    _write_json(run_dir / "eval.json", results)
    write_model_card(model_dir, config, results, facts)


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Mean over folds per encoder, input mode and metric (same layout as a single evaluation), plus the
    standard deviation in `cv.std` and every fold's value in `cv.values`."""
    def collect(get) -> dict:
        values: dict = {}
        for result in fold_results:
            for mode, metrics in get(result).items():
                for metric, value in metrics.items():
                    values.setdefault(mode, {}).setdefault(metric, []).append(value)
        return values

    def reduce(values: dict, function) -> dict:
        return {mode: {metric: float(function(v)) for metric, v in metrics.items()} for mode, metrics in values.items()}

    model_values = collect(lambda result: result["model"])
    names = list(dict.fromkeys(name for result in fold_results for name in result["baselines"]))
    baseline_values = {name: collect(lambda result, name=name: result["baselines"].get(name, {})) for name in names}
    return {
        "model": reduce(model_values, np.mean),
        "baselines": {name: reduce(values, np.mean) for name, values in baseline_values.items()},
        "cv": {
            "folds": len(fold_results),
            "std": {
                "model": reduce(model_values, np.std),
                "baselines": {name: reduce(values, np.std) for name, values in baseline_values.items()},
            },
            "values": {"model": model_values, "baselines": baseline_values},
        },
    }


def _cross_validate(config: RunConfig, run_dir: Path, model_dir: Path | None, writer: StatusWriter) -> None:
    """k-fold cross-validation, then (with `final_model`) one model trained on all documents."""
    _seed_everything(config.seed)
    k = config.cross_validation.folds
    writer.stage(STAGE_DATA)
    _check_baselines(config)
    corpus = load_corpus(config)
    df = corpus.documents
    folds = assign_folds(df, k, config.seed)
    sizes = [folds.count(fold) for fold in range(k)]
    if min(sizes) == 0:
        raise ValueError(f"A fold is empty ({sizes}); use fewer folds or more documents.")
    logger.info("%d documents in %d folds: %s", len(df), k, sizes)
    facts: dict[str, object] = {**corpus.facts, "documents": len(df), "folds": k}

    pool, vectors = None, None
    if config.tokenlearn.enabled:  # one pool and one teacher pass for all folds and the final model
        pool = build_pool(config, df, corpus.keyword_terms)
        writer.stage(STAGE_TARGETS)
        vectors = _encode_pool(config, pool, writer)
    shared = None
    if config.cross_validation.shared_vocabulary:
        shared, shared_facts = _distill(config, df, corpus.keyword_terms, writer, run_dir=run_dir)
        facts.update(shared_facts)

    fold_results = []
    fold_of = pd.Series(folds, index=df.index)
    for fold in range(k):
        prefix = f"Fold {fold + 1}/{k}: "
        _seed_everything(config.seed)
        train_df, test_df = df[fold_of != fold], df[fold_of == fold]
        model, _ = _fit(config, train_df, corpus.keyword_terms, writer, prefix, pool, vectors, base_model=shared)
        model = finalize(model, config.quantize_to, config.vocabulary_quantization)
        writer.stage(prefix + STAGE_EVAL)
        result = _evaluate_all(config, model, train_df, test_df, writer.progress)
        result.update(test_split=_split_summary(test_df), train_documents=len(train_df))
        _write_json(run_dir / "folds" / f"fold-{fold + 1}" / "eval.json", result)
        fold_results.append(result)
        logger.info("%sraw metrics: %s", prefix, result["model"].get("raw", {}))
        del model
        gc.collect()

    results = aggregate_folds(fold_results)
    results.update(test_split=_cv_summary(df, folds, k), eval_options=asdict(config.eval))
    if config.cross_validation.final_model and model_dir is not None:
        _seed_everything(config.seed)
        model, fit_facts = _fit(
            config, df, corpus.keyword_terms, writer, f"{FINAL_PREFIX}: ", pool, vectors, run_dir, base_model=shared
        )
        facts.update(fit_facts)
        writer.stage(STAGE_SAVE)
        model = finalize(model, config.quantize_to, config.vocabulary_quantization)
        model.save_pretrained(model_dir)
        logger.info("Saved final model (all documents) to %s", model_dir)
    results["facts"] = facts
    _write_json(run_dir / "eval.json", results)
    if config.cross_validation.final_model and model_dir is not None:
        write_model_card(model_dir, config, results, facts)


def _discard_or_keep(staging: Path) -> bool:
    """After a failure: delete an unfinished staging folder, but keep a model that was already trained and saved
    (a late failure such as evaluation must not throw away hours of training). Returns True if kept."""
    if (staging / "model.safetensors").is_file():
        logger.warning("Keeping the trained but unevaluated model in %s", staging)
        return True
    shutil.rmtree(staging, ignore_errors=True)
    return False


def run_pipeline(config: RunConfig, run_dir: Path) -> Path | None:
    """Run all stages, writing config.json, status.json, vocabulary.txt and eval.json to `run_dir` and the model
    to `<models_dir>/<run_dir name>`. Returns the model directory. The model is built in
    `<run_dir name>.partial` and only renamed when every stage succeeded, so the models folder never holds an
    unevaluated model. If a stage after saving fails, the `.partial` folder is kept and named in the error.

    Cross-validation runs also write `folds/fold-<i>/eval.json`; without `final_model` they save no model and
    return None.

    Failures and cancellations (KeyboardInterrupt) are recorded in the status and re-raised.

    :raises FileExistsError: if the model directory already holds files.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.json")
    model_dir = model_dir_for(config, run_dir)
    staging = model_dir.with_name(model_dir.name + PARTIAL_SUFFIX)
    writer = StatusWriter(run_dir, stages_for(config))
    started = time.perf_counter()
    try:
        if model_dir.exists() and any(model_dir.iterdir()):
            raise FileExistsError(f"Model directory {model_dir} is not empty; use a new run name.")
        shutil.rmtree(staging, ignore_errors=True)  # left over from an earlier, interrupted attempt of this run
        if is_cross_validation(config):
            _cross_validate(config, run_dir, staging if produces_model(config) else None, writer)
        else:
            _train(config, run_dir, staging, writer)
        if not produces_model(config):
            logger.info("Cross-validation finished in %.0f s (no final model requested)", time.perf_counter() - started)
            writer.finish(None)
            return None
        if model_dir.exists():
            model_dir.rmdir()  # empty (checked above); Windows cannot rename onto an existing folder
        replace_with_retry(staging, model_dir, seconds=_RENAME_SECONDS, pause=0.5)
    except KeyboardInterrupt:
        logger.warning("Run cancelled")
        _discard_or_keep(staging)
        writer.cancel()
        raise
    except Exception as error:
        logger.exception("Run failed")
        kept = _discard_or_keep(staging)
        if kept:
            error.add_note(f"Das trainierte Modell liegt unausgewertet in {staging}")
        writer.fail(error)
        raise
    logger.info("Run finished in %.0f s", time.perf_counter() - started)
    writer.finish(model_dir)
    return model_dir
