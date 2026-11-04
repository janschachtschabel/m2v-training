"""Command line: `python -m m2vtrain init-config|run|extract-keywords|compare|upload ...`."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from m2vtrain.config import RunConfig, load_config, save_config
from m2vtrain.status import RunState, Status, write_status

EXIT_FAILED = 1
EXIT_CANCELLED = 130
LOG_FILE = "run.log"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


_MAX_NAME_CHARS = 60  # keeps run and model paths below Windows' 260-character limit
# Third-party loggers that would flood run.log: model2vec warns once per added token, httpx logs every request.
_QUIET_LOGGERS = {"model2vec.tokenizer": logging.ERROR, "httpx": logging.WARNING, "huggingface_hub": logging.WARNING}


def new_run_dir(config: RunConfig, root: Path = Path(".")) -> Path:
    """Create and return a new `<root>/<runs_dir>/<timestamp>-<name>[-n]`, safe as a folder name on Windows."""
    safe = "".join(char if char.isalnum() or char in "-_." else "-" for char in config.name).strip("-.") or "run"
    base = Path(root) / config.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{safe[:_MAX_NAME_CHARS]}"
    base.parent.mkdir(parents=True, exist_ok=True)
    run_dir, suffix = base, 1
    while True:
        try:
            run_dir.mkdir()  # atomic claim, so two starts in the same second never share a folder
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = base.with_name(f"{base.name}-{suffix}")


def _setup_logging(run_dir: Path) -> logging.Handler:
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(console)
    root.setLevel(logging.INFO)
    for name, level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(level)
    return handler


def _terminate_as_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def _init_config(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} exists; use --force to overwrite.", file=sys.stderr)
        return EXIT_FAILED
    raw: dict = {"name": args.name}
    if args.synthetic:
        raw["synthetic"] = {"enabled": True}
        raw["data_files"] = args.data or []
    else:
        raw["data_files"] = args.data or ["data/*.csv.gz", "data/*.csv"]
    if args.keywords:
        raw["keyword_lists"] = {"files": args.keywords}
    if args.folds:
        raw["cross_validation"] = {"folds": args.folds}
    try:
        config = RunConfig.from_dict(raw)
    except ValueError as error:
        print(f"Invalid options: {error}", file=sys.stderr)
        return EXIT_FAILED
    save_config(config, path)
    print(f"Wrote {path}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from m2vtrain.pipeline import run_pipeline  # heavy imports (torch) only when training

    try:
        config = load_config(Path(args.config))
    except (OSError, ValueError) as error:
        print(f"Invalid config: {error}", file=sys.stderr)
        if args.run_dir:  # a UI waiting on this run dir must see the failure
            write_status(Path(args.run_dir), Status(state=RunState.FAILED, stages=[], error=f"Invalid config: {error}"))
        return EXIT_FAILED
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(config)
    handler = _setup_logging(run_dir)
    previous = signal.getsignal(signal.SIGTERM)
    if sys.platform != "win32":
        # POSIX: a stop from the UI ends the run like Ctrl+C, so the status says "cancelled". On Windows the UI's
        # stop terminates the process outright and the UI records the cancellation itself.
        signal.signal(signal.SIGTERM, _terminate_as_interrupt)
    try:
        model_dir = run_pipeline(config, run_dir)
    except KeyboardInterrupt:
        return EXIT_CANCELLED
    except Exception:  # already logged with traceback and recorded in status.json
        return EXIT_FAILED
    finally:
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, previous)
        logging.getLogger().removeHandler(handler)
        handler.close()
    print(f"Model saved to {model_dir}" if model_dir else f"Cross-validation results in {run_dir / 'eval.json'}")
    return 0


def _upload(args: argparse.Namespace) -> int:
    from m2vtrain.export import upload_model

    try:
        url = upload_model(Path(args.model_dir), args.repo_id, private=not args.public)
    except Exception as error:  # missing model, login, Hub errors, network (httpx errors are no OSError)
        print(f"Upload failed: {error}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Uploaded to {url}")
    return 0


def _extract_keywords(args: argparse.Namespace) -> int:
    from m2vtrain.data import DataOptions, load_exports, prepare
    from m2vtrain.sources import keyword_document_frequencies, write_keyword_list

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"{output} exists; use --force to overwrite.", file=sys.stderr)
        return EXIT_FAILED
    try:
        options = DataOptions(languages=tuple(args.languages))
        df = prepare(load_exports([Path(path) for path in args.exports]), options)
    except (OSError, ValueError) as error:
        print(f"Could not read the exports: {error}", file=sys.stderr)
        return EXIT_FAILED
    written = write_keyword_list(output, keyword_document_frequencies(df["keywords"]), args.min_docs, args.with_counts)
    print(f"Wrote {written} keywords to {output}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from m2vtrain.report import comparison_table

    run_dirs = [Path(path) for path in args.runs] or sorted(path for path in Path("runs").glob("*") if path.is_dir())
    print(comparison_table(run_dirs, args.mode, not args.no_baselines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m2vtrain", description="Train German education Model2Vec models.")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init-config", help="write a config file with default options")
    init.add_argument("path")
    init.add_argument("--name", default="m2v-de-bildung")
    init.add_argument("--data", nargs="+", help="data files or glob patterns (default: data/*.csv.gz data/*.csv)")
    init.add_argument("--synthetic", action="store_true", help="use the synthetic dataset (then --data is optional)")
    init.add_argument("--keywords", nargs="+", help="keyword list files (one term per line)")
    init.add_argument("--folds", type=int, default=0, help="k-fold cross-validation with this many folds (default: off)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(handler=_init_config)

    run = commands.add_parser("run", help="run training, evaluation and export for a config file")
    run.add_argument("config")
    run.add_argument("--run-dir", help="default: <runs_dir>/<timestamp>-<name>")
    run.set_defaults(handler=_run)

    extract = commands.add_parser(
        "extract-keywords",
        help="write the keywords of local metadata exports as a plain term list (no titles, descriptions or ids)",
    )
    extract.add_argument("exports", nargs="+", help="export files (';'-separated CSV, also .csv.gz)")
    extract.add_argument("output", help="keyword list to write, e.g. data/keywords.txt")
    extract.add_argument("--min-docs", type=int, default=3, help="keep keywords used by at least this many documents")
    extract.add_argument("--languages", nargs="+", default=["de"], help="document languages to read (default: de)")
    extract.add_argument("--with-counts", action="store_true", help="also write the document count per keyword")
    extract.add_argument("--force", action="store_true", help="overwrite an existing file")
    extract.set_defaults(handler=_extract_keywords)

    compare = commands.add_parser("compare", help="print a Markdown table comparing finished runs")
    compare.add_argument("runs", nargs="*", help="run folders (default: all folders in runs/)")
    compare.add_argument("--mode", choices=["raw", "lemma"], default="raw", help="evaluation input mode")
    compare.add_argument("--no-baselines", action="store_true", help="only the trained models")
    compare.set_defaults(handler=_compare)

    upload = commands.add_parser("upload", help="upload a finished model folder to the Hugging Face Hub")
    upload.add_argument("model_dir")
    upload.add_argument("repo_id", help="e.g. JanSchachtschabel/m2v-de-bildung")
    upload.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    upload.set_defaults(handler=_upload)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
