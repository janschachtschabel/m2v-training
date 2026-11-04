"""Markdown comparison of finished runs (mean ± std for cross-validated runs), for the terminal or the docs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from m2vtrain.export import CARD_METRICS, _format


def _load(run_dir: Path) -> tuple[dict, dict] | None:
    eval_file, config_file = run_dir / "eval.json", run_dir / "config.json"
    if not eval_file.is_file():
        return None
    results = json.loads(eval_file.read_text(encoding="utf-8"))
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    return results, config


def comparison_table(run_dirs: Sequence[Path], mode: str = "raw", with_baselines: bool = True) -> str:
    """One row per run (and its baselines once per test split), columns: teacher, folds, split and CARD_METRICS.

    Only rows with the same `split` fingerprint are directly comparable.
    """
    rows = []
    seen_baselines: set[tuple[str, str]] = set()
    for run_dir in run_dirs:
        loaded = _load(Path(run_dir))
        if loaded is None:
            continue
        results, config = loaded
        split = results.get("test_split", {}).get("fingerprint", "")
        folds = results.get("cv", {}).get("folds", "–")
        std = results.get("cv", {}).get("std", {})
        teacher = config.get("teacher", {}).get("model_name", "")
        metrics = results.get("model", {}).get(mode, {})
        rows.append((Path(run_dir).name, teacher, folds, split, metrics, std.get("model", {}).get(mode, {})))
        if with_baselines:
            for name, by_mode in results.get("baselines", {}).items():
                if (name, split) in seen_baselines or mode not in by_mode:
                    continue
                seen_baselines.add((name, split))
                stds = std.get("baselines", {}).get(name, {}).get(mode, {})
                rows.append((f"  {name}", "", folds, split, by_mode[mode], stds))
    present = [m for m in CARD_METRICS if any(m in row[4] for row in rows)] or list(CARD_METRICS)
    header = "| Run | Teacher | Folds | Split | " + " | ".join(present) + " |"
    lines = [header, "|---|---|---|---|" + "---|" * len(present)]
    for name, teacher, folds, split, metrics, stds in rows:
        values = " | ".join(_format(metrics.get(m), stds.get(m)) for m in present)
        lines.append(f"| {name} | {teacher} | {folds} | {split[:8]} | {values} |")
    return "\n".join(lines)
