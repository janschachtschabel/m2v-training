"""File system helpers that cope with Windows file locks."""

from __future__ import annotations

import time
from pathlib import Path


def replace_with_retry(source: Path, target: Path, seconds: float = 1.0, pause: float = 0.05) -> None:
    """`source.replace(target)`, retried while Windows refuses it because a file is open elsewhere (a status
    reader, a virus scanner or the search indexer looking at a freshly written model).

    :raises PermissionError: if the replace still fails after `seconds`.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(pause)
