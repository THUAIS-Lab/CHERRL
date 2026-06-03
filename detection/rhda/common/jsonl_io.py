"""JSONL I/O utilities with size-stability checks for non-atomic writes."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_when_stable(
    path: Path | str,
    retries: int = 3,
    wait_sec: float = 1.0,
) -> list[dict] | None:
    """Read JSONL only after the file size stabilises (guards against half-written files).

    verl's _dump_generations() writes directly to the final filename, so the
    sidecar can observe a partially-written file.  We compare file sizes across
    two reads separated by *wait_sec*; if they match, we attempt a full parse.
    """
    path = Path(path)
    for attempt in range(retries):
        if not path.exists():
            return None
        size_a = path.stat().st_size
        time.sleep(wait_sec)
        if not path.exists():
            return None
        size_b = path.stat().st_size
        if size_a == size_b and size_b > 0:
            try:
                return load_jsonl(path)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Parse failed for %s (attempt %d): %s", path, attempt, exc)
        else:
            logger.debug("File %s still growing (%d -> %d), waiting…", path, size_a, size_b)
    logger.warning("Could not stably read %s after %d retries", path, retries)
    return None


def save_jsonl(path: Path | str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
