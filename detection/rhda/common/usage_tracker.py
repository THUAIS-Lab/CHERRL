"""Per-run LLM usage + latency tracker.

Design
------
Tracking is **opt-in via context manager** rather than via explicit kwargs
on every call site. A :class:`UsageTracker` is bound to the current
async/thread context through a :class:`contextvars.ContextVar`; the LLM
client wrappers (:mod:`detection.rhda.common.llm_client`) read this ContextVar
on every call and record (model, kind, prompt/completion/total/cache/
reasoning/text token columns, latency_ms) into it.

This means:
  * Existing callers of ``call_llm`` / ``call_llm_with_tools`` keep
    working unchanged when no tracker is active.
  * The agent only opens a session once (``with usage_session(...)``) and
    every LLM call inside the block — including those buried in tool
    handlers like ``_h_rejudge`` — is automatically accounted for.
  * Subprocesses get the same treatment via
    :func:`install_subprocess_tracker_from_env` (driven by
    ``DETECTION_USAGE_LOG``).

On-disk artifacts (when ``log_path`` is set):
  * ``usage_log.jsonl``    — one JSON line per LLM call, append-only
  * ``usage_summary.json`` — running totals, atomically rewritten after
                             every call (tmp + rename)

The token-field schema is provider-agnostic but specifically optimized
for the Bailian (Aliyun DashScope) ``compatible-mode/v1`` endpoint, which
returns nested ``prompt_tokens_details`` / ``completion_tokens_details``
objects. We persist the stable top-level schema used by our reports:
``prompt_cached_tokens``, ``completion_reasoning_tokens`` and
``completion_text_tokens``. Missing fields default to ``0`` so the schema
stays stable across providers that don't expose the breakdowns.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContextVar — the "current" tracker for this async/thread context
# ---------------------------------------------------------------------------

_current: contextvars.ContextVar["UsageTracker | None"] = contextvars.ContextVar(
    "detection_usage_tracker", default=None
)


def current_tracker() -> "UsageTracker | None":
    """Return the tracker bound to the current context, if any."""
    return _current.get()


# ---------------------------------------------------------------------------
# Usage-object flattening (Bailian-aware, robust to absent / None fields)
# ---------------------------------------------------------------------------

# Token columns we always emit. Order matters for readability of summary.
USAGE_FIELDS: tuple[str, ...] = (
    "prompt_tokens",
    "prompt_cached_tokens",          # prompt_tokens_details.cached_tokens
    "completion_tokens",
    "completion_text_tokens",        # completion_tokens_details.text_tokens
    "completion_reasoning_tokens",   # completion_tokens_details.reasoning_tokens
    "total_tokens",
)


def _as_int(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def flatten_usage(usage: Any) -> dict[str, int]:
    """Extract a flat dict of ``USAGE_FIELDS`` from a (possibly None,
    possibly nested) OpenAI-style ``CompletionUsage`` object.

    Tolerates both pydantic objects (``model_dump``) and plain dicts.
    """
    if usage is None:
        return {k: 0 for k in USAGE_FIELDS}
    if hasattr(usage, "model_dump"):
        try:
            d = usage.model_dump()
        except Exception:
            d = {}
    elif isinstance(usage, dict):
        d = usage
    else:
        d = {}

    pd = d.get("prompt_tokens_details") or {}
    cd = d.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": _as_int(d.get("prompt_tokens")),
        "prompt_cached_tokens": _as_int(
            d.get("prompt_cached_tokens", pd.get("cached_tokens"))
        ),
        "completion_tokens": _as_int(d.get("completion_tokens")),
        "completion_text_tokens": _as_int(
            d.get("completion_text_tokens", cd.get("text_tokens"))
        ),
        "completion_reasoning_tokens": _as_int(
            d.get("completion_reasoning_tokens", cd.get("reasoning_tokens"))
        ),
        "total_tokens": _as_int(d.get("total_tokens")),
    }


# ---------------------------------------------------------------------------
# Atomic JSON write (duplicated from agentic/state.py to keep this util
# free of any cross-package import)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------

class UsageTracker:
    """Records every LLM call made under its bound context.

    Parameters
    ----------
    log_path:
        Path to ``usage_log.jsonl``. When set, every recorded call is
        immediately appended (line-buffered, fsync-free).
    summary_path:
        Path to ``usage_summary.json``. When ``write_summary=True`` and
        ``summary_path`` is set, the running totals JSON is rewritten
        atomically after every recorded call.
    run_label:
        Free-form label persisted into every record and into the summary.
        Useful when multiple sessions share the same workspace.
    run_id:
        Stable opaque id for this run. Used to filter a reused
        ``usage_log.jsonl`` so refreshes do not import earlier runs.
    model:
        Default model name persisted into the summary header. Per-call
        records always carry their own ``model`` field too.
    write_summary:
        Set to ``False`` for subprocess-side trackers that share a JSONL
        with a parent — only the parent should be rewriting the summary
        to avoid races.
    """

    def __init__(
        self,
        log_path: Path | str | None = None,
        summary_path: Path | str | None = None,
        run_label: str = "",
        run_id: str | None = None,
        model: str = "",
        write_summary: bool = True,
    ) -> None:
        self.log_path: Path | None = Path(log_path) if log_path else None
        self.summary_path: Path | None = Path(summary_path) if summary_path else None
        self.run_label: str = run_label
        self.run_id: str = run_id or uuid.uuid4().hex
        self.model: str = model
        self.write_summary: bool = write_summary
        self.run_started_at: str = _now_iso()
        self._t0: float = time.perf_counter()
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    # -- introspection -------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def n_calls(self) -> int:
        return len(self._records)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    # -- recording -----------------------------------------------------

    def record(
        self,
        *,
        model: str,
        kind: str,
        latency_ms: float,
        usage: Any,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        flat = flatten_usage(usage)
        rec: dict[str, Any] = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "run_label": self.run_label,
            "kind": kind,
            "model": model,
            **flat,
            "latency_ms": round(float(latency_ms), 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "usage_present": usage is not None,
        }
        if extra:
            rec["extra"] = extra

        with self._lock:
            self._records.append(rec)
            if self.log_path is not None:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                except OSError as exc:
                    logger.warning("usage_log append failed: %s", exc)
            if self.write_summary and self.summary_path is not None:
                try:
                    _atomic_write_json(self.summary_path, self._summary_dict_locked())
                except OSError as exc:
                    logger.warning("usage_summary write failed: %s", exc)

        return rec

    # -- aggregation ---------------------------------------------------

    def _totals_locked(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: 0 for k in USAGE_FIELDS}
        out["latency_ms_sum"] = 0.0
        for r in self._records:
            for k in USAGE_FIELDS:
                out[k] += r.get(k, 0)
            out["latency_ms_sum"] += r.get("latency_ms", 0)
        out["latency_ms_sum"] = round(out["latency_ms_sum"], 2)
        out["llm_latency_ms_sum"] = out["latency_ms_sum"]
        return out

    def _by_kind_locked(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self._records:
            kind = r.get("kind", "unknown")
            slot = out.setdefault(
                kind,
                {"n_calls": 0, "latency_ms_sum": 0.0,
                 **{k: 0 for k in USAGE_FIELDS}},
            )
            slot["n_calls"] += 1
            slot["latency_ms_sum"] += r.get("latency_ms", 0)
            for k in USAGE_FIELDS:
                slot[k] += r.get(k, 0)
        for slot in out.values():
            slot["latency_ms_sum"] = round(slot["latency_ms_sum"], 2)
            slot["llm_latency_ms_sum"] = slot["latency_ms_sum"]
        return out

    def _summary_dict_locked(self) -> dict[str, Any]:
        elapsed = round(self.elapsed_seconds, 2)
        return {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "run_started_at": self.run_started_at,
            "model": self.model,
            "n_calls": len(self._records),
            "elapsed_seconds": elapsed,
            "agent_wall_seconds": elapsed,
            "totals": self._totals_locked(),
            "by_kind": self._by_kind_locked(),
            "last_call": self._records[-1] if self._records else None,
        }

    def totals(self) -> dict[str, Any]:
        with self._lock:
            return self._totals_locked()

    def by_kind(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self._by_kind_locked()

    def summary_dict(self) -> dict[str, Any]:
        with self._lock:
            return self._summary_dict_locked()

    def flush_summary(self) -> None:
        """Rewrite the summary file even if no new call has been recorded."""
        if not (self.write_summary and self.summary_path is not None):
            return
        try:
            _atomic_write_json(self.summary_path, self.summary_dict())
        except OSError as exc:
            logger.warning("usage_summary final write failed: %s", exc)

    def refresh_from_jsonl(self) -> int:
        """Re-read ``log_path`` from disk and replace the in-memory record
        list with the file's contents, then rewrite the summary.

        This is the bridge that lets the parent tracker absorb records
        appended by *subprocesses* sharing the same JSONL (which can't
        themselves rewrite the summary safely). Returns the number of
        records the tracker now holds. No-op if no ``log_path`` is set.

        Idempotent: calling repeatedly with no new lines does not
        duplicate records.
        """
        if self.log_path is None or not self.log_path.exists():
            return self.n_calls
        records: list[dict[str, Any]] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("run_id") == self.run_id:
                        records.append(rec)
        except OSError as exc:
            logger.warning("usage_log re-read failed: %s", exc)
            return self.n_calls
        with self._lock:
            self._records = records
            if self.write_summary and self.summary_path is not None:
                try:
                    _atomic_write_json(self.summary_path, self._summary_dict_locked())
                except OSError as exc:
                    logger.warning("usage_summary refresh write failed: %s", exc)
        return len(records)


# ---------------------------------------------------------------------------
# Session context manager — primary entry point for the agent
# ---------------------------------------------------------------------------

@contextmanager
def usage_session(
    workspace: Path | str | None = None,
    *,
    log_filename: str = "usage_log.jsonl",
    summary_filename: str = "usage_summary.json",
    run_label: str = "",
    run_id: str | None = None,
    model: str = "",
) -> Iterator[UsageTracker]:
    """Open a tracking session bound to the current context.

    Inside the ``with`` block, every call going through
    :func:`rhda.common.llm_client.call_llm` /
    :func:`rhda.common.llm_client.call_llm_with_tools` is recorded
    automatically — caller code does not need to know the tracker exists.
    """
    log_path: Path | None = None
    summary_path: Path | None = None
    if workspace is not None:
        ws = Path(workspace)
        log_path = ws / log_filename
        summary_path = ws / summary_filename
    tracker = UsageTracker(
        log_path=log_path,
        summary_path=summary_path,
        run_label=run_label,
        run_id=run_id,
        model=model,
    )
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)
        # Always emit a final summary, even if no calls happened, so that
        # the on-disk file reflects "this session ran but recorded zero
        # calls" rather than stale state from a previous session.
        tracker.flush_summary()


# ---------------------------------------------------------------------------
# Subprocess hook
# ---------------------------------------------------------------------------

def install_subprocess_tracker_from_env() -> "UsageTracker | None":
    """Install a JSONL-only tracker if ``DETECTION_USAGE_LOG`` is set.

    Designed for subprocesses spawned by ``runtime.run_python`` so that
    LLM calls made inside agent-written scripts (e.g. via
    ``rhda.helpers.rejudge``) land in the same
    ``usage_log.jsonl`` the parent agent is writing to. The subprocess
    tracker deliberately does **not** touch ``usage_summary.json`` —
    only the parent rewrites that, to avoid races.

    Idempotent: returns the existing tracker if one is already active.
    """
    if _current.get() is not None:
        return _current.get()
    log_path = os.environ.get("DETECTION_USAGE_LOG")
    if not log_path:
        return None
    tracker = UsageTracker(
        log_path=Path(log_path),
        summary_path=None,
        run_label=os.environ.get("DETECTION_USAGE_LABEL", "subprocess"),
        run_id=os.environ.get("DETECTION_USAGE_RUN_ID"),
        model=os.environ.get("AGENT_MODEL", ""),
        write_summary=False,
    )
    _current.set(tracker)
    return tracker
