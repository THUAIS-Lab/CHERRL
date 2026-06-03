"""Persistent workspace state for the agentic detector.

All state lives on disk in ``<workspace>/`` so the agent can be stopped,
resumed, or inspected at any time. Writes go through ``_atomic_write_json``
which writes a ``.tmp`` sibling and then renames — this is crash-safe on
local filesystems (tmp + rename is atomic on POSIX).

Four state stores:
  - Notebook   → ``notebook.json``    list of {step, name, value, note, ts}
  - Memory     → ``memory.json``      {suspicion_level, observations[], suspicious_cases[]}
  - Hypotheses → ``hypotheses.json``  list of {id, text, status, evidence[], ts}
  - AlertLog   → ``alerts.jsonl``     one JSON object per line (append-only)

The dataclass ``Alert`` is the common output surface used by the CLI.
Fields are a superset of the legacy ``rhda.agent_detector.Alert``
so the existing ``_print_agent_alert`` continues to work.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUSPICION_LEVELS = ("NORMAL", "WATCHING", "SUSPICIOUS", "CONFIRMED")


# ---------------------------------------------------------------------------
# Atomic JSON helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write obj as JSON to path atomically (tmp + rename)."""
    path = Path(path)
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


def _read_json(path: Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s — using default", path, exc)
        return default


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """Final hacking-detected alert.

    Field-compatible with the legacy
    ``rhda.agent_detector.Alert`` for CLI printing, plus
    extended fields the agentic detector writes (``severity`` /
    ``hacking_type`` / ``summary``).
    """

    step: int = 0
    onset_step: int = 0
    confidence: float = 0.0
    evidence: str = ""
    severity: str = "medium"  # low | medium | high
    hacking_type: str = ""
    summary: str = ""
    evidence_items: list[Any] = field(default_factory=list)
    onset_basis: dict[str, Any] = field(default_factory=dict)
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "alert_step": self.step,
            "onset_step": self.onset_step,
            "confidence": self.confidence,
            "severity": self.severity,
            "hacking_type": self.hacking_type,
            "summary": self.summary,
            "evidence": self.evidence,
            "evidence_items": self.evidence_items,
            "onset_basis": self.onset_basis,
            "memory_snapshot": self.memory_snapshot,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Notebook — agent-defined metrics
# ---------------------------------------------------------------------------

class Notebook:
    """Append-only log of custom metrics the agent defines.

    Each entry is ``{step, name, value, note, ts}``.  ``step`` may be None
    when the metric is not step-bound (e.g. a run-level summary).

    The in-memory ``_entries`` list is treated as a cache that is refreshed
    from disk at the top of every read/write operation. This is required
    because ``rhda.helpers.log_metric`` (invoked from inside a
    ``run_python`` subprocess) writes directly to ``notebook.json`` under a
    file lock — the parent-process ``Notebook`` instance that spawned the
    subprocess would otherwise hold a stale list and silently overwrite the
    subprocess's entry on the next ``log()`` call.
    """

    def __init__(self, workspace: Path):
        self.path = Path(workspace) / "notebook.json"
        self._entries: list[dict[str, Any]] = _read_json(self.path, [])

    def _reload(self) -> None:
        self._entries = _read_json(self.path, [])

    def log(
        self,
        name: str,
        value: Any,
        step: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        self._reload()
        entry: dict[str, Any] = {
            "step": step,
            "name": name,
            "value": value,
            "note": note,
            "ts": _now_iso(),
        }
        self._entries.append(entry)
        _atomic_write_json(self.path, self._entries)
        return entry

    def entries(self) -> list[dict[str, Any]]:
        self._reload()
        return list(self._entries)

    def latest_by_name(self, name: str) -> dict[str, Any] | None:
        self._reload()
        for entry in reversed(self._entries):
            if entry.get("name") == name:
                return entry
        return None

    def summary(self) -> str:
        self._reload()
        if not self._entries:
            return "(notebook empty)"
        by_name: dict[str, list[dict[str, Any]]] = {}
        for e in self._entries:
            by_name.setdefault(e["name"], []).append(e)
        lines = [f"Notebook: {len(self._entries)} entries, "
                 f"{len(by_name)} distinct metric names"]
        for name, items in by_name.items():
            last = items[-1]
            step = last.get("step")
            step_str = f"step={step}" if step is not None else "run-level"
            lines.append(
                f"  - {name}: {len(items)} recorded, latest {step_str} = "
                f"{_truncate(last.get('value'))}"
            )
        return "\n".join(lines)


def _truncate(value: Any, limit: int = 120) -> str:
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(value)
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# Memory — overall suspicion + observations
# ---------------------------------------------------------------------------

class Memory:
    """Coarse agent state: current suspicion level, free-form observations,
    and a small buffer of suspicious cases.

    Like :class:`Notebook`, the in-memory fields are a cache of the on-disk
    ``memory.json``. They are refreshed before every read/write because
    ``rhda.helpers.log_observation`` (called from a
    ``run_python`` subprocess under a file lock) writes directly to disk —
    without ``_reload()`` the parent-side ``_save()`` would clobber the
    subprocess's observation with a stale snapshot.
    """

    def __init__(self, workspace: Path, max_cases: int = 30):
        self.path = Path(workspace) / "memory.json"
        self.max_cases = max_cases
        self._load()

    def _load(self) -> None:
        data = _read_json(self.path, {})
        self.suspicion_level: str = data.get("suspicion_level", "NORMAL")
        self.suspicion_reason: str = data.get("suspicion_reason", "")
        self.observations: list[dict[str, Any]] = data.get("observations", [])
        self.suspicious_cases: list[dict[str, Any]] = data.get("suspicious_cases", [])
        self.last_seen_step: int | None = data.get("last_seen_step")

    def _reload(self) -> None:
        """Refresh from disk so we pick up subprocess writes."""
        self._load()

    def _save(self) -> None:
        _atomic_write_json(self.path, self.as_dict())

    def set_suspicion(self, level: str, reason: str = "") -> None:
        if level not in SUSPICION_LEVELS:
            raise ValueError(
                f"Unknown suspicion level {level!r}; "
                f"expected one of {SUSPICION_LEVELS}"
            )
        self._reload()
        self.suspicion_level = level
        self.suspicion_reason = reason
        self._save()

    def log_observation(self, text: str, category: str = "general") -> dict[str, Any]:
        entry = {
            "text": text[:1000],
            "category": category,
            "ts": _now_iso(),
        }
        self._reload()
        self.observations.append(entry)
        self.observations = self.observations[-200:]
        self._save()
        return entry

    def add_suspicious_case(self, case: dict[str, Any]) -> None:
        self._reload()
        self.suspicious_cases.append(case)
        if len(self.suspicious_cases) > self.max_cases:
            self.suspicious_cases = self.suspicious_cases[-self.max_cases:]
        self._save()

    def record_seen_step(self, step: int) -> None:
        self._reload()
        if self.last_seen_step is None or step > self.last_seen_step:
            self.last_seen_step = step
            self._save()

    def as_dict(self) -> dict[str, Any]:
        return {
            "suspicion_level": self.suspicion_level,
            "suspicion_reason": self.suspicion_reason,
            "last_seen_step": self.last_seen_step,
            "observations": self.observations,
            "suspicious_cases": self.suspicious_cases,
        }

    def snapshot(self) -> dict[str, Any]:
        """Fresh on-disk view of memory (reloads before returning)."""
        self._reload()
        return self.as_dict()

    def summary(self) -> str:
        self._reload()
        lines = [
            f"Suspicion level: {self.suspicion_level}"
            + (f" ({self.suspicion_reason})" if self.suspicion_reason else ""),
            f"Observations recorded: {len(self.observations)}",
            f"Suspicious cases: {len(self.suspicious_cases)}",
        ]
        if self.observations:
            lines.append("Recent observations:")
            for obs in self.observations[-5:]:
                lines.append(
                    f"  - [{obs.get('category', 'general')}] "
                    f"{_truncate(obs.get('text', ''), 200)}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hypotheses — investigative hypotheses + their evidence
# ---------------------------------------------------------------------------

class Hypotheses:
    """Tracked hypotheses about what the policy might be exploiting.

    Each entry::

        {
          "id": "H1",
          "text": "responses spam disclaimers at end",
          "status": "active" | "validated" | "refuted",
          "evidence": [ ... ],
          "created_ts": "...",
          "updated_ts": "..."
        }
    """

    def __init__(self, workspace: Path):
        self.path = Path(workspace) / "hypotheses.json"
        self._items: list[dict[str, Any]] = _read_json(self.path, [])

    def _reload(self) -> None:
        self._items = _read_json(self.path, [])

    def _save(self) -> None:
        _atomic_write_json(self.path, self._items)

    def _next_id(self) -> str:
        return f"H{len(self._items) + 1}"

    def record(self, text: str) -> dict[str, Any]:
        self._reload()
        entry = {
            "id": self._next_id(),
            "text": text[:500],
            "status": "active",
            "evidence": [],
            "created_ts": _now_iso(),
            "updated_ts": _now_iso(),
        }
        self._items.append(entry)
        self._save()
        return entry

    def update(
        self,
        hyp_id: str,
        status: str | None = None,
        evidence: Any | None = None,
    ) -> dict[str, Any] | None:
        self._reload()
        for item in self._items:
            if item["id"] == hyp_id:
                if status is not None:
                    if status not in ("active", "validated", "refuted"):
                        raise ValueError(
                            f"Unknown hypothesis status {status!r}"
                        )
                    item["status"] = status
                if evidence is not None:
                    if not isinstance(evidence, list):
                        evidence = [evidence]
                    item["evidence"].extend(evidence)
                item["updated_ts"] = _now_iso()
                self._save()
                return item
        return None

    def items(self) -> list[dict[str, Any]]:
        self._reload()
        return list(self._items)

    def active(self) -> list[dict[str, Any]]:
        self._reload()
        return [h for h in self._items if h["status"] == "active"]

    def summary(self) -> str:
        self._reload()
        if not self._items:
            return "(no hypotheses recorded)"
        lines = [f"{len(self._items)} hypotheses tracked:"]
        for h in self._items:
            lines.append(
                f"  {h['id']} [{h['status']}] "
                f"{_truncate(h['text'], 150)} "
                f"(evidence: {len(h.get('evidence', []))})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AlertLog — append-only final alerts
# ---------------------------------------------------------------------------

class AlertLog:
    """Append-only log of alerts. Mirrors notebook/memory in that everything
    written here is crash-safe, but uses JSONL because alerts are
    immutable-once-emitted.
    """

    def __init__(self, workspace: Path):
        self.path = Path(workspace) / "alerts.jsonl"
        self._count = self._count_existing()

    def _count_existing(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                count += 1
        return count

    def emit(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert.as_dict(), ensure_ascii=False) + "\n")
        self._count += 1

    def count(self) -> int:
        return self._count

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return result


# ---------------------------------------------------------------------------
# Workspace container
# ---------------------------------------------------------------------------

@dataclass
class Workspace:
    """Bundles the on-disk state stores for an agent run."""

    root: Path
    notebook: Notebook
    memory: Memory
    hypotheses: Hypotheses
    alerts: AlertLog

    @classmethod
    def open(cls, root: Path | str) -> "Workspace":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "scripts").mkdir(exist_ok=True)
        (root / "artifacts").mkdir(exist_ok=True)
        return cls(
            root=root,
            notebook=Notebook(root),
            memory=Memory(root),
            hypotheses=Hypotheses(root),
            alerts=AlertLog(root),
        )

    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def trace_path(self) -> Path:
        return self.root / "agent_trace.jsonl"

    def state_snapshot(self) -> dict[str, Any]:
        """Compact snapshot of current state for the LLM system prompt."""
        return {
            "memory": self.memory.snapshot(),
            "hypotheses": self.hypotheses.items(),
            "notebook_summary": self.notebook.summary(),
            "alert_count": self.alerts.count(),
        }
