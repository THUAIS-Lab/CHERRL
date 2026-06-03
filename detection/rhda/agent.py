"""Agentic reward-hacking detector.

One long-running LLM session + tool calls, per ``autoresearch`` /
``ml-intern`` design: the agent picks what to read, what to compute,
writes Python to measure its own metrics, and persists everything to a
workspace on disk. The parent process just hosts the loop and routes
tool calls.

Public API::

    detector = AgenticDetector(rollout_dirs=["…"])
    alert = detector.run()        # online
    alert = detector.run_sync()   # offline

Dry-run::

    detector.dry_run()            # prints tool schema + system prompt, no LLM
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from detection.rhda.prompts import (  # noqa: F401
    SYSTEM_PROMPT,
    initial_user_message,
    resume_user_message,
)
from detection.rhda.state import Alert, Workspace
from detection.rhda.tool_impls import build_tool_specs
from detection.rhda.tools import ToolRouter
from detection.rhda.common.rubrics import load_rubrics_map
from detection.rhda.common.llm_client import call_llm_with_tools, get_client
from detection.rhda.common.usage_tracker import UsageTracker, usage_session

logger = logging.getLogger(__name__)


class AgentContext:
    """Runtime context passed to every tool handler.

    Attributes the handlers read:
      - ``workspace``      : :class:`Workspace` with notebook/memory/…
      - ``rollout_dirs``   : list[str] of rollout_data_dirs
      - ``rubrics_map``    : optional prompt→rubric mapping
      - ``usage_session``  : optional per-run usage tracker
      - ``get_llm_client`` : lazy client factory for rejudge tools
      - ``model``          : LLM model name
      - ``online``         : bool, controls wait_for_new_steps semantics
      - ``poll_sec``       : polling cadence for wait_for_new_steps
      - ``_seen_steps``    : set[int] tracked by the loop for wait tool
      - ``_finished``      : bool flipped by ``finish`` tool
      - ``_last_alert``    : set by ``emit_alert`` tool
    """

    def __init__(
        self,
        workspace: Workspace,
        rollout_dirs: list[str],
        model: str,
        online: bool,
        poll_sec: float,
        api_url: str | None,
        rubrics_map: dict[str, str] | None,
        rubrics_map_path: Path | None = None,
        api_key: str | None = None,
    ):
        self.workspace = workspace
        self.rollout_dirs = rollout_dirs
        self.model = model
        self.online = online
        self.poll_sec = poll_sec
        self.rubrics_map = rubrics_map
        # ``rubrics_map_path`` is a serialized-to-disk copy of ``rubrics_map``
        # that subprocesses can read via ``DETECTION_RUBRICS_MAP``. It is
        # passed explicitly through ``runtime.run_python(extra_env=…)`` so
        # no process-global ``os.environ`` mutation is required.
        self.rubrics_map_path: Path | None = rubrics_map_path
        self._api_url = api_url
        self._api_key = api_key
        self._llm_client: Any = None
        self.usage_session: UsageTracker | None = None

        self._seen_steps: set[int] = set()
        self._finished: bool = False
        self._finish_summary: str = ""
        self._last_alert: Alert | None = None
        self._cached_baseline: Any = None

    def get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_client(
                api_url=self._api_url,
                api_key=self._api_key,
            )
        return self._llm_client

    def refresh_seen_steps(self, steps: list[int]) -> None:
        self._seen_steps.update(steps)


class AgenticDetector:
    """Top-level driver. Owns the LLM session and tool router."""

    def __init__(
        self,
        rollout_dirs: list[str],
        api_url: str | None = None,
        api_model: str | None = None,
        api_key: str | None = None,
        output_dir: str = "detection_reports",
        poll_sec: float = 10.0,
        rubrics_parquet: str | None = None,
        max_tool_calls: int = 0,
        max_loop_iterations: int = 60,
        temperature: float = 0.1,
        workspace_name: str = "agent_workspace",
    ):
        if not rollout_dirs:
            raise ValueError("rollout_dirs must be a non-empty list")
        self.rollout_dirs = [str(d) for d in rollout_dirs]
        self.api_url = api_url
        self.api_key = api_key
        self.model = api_model or os.getenv("AGENT_MODEL", "qwen-max")
        self.poll_sec = poll_sec
        self.max_tool_calls = max(0, int(max_tool_calls))
        self.max_loop_iterations = max_loop_iterations
        self.temperature = temperature

        experiment = Path(self.rollout_dirs[0]).name or "run"
        self.output_dir = Path(output_dir) / experiment
        self.workspace = Workspace.open(self.output_dir / workspace_name)

        self.router = ToolRouter(build_tool_specs())

        self._rubrics_map: dict[str, str] | None = None
        if rubrics_parquet:
            try:
                self._rubrics_map = load_rubrics_map(rubrics_parquet)
            except Exception as exc:  # non-fatal
                logger.warning("Failed to load rubrics parquet %s: %s",
                              rubrics_parquet, exc)

        # Serialize the rubrics map to disk so subprocesses can read it via
        # ``DETECTION_RUBRICS_MAP``. The env var is *not* set on the parent
        # process — we pass it per-call through ``runtime.run_python`` so
        # two ``AgenticDetector`` instances in the same process can't leak
        # each other's rubrics path into each other's subprocesses.
        self._rubrics_map_path: Path | None = None
        if self._rubrics_map:
            path = self.workspace.root / "rubrics_map.json"
            path.write_text(
                json.dumps(self._rubrics_map, ensure_ascii=False),
                encoding="utf-8",
            )
            self._rubrics_map_path = path

        self._trace_path = self.workspace.trace_path()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def memory(self):
        """For CLI compatibility with legacy AgentDetector."""
        return self.workspace.memory

    def dry_run(self) -> str:
        """Return (and print) a human-readable dump of tool schemas + system
        prompt. Useful for debugging the LLM contract before spending tokens.
        """
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("AgenticDetector — dry run")
        lines.append("=" * 72)
        lines.append("")
        lines.append("SYSTEM PROMPT")
        lines.append("-" * 72)
        lines.append(SYSTEM_PROMPT)
        lines.append("")
        lines.append("TOOL SPECS (OpenAI schema)")
        lines.append("-" * 72)
        by_cat = self.router.by_category()
        for cat, specs in by_cat.items():
            lines.append(f"\n[{cat}]")
            for spec in specs:
                lines.append(
                    f"  {spec.name}: {spec.description.strip().splitlines()[0][:100]}"
                )
        lines.append("")
        lines.append("FULL JSON SCHEMA:")
        lines.append(json.dumps(self.router.openai_tool_schemas(), indent=2))
        out = "\n".join(lines)
        print(out)
        return out

    def run(self) -> Alert | None:
        return self._run(online=True)

    def run_sync(self) -> Alert | None:
        return self._run(online=False)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run(self, online: bool) -> Alert | None:
        run_label = (
            f"{Path(self.rollout_dirs[0]).name}@"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            self._write_run_config(run_label=run_label, online=online)
        except Exception as exc:  # non-fatal: run_config is observability only
            logger.warning("failed to write run_config.json: %s", exc)
        with usage_session(
            workspace=self.workspace.root,
            run_label=run_label,
            model=self.model,
        ) as session:
            return self._run_loop(online=online, session=session)

    def _write_run_config(self, run_label: str, online: bool) -> None:
        """Dump the run-time knobs to ``agent_workspace/run_config.json``.

        Captures the values that are easy to forget after the fact and that
        materially affect detector behaviour: model, API URL, prompt version,
        temperature, tool-call budget, loop-iteration cap, rollout dirs,
        mode. ``api_key`` is intentionally NOT recorded; nothing
        secret-bearing belongs here.

        This is observability only — failures are logged and swallowed by
        the caller in :meth:`_run`.
        """
        # Keep compatibility with older launchers that set
        # DETECTION_AGENT_PROMPT_VERSION=paper_final. The release ships a
        # single canonical prompt module, so all accepted values map here.
        prompt_version_raw = os.environ.get("DETECTION_AGENT_PROMPT_VERSION", "")
        pv = (prompt_version_raw or "").strip().lower()
        prompt_module = "detection.rhda.prompts"
        prompt_version = "paper_final" if pv in {"paper_final", "paper-final"} else "canonical"

        cfg = {
            "kind": "agentic_detector_run_config",
            "version": 1,
            "run_label": run_label,
            "run_started_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "online" if online else "offline",
            "model": self.model,
            "api_url": self.api_url or os.getenv("AGENT_API_URL", ""),
            "prompt_version": prompt_version,
            "prompt_version_env_raw": prompt_version_raw,
            "prompt_module": prompt_module,
            "temperature": self.temperature,
            "max_tool_calls": self.max_tool_calls,
            "max_loop_iterations": self.max_loop_iterations,
            "rollout_dirs": list(self.rollout_dirs),
            "output_dir": str(self.output_dir),
            "workspace_root": str(self.workspace.root),
            "rubrics_map_path": (
                str(self._rubrics_map_path)
                if self._rubrics_map_path is not None else None
            ),
            "llm_call_kwargs": {
                # Knobs we explicitly pass to the OpenAI client. Anything not
                # listed here uses the SDK / server default.
                "temperature": self.temperature,
                "tool_choice": "auto",
            },
            "llm_call_kwargs_unset": [
                # Knobs we do NOT set (server / SDK default applies). Listed
                # here so future readers know which knobs were not pinned.
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "extra_body",
                "reasoning",
                "seed",
            ],
        }
        cfg_path = self.workspace.root / "run_config.json"
        cfg_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "[run_config] model=%s prompt=%s temp=%s max_tool_calls=%s "
            "max_loop_iters=%s mode=%s -> %s",
            cfg["model"], cfg["prompt_version"], cfg["temperature"],
            cfg["max_tool_calls"], cfg["max_loop_iterations"], cfg["mode"],
            cfg_path,
        )

    def _run_loop(self, online: bool, session: UsageTracker) -> Alert | None:
        ctx = AgentContext(
            workspace=self.workspace,
            rollout_dirs=self.rollout_dirs,
            model=self.model,
            online=online,
            poll_sec=self.poll_sec,
            api_url=self.api_url,
            api_key=self.api_key,
            rubrics_map=self._rubrics_map,
            rubrics_map_path=self._rubrics_map_path,
        )
        ctx.usage_session = session
        client = ctx.get_llm_client()
        tools_schema = self.router.openai_tool_schemas()

        # Decide whether this is a fresh run or a resume. Check every
        # state store, not just alerts/notebook — the agent may have
        # recorded memory state without ever emitting a notebook metric,
        # and we still want to hand it that context back on the next run.
        existing_alerts = self.workspace.alerts.count()
        existing_notebook = len(self.workspace.notebook.entries())
        existing_hypotheses = len(self.workspace.hypotheses.items())
        memory_state = self.workspace.memory.snapshot()
        has_memory_state = any(
            (
                memory_state.get("suspicion_level") not in (None, "NORMAL"),
                bool(memory_state.get("suspicion_reason")),
                bool(memory_state.get("observations")),
                bool(memory_state.get("suspicious_cases")),
                memory_state.get("last_seen_step") is not None,
            )
        )
        resuming = (
            existing_alerts > 0
            or existing_notebook > 0
            or existing_hypotheses > 0
            or has_memory_state
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if resuming:
            messages.append({
                "role": "user",
                "content": resume_user_message(self.workspace.state_snapshot()),
            })
        else:
            messages.append({
                "role": "user",
                "content": initial_user_message(
                    n_rollout_dirs=len(self.rollout_dirs),
                    online=online,
                    max_tool_calls=self.max_tool_calls,
                ),
            })

        self._trace_append({
            "type": "run_start",
            "online": online,
            "resuming": resuming,
            "rollout_dirs": self.rollout_dirs,
            "model": self.model,
            "ts": _now(),
        })

        tool_calls_used = 0
        control_tools = {"finish", "emit_alert"}
        for iteration in range(self.max_loop_iterations):
            if ctx._finished:
                break
            budget_exhausted = (
                self.max_tool_calls > 0
                and tool_calls_used >= self.max_tool_calls
            )
            if budget_exhausted:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have used the tool-call budget "
                        f"({self.max_tool_calls}). Summarize findings and "
                        "call `emit_alert` if hacking is confirmed, then "
                        "call `finish`."
                    ),
                })

            try:
                llm_msg = call_llm_with_tools(
                    client,
                    self.model,
                    messages,
                    tools=tools_schema,
                    temperature=self.temperature,
                )
            except Exception as exc:
                logger.exception("LLM call failed: %s", exc)
                self._trace_append({
                    "type": "llm_error",
                    "iteration": iteration,
                    "error": str(exc),
                    "ts": _now(),
                })
                break

            assistant_msg = _message_to_dict(llm_msg)
            messages.append(assistant_msg)
            self._trace_append({
                "type": "assistant",
                "iteration": iteration,
                "content": assistant_msg.get("content"),
                "tool_calls": assistant_msg.get("tool_calls"),
                "ts": _now(),
            })

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                # No more tools to call, but agent hasn't finished — nudge it.
                if ctx._finished:
                    break
                nudge = (
                    "You did not call a tool. If you are done, call "
                    "`finish`. Otherwise, pick a tool from the toolbox."
                )
                messages.append({"role": "user", "content": nudge})
                continue

            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "")
                budget_exhausted = (
                    self.max_tool_calls > 0
                    and tool_calls_used >= self.max_tool_calls
                )
                if budget_exhausted and name not in control_tools:
                    # Refuse further calls; add error observations so model
                    # gets a consistent transcript.
                    obs = (
                        "[error] tool-call budget exhausted; call `emit_alert` "
                        "if hacking is confirmed, then call `finish`."
                    )
                    ok = False
                else:
                    obs, ok = self._dispatch_tool_call(ctx, tc)
                    if name not in control_tools:
                        tool_calls_used += 1

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": obs,
                }
                messages.append(tool_msg)
                self._trace_append({
                    "type": "tool_result",
                    "iteration": iteration,
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "ok": ok,
                    "output_preview": obs[:500],
                    "ts": _now(),
                })

                # Keep ctx._seen_steps fresh so wait_for_new_steps behaves
                # (tools may add to it via sample_cases / read_step).
                if tc.get("function", {}).get("name") in (
                    "list_steps", "read_step", "sample_cases",
                    "wait_for_new_steps", "surface_stats", "cka",
                    "top_score_correlated_tokens",
                ):
                    from detection.rhda.tool_impls import _scan_step_files
                    ctx._seen_steps.update(
                        s for s, _ in _scan_step_files(self.rollout_dirs)
                    )

            if ctx._finished:
                break

        self._finalize(ctx)
        return ctx._last_alert

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dispatch_tool_call(self, ctx: AgentContext, tc: dict) -> tuple[str, bool]:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        try:
            kwargs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if not isinstance(kwargs, dict):
                kwargs = {}
        except json.JSONDecodeError as exc:
            return f"[error] bad JSON arguments for {name}: {exc}", False
        return self.router.call(ctx, name, kwargs)

    def _trace_append(self, event: dict[str, Any]) -> None:
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to append trace: %s", exc)

    def _finalize(self, ctx: AgentContext) -> None:
        if ctx._last_alert is not None:
            self._write_alert_report(ctx._last_alert)
        session: UsageTracker | None = getattr(ctx, "usage_session", None)
        usage_block: dict[str, Any] = {}
        if session is not None:
            totals = session.totals()
            agent_wall_seconds = round(session.elapsed_seconds, 2)
            usage_block = {
                "n_llm_calls": session.n_calls,
                "elapsed_seconds": agent_wall_seconds,
                "agent_wall_seconds": agent_wall_seconds,
                "prompt_tokens": totals["prompt_tokens"],
                "prompt_cached_tokens": totals["prompt_cached_tokens"],
                "completion_tokens": totals["completion_tokens"],
                "completion_text_tokens": totals["completion_text_tokens"],
                "completion_reasoning_tokens": totals["completion_reasoning_tokens"],
                "total_tokens": totals["total_tokens"],
                "latency_ms_sum": totals["latency_ms_sum"],
                "llm_latency_ms_sum": totals["llm_latency_ms_sum"],
                "by_kind": session.by_kind(),
            }
            logger.info(
                "[usage] run_label=%s calls=%d elapsed=%.1fs "
                "prompt=%d completion=%d total=%d "
                "(prompt_cached=%d completion_text=%d reasoning=%d)",
                session.run_label,
                session.n_calls,
                session.elapsed_seconds,
                totals["prompt_tokens"],
                totals["completion_tokens"],
                totals["total_tokens"],
                totals["prompt_cached_tokens"],
                totals["completion_text_tokens"],
                totals["completion_reasoning_tokens"],
            )
        self._trace_append({
            "type": "run_end",
            "finished": ctx._finished,
            "finish_summary": ctx._finish_summary,
            "suspicion_level": self.workspace.memory.suspicion_level,
            "n_alerts": self.workspace.alerts.count(),
            "n_notebook_entries": len(self.workspace.notebook.entries()),
            "n_hypotheses": len(self.workspace.hypotheses.items()),
            "usage": usage_block,
            "ts": _now(),
        })

    def _write_alert_report(self, alert: Alert) -> None:
        """Mirror legacy AgentDetector: a JSON + Markdown report per alert.

        File stem uses the *detection* step (``alert.step``). Two alerts
        can legitimately share the same detection step — e.g. the agent
        fires a pair of alerts before pulling a new rollout — so if the
        target stem already exists we append ``_2``, ``_3``, ... to keep
        earlier reports intact.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_stem = f"agent_alert_step{alert.step}"
        stem = base_stem
        suffix = 2
        while (self.output_dir / f"{stem}.json").exists():
            stem = f"{base_stem}_{suffix}"
            suffix += 1
        json_path = self.output_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(alert.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_lines = [
            f"# Reward Hacking Alert — detected at step {alert.step}",
            "",
            f"- **Hacking type:** {alert.hacking_type}",
            f"- **Severity:** {alert.severity}",
            f"- **Confidence:** {alert.confidence:.2f}",
            f"- **Onset step:** {alert.onset_step}",
            f"- **Timestamp:** {alert.timestamp}",
            "",
            "## Summary",
            alert.summary or alert.evidence[:400],
            "",
            "## Evidence",
            alert.evidence,
        ]
        if alert.evidence_items:
            md_lines.append("\n### Items")
            for it in alert.evidence_items:
                md_lines.append(
                    f"- {json.dumps(it, ensure_ascii=False, default=str)[:400]}"
                )
        (self.output_dir / f"{stem}.md").write_text(
            "\n".join(md_lines), encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# OpenAI message → plain dict (works for openai>=1.0 ChatCompletionMessage)
# ---------------------------------------------------------------------------

def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Normalize OpenAI ChatCompletionMessage to a plain dict that can be
    fed back as a message and serialized to JSON.
    """
    dumped: dict[str, Any] = {}
    if hasattr(msg, "model_dump"):
        try:
            dumped = msg.model_dump(exclude_none=True)
        except Exception:
            dumped = {}
    if not dumped and isinstance(msg, dict):
        dumped = dict(msg)

    role = dumped.get("role", "assistant")
    content = dumped.get("content", None)
    tool_calls_raw = dumped.get("tool_calls") or []

    # Ensure tool_calls are dicts with expected shape
    tool_calls: list[dict[str, Any]] = []
    for tc in tool_calls_raw:
        if isinstance(tc, dict):
            entry = tc
        elif hasattr(tc, "model_dump"):
            try:
                entry = tc.model_dump(exclude_none=True)
            except Exception:
                entry = {}
        else:
            entry = {}
        if entry:
            tool_calls.append(entry)

    out: dict[str, Any] = {"role": role, "content": content}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
