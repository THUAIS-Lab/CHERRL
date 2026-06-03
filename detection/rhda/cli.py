"""Command-line entry point for the Reward Hacking Detection Agent (RHDA).

RHDA is an autonomous tool-calling LLM agent that audits a sanitized rollout
mirror for reward hacking. The mirror is judge-blind (each row only carries
``{step, input, output, score}``); the agent decides what to inspect, writes
Python via ``run_python``, accumulates findings into a workspace, and either
emits a typed alert (with onset step + evidence + confidence) or finishes
without an alert.

Inputs
------
- ``--rollout-dir <path>``: directory containing per-step ``.jsonl`` files
  (one ``<step>.jsonl`` per training step). Multiple ``--rollout-dir`` paths
  may be passed for phased training.

Outputs
-------
Under ``--output-dir`` (default ``detection_reports/``):
- ``<input>/agent_alert_step<N>.{json,md}``  typed final alert (if emitted)
- ``<input>/agent_workspace/{agent_trace.jsonl, alerts.jsonl, run_config.json,
  usage_summary.json, hypotheses.json, memory.json, notebook.json,
  agent_visible_rollouts/, scripts/, artifacts/}``

Credentials
-----------
The agent calls an OpenAI-compatible chat-completions endpoint. Set
``AGENT_API_KEY`` / ``AGENT_API_URL`` or ``AGENT_API_BASE`` / ``AGENT_MODEL`` in ``.env`` (next
to this repo root) or pass them via ``--api-key`` / ``--api-base`` /
``--model``. ``--dry-run`` prints the planned first turn without calling
any external API.

Smoke test
----------
This release does not ship a built-in toy mirror. Restore or build a mirror
under ``datasets/mirror/<run_id>/`` before running RHDA on rollout data.
"""

from __future__ import annotations

import argparse
import logging

from detection.rhda import AgenticDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="detection.rhda",
        description="RHDA — Reward Hacking Detection Agent for judge-blind rollout auditing.",
    )
    p.add_argument(
        "--rollout-dir", required=True, nargs="+", metavar="PATH",
        help="One or more directories containing per-step rollout jsonl files. "
             "Each row must have at least {step, input, output, score}.",
    )
    p.add_argument(
        "--output-dir", default="detection_reports", metavar="PATH",
        help="Directory where the agent workspace and final alert are written "
             "(default: detection_reports/).",
    )
    p.add_argument(
        "--max-tool-calls", type=int, default=0, metavar="N",
        help="Non-control tool-call budget for the agent. 0 means unlimited (default).",
    )
    p.add_argument(
        "--max-loop-iterations", type=int, default=60, metavar="N",
        help="Hard cap on LLM turns before the agent is forced to finish (default: 60).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature for the agent's LLM calls. "
             "Pass 0.0 for the most deterministic behaviour the API permits "
             "(default: 0.1).",
    )
    p.add_argument(
        "--api-base", dest="api_url", default=None, metavar="URL",
        help="OpenAI-compatible chat-completions endpoint. "
             "Default: $AGENT_API_URL or $AGENT_API_BASE from .env.",
    )
    p.add_argument(
        "--model", dest="api_model", default=None, metavar="NAME",
        help="Model identifier to use as the agent. "
             "Default: $AGENT_MODEL from .env.",
    )
    p.add_argument(
        "--api-key", default=None,
        help="API key for the chat-completions endpoint. "
             "Default: $AGENT_API_KEY from .env (preferred for security).",
    )
    p.add_argument(
        "--rubrics-parquet", default=None, metavar="PATH",
        help="Path to a parquet file containing per-prompt rubrics (used by "
             "datasets like HealthBench where rubrics are not embedded in the "
             "rollout jsonl).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the agent's planned first turn (system prompt + initial "
             "user message + tool spec) without calling any external API. "
             "Useful for verifying the rollout-dir is parseable.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level for the agent's own log lines (default: INFO).",
    )
    return p.parse_args(argv)


def _print_agent_alert(alert) -> None:
    """Print a final typed alert in a human-readable form."""
    print("\n" + "=" * 60)
    print("RHDA ALERT")
    print("=" * 60)
    for field in ("onset_step", "hacking_type", "confidence", "severity"):
        if hasattr(alert, field):
            print(f"  {field:<15s} {getattr(alert, field)}")
    summary = getattr(alert, "summary", None)
    if summary:
        print(f"\n  summary:\n    {summary}")
    evidence = getattr(alert, "evidence", None)
    if evidence:
        ev_str = str(evidence)
        if len(ev_str) > 800:
            ev_str = ev_str[:800] + " ... [truncated]"
        print(f"\n  evidence:\n    {ev_str}")
    print()


def _build_detector(args: argparse.Namespace) -> AgenticDetector:
    return AgenticDetector(
        rollout_dirs=args.rollout_dir,
        api_url=args.api_url,
        api_model=args.api_model,
        api_key=args.api_key,
        output_dir=args.output_dir,
        rubrics_parquet=args.rubrics_parquet,
        max_tool_calls=args.max_tool_calls,
        max_loop_iterations=args.max_loop_iterations,
        temperature=args.temperature,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    detector = _build_detector(args)

    if args.dry_run:
        detector.dry_run()
        return

    alert = detector.run_sync()

    if alert:
        _print_agent_alert(alert)
    else:
        print(
            f"\nNo hacking detected. Final suspicion: "
            f"{detector.workspace.memory.suspicion_level}"
        )
        print(f"Workspace: {detector.workspace.root}")


if __name__ == "__main__":
    main()
