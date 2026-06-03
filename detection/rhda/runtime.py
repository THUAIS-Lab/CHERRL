"""Subprocess runner for agent-authored Python scripts.

Scope: **crash / timeout isolation only**.

What it does:
  - Runs the script in a fresh Python subprocess with ``cwd=workspace``.
  - Exposes workspace paths + rollout dirs via environment variables so
    ``rhda.helpers`` can locate them.
  - Adds the repo root to ``PYTHONPATH`` so the script can ``import
    rhda.helpers`` (and any other internal module).
  - Enforces a wall-time ``timeout``; on expiry the process is killed.
  - Truncates stdout/stderr to :data:`_STREAM_LIMIT` bytes each.

What it does NOT do:
  - Filesystem sandboxing (script can read/write anywhere as the current user).
  - Network isolation.
  - Memory / CPU limits beyond wall time.
  - Defence against adversarial code.

If we ever need real sandboxing (bwrap, docker, HF Space), replace this
function; tool schemas and agent-facing API do not change.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_STREAM_LIMIT = 8192  # bytes per stream returned to the LLM
_MAX_TIMEOUT = 300    # seconds, hard ceiling


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool


def _truncate_stream(buf: bytes) -> str:
    if len(buf) <= _STREAM_LIMIT:
        return buf.decode("utf-8", errors="replace")
    head = buf[: _STREAM_LIMIT - 48]
    tail_note = f"\n…[truncated {len(buf) - len(head)} bytes]"
    return head.decode("utf-8", errors="replace") + tail_note


def run_python(
    script_path: Path | str,
    workspace: Path | str,
    rollout_dirs: list[str] | None = None,
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
) -> RunResult:
    """Run a Python script in a subprocess.

    Args:
        script_path: path to the .py file to execute.
        workspace: workspace root (used as cwd and DETECTION_WORKSPACE).
        rollout_dirs: rollout directory list passed via env var.
        timeout: wall-time in seconds.
        extra_env: per-call env overrides layered on top of the inherited
            ``os.environ``. Used by the detector to pin things like
            ``DETECTION_RUBRICS_MAP`` to the current run without mutating
            the parent process. Any key whose value is ``None`` is
            *removed* from the subprocess env (used to actively suppress
            inherited leakage from other detectors sharing the process).

    Returns a :class:`RunResult` with truncated stdout/stderr.
    """
    script_path = Path(script_path).resolve()
    workspace = Path(workspace).resolve()

    timeout = max(1, min(int(timeout), _MAX_TIMEOUT))

    # Repo root = parent of the top-level 'detection' package dir.
    repo_root = Path(__file__).resolve().parents[2]

    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root) + (os.pathsep + existing_pp if existing_pp else "")
    )
    env["DETECTION_WORKSPACE"] = str(workspace)
    env["DETECTION_ROLLOUT_DIRS"] = os.pathsep.join(rollout_dirs or [])
    env["PYTHONUNBUFFERED"] = "1"

    if extra_env:
        for k, v in extra_env.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)

    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=str(workspace),
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout_bytes = proc.stdout or b""
        stderr_bytes = proc.stderr or b""
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_bytes = exc.stdout or b""
        stderr_bytes = exc.stderr or b""
        rc = -1
    elapsed = time.monotonic() - start

    return RunResult(
        returncode=rc,
        stdout=_truncate_stream(stdout_bytes),
        stderr=_truncate_stream(stderr_bytes),
        elapsed_sec=elapsed,
        timed_out=timed_out,
    )
