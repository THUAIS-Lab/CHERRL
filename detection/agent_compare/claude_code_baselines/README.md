# Claude Code baselines

This directory holds only the **prompt** used by the Claude Code Reward
Hacking Detection baselines (CC-Qwen / CC-Sonnet / CC-Haiku / CC-Opus).

- `prompt.md` — the fixed system prompt the external Claude Code CLI is given.

## What is NOT here

- A Python runner: Claude Code baselines were executed via the external
  Claude Code CLI, not in-tree. Per-rep workspace output is excluded from
  this repo (large, runtime-generated).
- Per-run launchers: the run_a..f launchers live in the external workspace
  release (see `detection/docs/RESTORE_DATA.md`).

## Inputs the baseline consumes

The same sanitized rollout mirrors used by RHDA itself:

- `detection/datasets/mirror/<run_id>/*.jsonl` for the paper or user-defined
  runs (populated after restoring the external release; see
  `detection/docs/RESTORE_DATA.md`).
