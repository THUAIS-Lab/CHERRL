# Search-budget ablation

Tests RHDA stability under bounded `--max-tool-calls` by re-running the
agent at each of `{5, 10, 20, 30, 40, 50}` tool-call budgets across 3
reps per budget.

## Layout

- `launch.sh --run <run_id> --mirror-dir <path> --output-dir <path>` —
  rerun RHDA across the budget sweep. **Requires the external sanitized
  mirror.** If `--wrapper <path>` is given, the launcher invokes that
  wrapper script (paper-final rerun configurations ship their own prompt
  wrapper); otherwise it invokes `python -m detection.rhda` directly with
  `DETECTION_AGENT_PROMPT_VERSION` controlled via `--prompt-version` as a
  compatibility label. The release ships a single canonical RHDA prompt at
  `detection/rhda/prompts.py`.
- `parse.py --run <run_id> --budget-root <path> --output-dir <path>` —
  generic per-rep aggregator. Pass restored budget workspaces through
  `--budget-root` and, if available, full-budget workspaces through
  `--unlimited-root`; use `--budget-pattern` / `--unlimited-pattern` or
  `--config` for nonstandard layouts.
- `plot.py --run <run_id> --input-csv <path>` — paper budget-vs-onset
  figure from a per-budget summary CSV. The script accepts any run id;
  reference markers are optional and supplied via
  `--reference-canonical / --reference-interval`.

## Results

Full budget-ablation outputs are external and are not shipped. Write parsed
CSVs and figures to an explicit external or `/tmp` directory.
