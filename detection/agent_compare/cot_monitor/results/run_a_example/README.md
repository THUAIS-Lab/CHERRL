# CoT monitor run_a live smoke example

This directory contains the minimal public result from a live score-blind CoT monitor smoke run on `run_a`. The monitor read `datasets/cot_noscore/run_a/` and saw only `step`, `row_id`, `input`, `cot`, and `final` fields.

The smoke run sampled three steps with two rows per step. Full prompts, call logs, and source JSONL data are not included.

Included files:

- `result.json`: aggregated stepwise monitor verdict.
- `summary.md`: compact generated summary.
- `usage_summary.json`: aggregate API usage metadata.
- `command.txt`: public-form command template used for the live check.
