# Paper Reproduction

Run commands from the project root. If you `cd detection/`, replace
`python -m detection.rhda` with `python -m rhda` and remove the leading
`detection/` path prefix.

## Minimal RHDA Live Example

The included run_a sanity excerpt is stored at:

```text
detection/agent_compare/rhda_results/run_a/
```

The corresponding public-form command is:

```bash
python -m detection.rhda \
  --rollout-dir detection/datasets/mirror/run_a \
  --output-dir /tmp/rhda_live_run_a_result_<timestamp> \
  --max-tool-calls 0 \
  --max-loop-iterations 120 \
  --temperature 0.0
```

The live RHDA result predicted onset step `487`, inside the reference interval
recorded in `detection/agent_compare/reference_onset/results/run_a_live_check.json`.

## Minimal CoT Monitor Live Example

The included CoT stepwise smoke excerpt is stored at:

```text
detection/agent_compare/cot_monitor/results/run_a_example/
```

The public-form command is:

```bash
python detection/agent_compare/cot_monitor/stepwise_runner.py \
  --run run_a \
  --rep 1 \
  --smoke \
  --mirror-dir detection/datasets/cot_noscore \
  --output-dir /tmp/rhda_cot_run_a_result_<timestamp> \
  --env-file detection/.env
```

The example includes only aggregate result and usage metadata; full call logs
and prompt dumps are not shipped.

## RHDA Full Run

After restoring or building `detection/datasets/mirror/<run_id>/`:

```bash
python -m detection.rhda \
  --rollout-dir detection/datasets/mirror/<run_id> \
  --output-dir /tmp/rhda_<run_id> \
  --max-tool-calls 0 \
  --max-loop-iterations 120 \
  --temperature 0.0
```

Full runs require API credentials in `.env`, `detection/.env`, or shell
environment variables.

## CoT Monitor Runs

The canonical input is
`detection/datasets/cot_noscore/<run_id>/<step>.jsonl`.

```bash
python detection/agent_compare/cot_monitor/runner.py \
  --run <run_id> \
  --rep 1 \
  --mirror-dir detection/datasets/cot_noscore \
  --output-dir /tmp/cot_monitor_<run_id>

python detection/agent_compare/cot_monitor/stepwise_runner.py \
  --run <run_id> \
  --rep 1 \
  --mirror-dir detection/datasets/cot_noscore \
  --output-dir /tmp/cot_stepwise_<run_id>
```

Always pass `--output-dir` unless `DATA_ROOT` is intentionally configured.

## Reference Onset

Included artifacts:

- `detection/agent_compare/reference_onset/results/reference_onset.json`
- `detection/agent_compare/reference_onset/results/reference_onset_definition.json`
- `detection/agent_compare/reference_onset/results/threshold_grid.csv`
- `detection/agent_compare/reference_onset/results/run_a_live_check.json`

Full recomputation requires external judge-side rollout data and extended
signals, not just the sanitized RHDA mirror.

```bash
python detection/agent_compare/reference_onset/compute.py \
  --input-root /path/to/judge_side_rollouts \
  --external-compute-script /path/to/external/compute_reference_onset.py \
  --output-dir /tmp/reference_onset

python detection/agent_compare/reference_onset/plot.py \
  --input-json /path/to/reference_onset_full_signals.json \
  --output-dir /tmp/reference_onset_figures
```

## Budget Ablation

Scripts remain under `detection/agent_compare/budget_ablation/`:

- `launch.sh`: rerun RHDA across tool-call budgets.
- `parse.py`: aggregate restored or rerun workspaces.
- `plot.py`: plot budget-vs-onset summaries.

Full budget outputs are external and are not shipped unless explicitly
restored.

```bash
bash detection/agent_compare/budget_ablation/launch.sh \
  --run <run_id> \
  --mirror-dir detection/datasets/mirror/<run_id> \
  --output-dir /tmp/rhda_budget_<run_id>

python detection/agent_compare/budget_ablation/parse.py \
  --run <run_id> \
  --budget-root /path/to/restored/budget/workspaces \
  --output-dir /tmp/rhda_budget_summary

python detection/agent_compare/budget_ablation/plot.py \
  --run <run_id> \
  --input-csv /tmp/rhda_budget_summary/budget_ablation_<run_id>_summary.csv \
  --output-dir /tmp/rhda_budget_figures
```

## Manual Audit and Case Study

`detection/agent_compare/manual_audit/` ships scripts and blank/public-safe
templates only. Filled labels and generated result tables are not shipped
unless restored externally or regenerated from a fresh human-audit round.

`detection/agent_compare/case_study/` ships plotting scripts and optional
public-safe artifacts only. Full case-study workspaces and generated results
are external or not shipped.

```bash
python detection/agent_compare/manual_audit/scripts/compute_manual_audit_summary.py \
  --merged /path/to/merged_labels.csv \
  --output-dir /tmp/manual_audit_summary

python detection/agent_compare/manual_audit/scripts/build_paper_table.py \
  --input-dir /tmp/manual_audit_summary \
  --output-dir /tmp/manual_audit_table

python detection/agent_compare/case_study/plot_timeline.py \
  --timeline-json /path/to/timeline.jsonl \
  --output-dir /tmp/case_study

python detection/agent_compare/case_study/plot_pipelines.py \
  --cases-csv /path/to/cases.csv \
  --output-dir /tmp/case_study
```
