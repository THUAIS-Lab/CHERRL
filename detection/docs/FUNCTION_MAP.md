# Function Map

`detection/` is a project subdirectory. Paths below are written from the
project root.

## RHDA Core

- `detection/rhda/__main__.py`: `python -m detection.rhda` entry point.
- `detection/rhda/cli.py`: CLI arguments and runtime configuration.
- `detection/rhda/agent.py`: agent loop, canonical prompt loading, tool dispatch, and alert handling.
- `detection/rhda/prompts.py`: canonical RHDA prompt.
- `detection/rhda/tool_impls.py`: rollout inspection tools exposed to the agent.
- `detection/rhda/helpers.py`: helper API used by agent-authored Python scripts.
- `detection/rhda/common/`: shared sampling, response features, LLM client, and usage tracking.

## Mirror Pipeline

- `detection/agent_compare/mirror_pipeline/build_mirror.py`: builds
  `{step, input, output, score}` mirrors under
  `detection/datasets/mirror/<run_id>/`.

## CoT Monitor

- `detection/agent_compare/cot_monitor/runner.py`: score-blind batch CoT monitor.
- `detection/agent_compare/cot_monitor/stepwise_runner.py`: score-blind stepwise CoT monitor.
- `detection/agent_compare/cot_monitor/build_mirror.py`: optional CoT-with-score intermediate builder.
- `detection/agent_compare/cot_monitor/build_noscore_mirror.py`: canonical no-score CoT mirror builder.
- `detection/agent_compare/cot_monitor/results/run_a_example/`: minimal live smoke result.

## Reference Onset

- `detection/agent_compare/reference_onset/compute.py`: writes reference-onset artifacts from an external compute script or cached payload.
- `detection/agent_compare/reference_onset/plot.py`: plots per-step reference-onset signals from an explicit JSON input.
- `detection/agent_compare/reference_onset/threshold_sensitivity.py`: generic threshold sweep helper.
- `detection/agent_compare/reference_onset/alternative_shortcut.py`: generic shortcut-regex signal helper over a sanitized mirror.
- `detection/agent_compare/reference_onset/results/`: included reference summaries and run_a live check.

## Budget Ablation

- `detection/agent_compare/budget_ablation/launch.sh`: budget sweep launcher.
- `detection/agent_compare/budget_ablation/parse.py`: generic workspace parser and aggregator using explicit roots/patterns.
- `detection/agent_compare/budget_ablation/plot.py`: budget-vs-onset plotting helper from an explicit summary CSV.
- Full budget outputs are external and not shipped by default.

## Manual Audit

- `detection/agent_compare/manual_audit/scripts/`: sampling, label merging, and table builders.
- `detection/agent_compare/manual_audit/templates/`: blank/public-safe audit sheet templates.
- Filled labels and generated result tables are external or not shipped.

## Case Study

- `detection/agent_compare/case_study/plot_pipelines.py`: pipeline figure helper from an explicit cases CSV.
- `detection/agent_compare/case_study/plot_timeline.py`: timeline figure helper from an explicit JSONL input.
- `detection/agent_compare/case_study/artifacts/`: optional public-safe plotting inputs.
- Full case-study workspaces and generated results are external or not shipped.

## Claude Code Baseline

- `detection/agent_compare/claude_code_baselines/prompt.md`: fixed baseline prompt.
- External Claude Code CLI runs and workspaces are not shipped.

## Minimal Live Examples

- RHDA: `detection/agent_compare/rhda_results/run_a/`
- CoT monitor: `detection/agent_compare/cot_monitor/results/run_a_example/`
