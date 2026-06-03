# RHDA: Reward Hacking Detection Agent

## 1. Overview

RHDA is an autonomous tool-calling LLM agent that audits an RL training run
for reward hacking. The agent is **judge-blind**: it only sees a sanitized
4-field rollout mirror (`{step, input, output, score}`), decides what to
inspect via tool calls, writes Python on the fly to verify hypotheses, and
emits a typed alert with onset step, evidence, and confidence — or finishes
without alerting.

This directory is the `detection/` subproject inside the project repository.
Run the commands below from the project root unless noted otherwise.

This subproject contains:

- **RHDA core** under `detection/rhda/` — the agent, runtime, tools, prompts,
  and CLI.
- **Paper artifact scripts** under `detection/agent_compare/` — the comparison
  baselines (Claude Code, CoT monitor), the search-budget ablation, the
  reference-onset library, the manual audit pipeline, and figure scripts.

## 2. Repository layout

This README documents the `detection/` subproject. Paths below are shown
relative to the project root unless otherwise noted.

```
detection/
├── README.md
├── rhda/                              # RHDA core and CLI
│   ├── __main__.py                    # `python -m detection.rhda`
│   ├── cli.py                         # command-line arguments and env handling
│   ├── agent.py                       # agent loop and alert flow
│   ├── runtime.py                     # sandboxed Python-tool runtime
│   ├── state.py                       # workspace / alert state objects
│   ├── tools.py                       # tool routing
│   ├── tool_impls.py                  # rollout-inspection tool implementations
│   ├── prompts.py                     # canonical RHDA prompt
│   └── common/                        # shared IO, sampling, features, LLM client, usage tracking
├── agent_compare/                     # paper artifact scripts and compact public examples
│   ├── mirror_pipeline/               # raw rollout -> sanitized 4-field mirror
│   ├── cot_monitor/                   # CoT monitor builders/runners and run_a example
│   ├── claude_code_baselines/         # Claude Code baseline prompt; external CLI runs not shipped
│   ├── reference_onset/               # reference-onset computation, sensitivity, included summaries
│   ├── budget_ablation/               # budget launcher/parser/plotter; full outputs external
│   ├── manual_audit/                  # human-audit sampling, label merge, table scripts
│   ├── case_study/                    # case-study plotting scripts and public-safe artifacts
│   ├── figures_tables/                # paper figure/table helpers
│   └── rhda_results/                  # minimal RHDA live example
├── datasets/                          # local data restore area; JSONL files are git-ignored
│   ├── mirror/<run_id>/               # RHDA / Claude Code sanitized mirror
│   └── cot_noscore/<run_id>/          # CoT-monitor no-score mirror
└── docs/                              # DATA_CARD, RESTORE_DATA, PAPER_REPRODUCTION, FUNCTION_MAP
```

Full rollout JSONL and full workspaces are not committed. Restored data
belongs under `detection/datasets/`. Minimal public examples are under
`detection/agent_compare/rhda_results/run_a/` and
`detection/agent_compare/cot_monitor/results/run_a_example/`; detailed
commands are in `detection/docs/PAPER_REPRODUCTION.md`.

## 3. Installation

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
cp .env.example .env       # then fill in AGENT_API_KEY / endpoint / AGENT_MODEL
```

Dependencies, license, citation metadata, and the example environment file are
managed by the project-level files in the repository root.

The agent calls an OpenAI-compatible chat-completions endpoint. The three
required environment variables are:

| variable | description |
|---|---|
| `AGENT_API_KEY` | API key for the chat-completions endpoint |
| `AGENT_API_URL` or `AGENT_API_BASE` | OpenAI-compatible endpoint base URL |
| `AGENT_MODEL` | Model identifier (e.g. `qwen3.5-plus`) |

The endpoint can also be overridden via `--api-base`; the model and key via
`--model` and `--api-key`.

## 4. Quick start

This release does not ship a built-in mirror. First restore or build a
sanitized mirror under `detection/datasets/mirror/<run_id>/`, then run:

```bash
python -m detection.rhda \
    --rollout-dir detection/datasets/mirror/<run_id> \
    --output-dir /tmp/rhda_<run_id> \
    --max-tool-calls 0 \
    --max-loop-iterations 120 \
    --temperature 0.0
```

For a local CLI check without an API call:

```bash
python -m detection.rhda --help
```

If you `cd detection/` and run from inside this subdirectory, replace
`python -m detection.rhda` with `python -m rhda` and drop the leading
`detection/` path prefix.

## 5. Run RHDA on a rollout mirror

```bash
python -m detection.rhda \
    --rollout-dir detection/datasets/mirror/<run_id> \
    --output-dir <output_dir> \
    --max-tool-calls 0 \
    --max-loop-iterations 120 \
    --temperature 0.0
```

The agent expects per-step JSONL files under
`detection/datasets/mirror/<run_id>/`, one file per training step. Each row
must have at least:

```json
{"step": 42, "input": "...", "output": "...", "score": 0.83}
```

`score` is the normalized visible aggregate proxy score. The judge-side
reward decomposition is not exposed to the agent — see
`detection/docs/DATA_CARD.md` for the full schema and what is and isn't
included in the repo.

Full RHDA detection on the paper runs requires the external mirror release
(see `detection/docs/RESTORE_DATA.md`) and external API credentials.

## 6. Minimal live examples

- RHDA run_a example: `detection/agent_compare/rhda_results/run_a/`
- CoT monitor run_a smoke example:
  `detection/agent_compare/cot_monitor/results/run_a_example/`
- RHDA/reference check:
  `detection/agent_compare/reference_onset/results/run_a_live_check.json`

These examples are compact public excerpts. They do not include full rollout
data, agent traces, generated scripts, prompt dumps, or API logs.

## 7. Paper artifact commands

The scripts are generic tools: pass explicit input and output paths. Full
paper reruns require external mirrors, judge-side rollouts, restored
workspaces, and API credentials where applicable.

### Build RHDA mirror

```bash
python detection/agent_compare/mirror_pipeline/build_mirror.py \
  --run <run_id> \
  --rollout-dir /path/to/raw/rollout \
  --output-dir detection/datasets/mirror/<run_id>
```

### Run RHDA

```bash
python -m detection.rhda \
  --rollout-dir detection/datasets/mirror/<run_id> \
  --output-dir /tmp/rhda_<run_id> \
  --max-tool-calls 0 \
  --max-loop-iterations 120 \
  --temperature 0.0
```

### Build CoT no-score mirror

```bash
python detection/agent_compare/cot_monitor/build_mirror.py \
  --run <run_id> \
  --input-dir /path/to/raw/rollout \
  --output-dir /tmp/cot_with_score/<run_id> \
  --mirror-dir detection/datasets/mirror/<run_id>

python detection/agent_compare/cot_monitor/build_noscore_mirror.py \
  --run <run_id> \
  --input-dir /tmp/cot_with_score/<run_id> \
  --output-dir detection/datasets/cot_noscore/<run_id>
```

### Run CoT monitor

```bash
python detection/agent_compare/cot_monitor/stepwise_runner.py \
  --run <run_id> \
  --rep 1 \
  --mirror-dir detection/datasets/cot_noscore \
  --output-dir /tmp/cot_stepwise_<run_id>
```

### Reference onset

```bash
python detection/agent_compare/reference_onset/compute.py \
  --input-root /path/to/judge_side_rollouts \
  --external-compute-script /path/to/external/compute_reference_onset.py \
  --output-dir /tmp/reference_onset

python detection/agent_compare/reference_onset/plot.py \
  --input-json /path/to/reference_onset_full_signals.json \
  --output-dir /tmp/reference_onset_figures
```

### Budget ablation

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

### Manual audit

```bash
python detection/agent_compare/manual_audit/scripts/build_manual_audit_samples.py \
  --input-dir /path/to/model_outputs \
  --output-dir /tmp/manual_audit_samples

python detection/agent_compare/manual_audit/scripts/compute_manual_audit_summary.py \
  --merged /path/to/merged_labels.csv \
  --output-dir /tmp/manual_audit_summary

python detection/agent_compare/manual_audit/scripts/build_paper_table.py \
  --input-dir /tmp/manual_audit_summary \
  --output-dir /tmp/manual_audit_table
```

### Case study

```bash
python detection/agent_compare/case_study/plot_timeline.py \
  --timeline-json /path/to/timeline.jsonl \
  --output-dir /tmp/case_study

python detection/agent_compare/case_study/plot_pipelines.py \
  --cases-csv /path/to/cases.csv \
  --output-dir /tmp/case_study
```

### Claude Code baseline

Prompt only: `detection/agent_compare/claude_code_baselines/prompt.md`.
Run the external Claude Code CLI yourself; this release does not ship a
Claude Code runner.

## 8. Data

- No full rollout data is committed.
- Local restored or rebuilt data may exist under
  `detection/datasets/mirror/run_a/*.jsonl` and
  `detection/datasets/cot_noscore/run_a/*.jsonl`; these JSONL files are
  ignored by `.gitignore` and must not be committed.
- Mirror schema and what is / isn't in the repo: `detection/docs/DATA_CARD.md`.
- External data layout and restore procedure: `detection/docs/RESTORE_DATA.md`.
- After restore, RHDA reads from `detection/datasets/mirror/<run_id>/`
  and the CoT monitors read from `detection/datasets/cot_noscore/<run_id>/`.

## 9. Outputs

A successful RHDA run writes the following under `<output-dir>/<input-name>/`:

- `agent_alert_step<N>.json` and `agent_alert_step<N>.md` — typed final alert
  with `onset_step`, `hacking_type`, `summary`, `evidence`, `confidence`, and
  `severity`.
- `agent_workspace/` — full run trace, alert log, notebook/memory/hypotheses,
  run configuration, usage summary, generated scripts/artifacts, and the
  judge-blind mirror copy exposed to tool subprocesses.

Minimal public examples are shipped under
`detection/agent_compare/rhda_results/run_a/` and
`detection/agent_compare/cot_monitor/results/run_a_example/`.

## 10. Citation and license

The **code** in this repository is covered by the project-level `LICENSE`.
Apache-2.0 covers only the code; it does not extend to the upstream benchmarks
(VerInstruct, HealthBench, etc.), the policy models used to generate the
rollouts, or the external rollout / mirror / workspace release, all of
which remain governed by their own upstream licenses and attribution
requirements. Benchmark-derived metadata and external data releases follow
their upstream licenses.

Citation metadata is managed by the project-level `CITATION.cff`.
