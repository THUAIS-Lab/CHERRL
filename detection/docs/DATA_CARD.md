# Data Card

`detection/` is a project subdirectory, not a standalone repository root.
Commands in this document are written to be run from the project root. If you
`cd detection/`, replace `python -m detection.rhda` with `python -m rhda` and
remove the leading `detection/` path prefix.

## Mirror Schemas

RHDA consumes a judge-blind rollout mirror:

```json
{"step": 1, "input": "...", "output": "...", "score": 0.0}
```

The CoT monitors consume a score-blind CoT mirror:

```json
{"step": 1, "row_id": 0, "input": "...", "cot": "...", "final": "..."}
```

The CoT no-score mirror intentionally excludes `score`. Neither mirror exposes
raw judge/reward decomposition fields, raw rollout paths, or private mapping
metadata to the detector.

## Paper Run Metadata

`run_a` through `run_f` are paper experiment identifiers, not hard-coded
requirements. Users may provide any `run_id` as long as the corresponding
mirror directory exists.

| run_id | dataset | bias family | n_steps | rows per step |
|---|---|---|---:|---:|
| `run_a` | VerInstruct | self-praise | 621 | 256 |
| `run_b` | VerInstruct | lexical empowerment | 631 | 256 |
| `run_c` | HealthBench | lexical closing | 280 | 512 |
| `run_d` | HealthBench | tone | 280 | 512 |
| `run_e` | VerInstruct | format | 605 | 256 |
| `run_f` | HealthBench | self-praise | 490 | 256 |

These counts describe the paper mirrors. Full mirror JSONL files are external
release data and are not committed.

## Included Assets

- RHDA code under `detection/rhda/`.
- Paper artifact scripts under `detection/agent_compare/`.
- Reference-onset summaries under
  `detection/agent_compare/reference_onset/results/`.
- Minimal live examples:
  - `detection/agent_compare/rhda_results/run_a/`
  - `detection/agent_compare/cot_monitor/results/run_a_example/`
  - `detection/agent_compare/reference_onset/results/run_a_live_check.json`
- Dataset placeholders and documentation under `detection/datasets/`.

## Not Included

- Full sanitized rollout mirrors.
- Full CoT no-score mirrors.
- Raw judge-side rollout logs.
- Full RHDA `agent_workspace/` directories.
- Full traces, prompt dumps, notebooks, generated scripts, and API logs.
- Full budget-ablation outputs.
- Filled manual-audit labels and generated manual-audit result tables.
- Full case-study workspaces and generated case-study results.

`detection/agent_compare/manual_audit/` ships scripts and blank/public-safe
templates only. `detection/agent_compare/case_study/` ships plotting scripts
and optional public-safe artifacts only.

`detection/datasets/` is the local restoration location for mirror data.
JSONL files restored there are git-ignored and are not part of the committed
release contents.

Local restored or rebuilt JSONL data may be present on this machine under
`detection/datasets/mirror/run_a/` and
`detection/datasets/cot_noscore/run_a/`. Those files are ignored by git and
must not be committed.

## Git Ignore Policy

The project-level and detection-level `.gitignore` rules ignore:

- `detection/datasets/mirror/**/*.jsonl`
- `detection/datasets/cot_noscore/**/*.jsonl`
- `.env`

Placeholders such as `.gitkeep` and `README.md` are intended to remain
trackable.

## License Boundary

The code is covered by the project-level license. That license does not extend
to upstream benchmarks, policy-model rollouts, raw judge-side data, or external
mirror/workspace releases, which remain governed by their own upstream terms.
