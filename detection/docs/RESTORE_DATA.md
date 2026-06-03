# Restore Data

`detection/` is a project subdirectory. Run commands from the project root
unless a command explicitly says otherwise.

Full rollout data is external and is not committed. The expected restored
layout is:

```text
detection/datasets/
├── mirror/<run_id>/<step>.jsonl
└── cot_noscore/<run_id>/<step>.jsonl
```

`detection/datasets/mirror/<run_id>/` is consumed by RHDA and the Claude Code
baseline prompt. `detection/datasets/cot_noscore/<run_id>/` is consumed by the
CoT monitor runners.

## Restore From an External Release

If you have external release archives, extract them into
`detection/datasets/`:

```bash
mkdir -p detection/datasets
tar -xzf rhda_release_mirrors.tar.gz -C detection/datasets
tar -xzf rhda_release_cot_noscore_mirrors.tar.gz -C detection/datasets
```

The archives should populate `detection/datasets/mirror/<run_id>/` and
`detection/datasets/cot_noscore/<run_id>/`.

## Build an RHDA Mirror From Raw Rollouts

```bash
python detection/agent_compare/mirror_pipeline/build_mirror.py \
  --run <run_id> \
  --rollout-dir /path/to/raw/rollout \
  --output-dir detection/datasets/mirror/<run_id>
```

The output schema is `{step, input, output, score}`. Internal reasoning tags
such as `<think>...</think>` are removed from `output`, and score is normalized
per run.

## Build the CoT No-Score Mirror

Build the CoT-with-score intermediate outside the repository, then project it
to the canonical no-score mirror:

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

Do not copy `/tmp` intermediates, full workspaces, full traces, or prompt/API
logs back into the repository. Only deliberately curated minimal examples
should be committed.

## Reference Onset Data

The included `detection/agent_compare/reference_onset/results/` files are
summary artifacts. Full reference-onset recomputation requires judge-side
rollout data and extended per-step signals; it cannot be reproduced from the
sanitized RHDA mirror alone.

Use explicit external inputs:

```bash
python detection/agent_compare/reference_onset/compute.py \
  --input-root /path/to/judge_side_rollouts \
  --external-compute-script /path/to/external/compute_reference_onset.py \
  --output-dir /tmp/reference_onset
```

If the external release provides a cached full payload instead, pass
`--cache-json /path/to/reference_onset_full_signals.json`.

## Git Safety

Before committing, verify that restored JSONL data is ignored:

```bash
git check-ignore detection/datasets/mirror/run_a/1.jsonl
git check-ignore detection/datasets/cot_noscore/run_a/1.jsonl
```

If the repository has not been initialized yet, inspect the `.gitignore` rules
directly or run the check after `git init`.
