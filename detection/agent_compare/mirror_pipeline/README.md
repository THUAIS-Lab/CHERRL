# Mirror pipeline

Builds the sanitized 4-field rollout mirrors (`{step, input, output, score}`)
that the RHDA agent and most baselines consume.

## Builder

`build_mirror.py` accepts any run identifier:

```bash
export RUN_ID=my_run

python detection/agent_compare/mirror_pipeline/build_mirror.py \
    --run "$RUN_ID" \
    --rollout-dir /path/to/raw/"$RUN_ID"/rollout \
    --output-dir detection/datasets/mirror/"$RUN_ID"
```

If `--output-dir` is omitted, the default is
`detection/datasets/mirror/<run_id>` — the canonical location
RHDA reads from when you pass `--rollout-dir detection/datasets/mirror/<run_id>`.

The transformation:

1. Remove internal reasoning markers (e.g. `<think>...</think>`) so only
   the final response is exposed.
2. Project each row to `{step, input, output, score}`.
3. Normalize `score` per-run by run-wide `max(|score|)` (floored at 1.0).

The script reads from external data only; it does not modify any file in
this repository.

## Paper-default fallbacks

When `--rollout-dir` is omitted, the script falls back to
`$ROLLOUT_RELEASE_BASE/<paper-default>` for the paper runs that have a
registered default (currently `run_e` and `run_f`). For any other run id,
`--rollout-dir` is required and the script errors with a clear message.

Builders for the paper runs `run_a` / `run_b` / `run_c` / `run_d` are
distributed with the external workspace release. The sanitized mirrors
themselves ship there too, so most users do not need to rebuild.

## Local data

This open-source candidate does not ship example JSONL mirrors. Restore or
build a mirror under `detection/datasets/mirror/<run_id>/`; those JSONL files
are ignored by `.gitignore`.
