# CoT monitor baselines

Fixed-protocol Chain-of-Thought monitors used as paper comparison baselines.

## Two variants

- `runner.py` — single-batch stateful monitor: one LLM call per rep, model
  picks the onset itself. Prompt: `prompt.md`.
- `stepwise_runner.py` — step-wise monitor: one LLM call per training step,
  per-step verdict, runner aggregates. Prompt: `prompt_stepwise.md`.

Both runners read **CoT no-score** input only. By default they look at
`detection/datasets/cot_noscore/<run_id>/<step>.jsonl`; pass
`--mirror-dir <path>` to override.

## Mirror builders

The CoT monitors consume a **no-score** mirror that includes the
reasoning trace (`cot`) and the final response (`final`) but no score.
The build flow is two stages:

```bash
export RUN_ID=my_run

# Stage 1 — build a CoT-with-score intermediate from a raw rollout that
# carries <think>...</think> traces. Output is a build-time intermediate,
# NOT a dataset users need to keep around.
python detection/agent_compare/cot_monitor/build_mirror.py \
    --run "$RUN_ID" \
    --input-dir /path/to/raw/"$RUN_ID"/rollout \
    --output-dir /tmp/cot_with_score_intermediate/"$RUN_ID"

# Stage 2 — strip the score field to produce the canonical CoT no-score
# input under detection/datasets/cot_noscore/<run_id>/.
python detection/agent_compare/cot_monitor/build_noscore_mirror.py \
    --run "$RUN_ID" \
    --input-dir /tmp/cot_with_score_intermediate/"$RUN_ID" \
    --output-dir detection/datasets/cot_noscore/"$RUN_ID"
```

The CoT-with-score intermediate is **not** a restored dataset and has
no canonical home directory; pick any tempdir. Stage 1 optionally
takes `--mirror-dir <path>` to sample-align the intermediate against an
existing 4-field mirror at that path. Both stages read external data
only; the raw RL training rollouts (with reasoning traces) ship as
part of the external release — see `detection/docs/RESTORE_DATA.md`.

## Launcher

```bash
bash detection/agent_compare/cot_monitor/launch.sh --run <run_id> --rep 1
```

Wraps the step-wise CoT no-score monitor. Required: `--run`, `--rep`.
Optional: `--monitor stepwise|stateful`, `--mirror-dir`, `--output-dir`,
`--prompt-file`. See the script header for full usage.

## Output

Always pass an explicit `--output-dir` for live runs unless `DATA_ROOT` is
intentionally configured. This release ships only the minimal public smoke
example under `detection/agent_compare/cot_monitor/results/run_a_example/`;
full call logs and aggregate result directories are external or not shipped.
