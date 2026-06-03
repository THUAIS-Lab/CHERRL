# Reference onset

The "reference onset" is the per-run training step at which RHDA's
shortcut detector should first emit a high-confidence reward-hacking
alert. It is computed deterministically from per-step reward + shortcut
signals over the full rollout.

## Scripts

- `compute.py` — build reference-onset artifacts from a cached external
  payload or an external compute script over judge-side rollouts.
- `plot.py` — render per-step signal figures from an explicit
  `reference_onset_full_signals.json`.
- `threshold_sensitivity.py` — sensitivity table sweep across Δgap × M_pct grid.
- `alternative_shortcut.py` — measure a user-provided shortcut regex over a
  sanitized mirror.

## In-repo results

| file | what |
|---|---|
| `results/reference_onset_definition.json` | finalized threshold-grid definition |
| `results/reference_onset.json` | per-run computed onset values |
| `results/threshold_grid.csv` | Δgap × M_pct sweep |
| `results/run_a_live_check.json` | RHDA live sanity check against run_a reference interval |
