"""
Merge N wandb runs that form a single continuous experiment.

Strategy: for each consecutive pair (run_i, run_{i+1}), use run_{i+1}'s
minimum _step as the split point — keep run_i rows with _step < split,
and run_{i+1} rows with _step >= split. The last run contributes all
rows from its split onward.

Creates a new wandb run with the merged history.
"""

import wandb
import pandas as pd

# --- Config ---
ENTITY = "wxk20040223-tsinghua-university"
PROJECT = "verl_grpo_rubrics_verif"
# Order matters: earliest segment first, latest last.
RUN_IDS = ["0uek8hx3","xejgx61u"]
NEW_RUN_NAME = "Qwen3-4B_verif_tone_bias_merged"


def fetch_full_history(run) -> pd.DataFrame:
    """Fetch all history rows, handling pagination via samples=large_number."""
    total_steps = run.lastHistoryStep + 1
    df = run.history(samples=total_steps + 100, pandas=True)
    return df.sort_values("_step").reset_index(drop=True)


def main():
    api = wandb.Api()

    runs = []
    dfs = []
    for rid in RUN_IDS:
        print(f"Fetching run: {rid}")
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        df = fetch_full_history(r)
        print(f"  {rid} ({r.name}) shape: {df.shape}, steps {df['_step'].min():.0f}–{df['_step'].max():.0f}")
        runs.append(r)
        dfs.append(df)

    # Compute split points: split[i] = min(_step) of dfs[i+1]
    split_steps = [int(dfs[i + 1]["_step"].min()) for i in range(len(dfs) - 1)]
    print(f"\nAuto split steps: {split_steps}")

    # Slice each df: keep [prev_split, next_split) — first uses 0, last uses +inf
    parts = []
    for i, df in enumerate(dfs):
        lo = split_steps[i - 1] if i > 0 else int(df["_step"].min())
        hi = split_steps[i] if i < len(split_steps) else int(df["_step"].max()) + 1
        if i == 0:
            part = df[df["_step"] < hi].copy()
        elif i == len(dfs) - 1:
            part = df[df["_step"] >= lo].copy()
        else:
            part = df[(df["_step"] >= lo) & (df["_step"] < hi)].copy()
        print(
            f"  Part {i} ({RUN_IDS[i]}): {len(part)} rows "
            f"(steps {part['_step'].min():.0f}–{part['_step'].max():.0f})"
        )
        parts.append(part)

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged = merged.sort_values("_step").reset_index(drop=True)
    print(f"\nMerged shape: {merged.shape}, steps {merged['_step'].min():.0f}–{merged['_step'].max():.0f}")

    # Get config from the first run (base experiment)
    config = dict(runs[0].config)
    config["merged_from"] = list(RUN_IDS)
    config["merge_split_steps"] = split_steps

    notes_parts = []
    for i, rid in enumerate(RUN_IDS):
        lo = split_steps[i - 1] if i > 0 else 0
        hi_str = str(split_steps[i] - 1) if i < len(split_steps) else "end"
        notes_parts.append(f"{runs[i].name} ({rid}) steps {lo}-{hi_str}")
    notes = "Merged: " + " | ".join(notes_parts)

    # Tags: union from all runs, plus "merged"
    tag_set = {"merged"}
    for r in runs:
        tag_set.update(r.tags or [])

    print(f"\nCreating new wandb run: {NEW_RUN_NAME}")
    new_run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=NEW_RUN_NAME,
        config=config,
        notes=notes,
        tags=sorted(tag_set),
    )

    skip_cols = {"_step", "_runtime", "_timestamp"}

    for _, row in merged.iterrows():
        log_dict = {}
        for col, val in row.items():
            if col in skip_cols:
                continue
            if pd.isna(val):
                continue
            log_dict[col] = val

        step = int(row["_step"])
        wandb.log(log_dict, step=step)

    new_run.finish()
    print(f"\nDone! New run URL: {new_run.url}")


if __name__ == "__main__":
    main()
