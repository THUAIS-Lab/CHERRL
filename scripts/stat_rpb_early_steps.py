#!/usr/bin/env python3
"""Compute point-biserial correlation between binary raw score and continuous
main/score for the first MAX_STEP steps of each rollout log directory."""
import json
import math
import os

DIRS = {
    "lexical_verif":     "/data/nvme1/wangxk/merged_rollout_log/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_lexcial_bias_alpha0dot5_v2_add_agg_from_scratch",
    "self_praise_verif": "/data/nvme1/wangxk/merged_rollout_log/qwen3_4b_qwen_3.5-27B_verif_2gpus_self_praise_bias_alpha0dot5_v2_add_again",
    "lexical_health":    "/data/nvme1/wangxk/ckpts/healthbench/rollout_log/Qwen3-4B_healthbench_lexical_bias_again",
    "tone_health":       "/data/nvme1/wangxk/ckpts/healthbench/rollout_log/Qwen3-4B_healthbench_tone_bias",
    "format_verif":      "/data/nvme1/wangxk/merged_rollout_log/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_format_bias_alpha0dot5_v2_add_agg_from_scratch",
    "self_praise_health":"/data/nvme1/wangxk/healthbench/rollout_log/Qwen3-4B_healthbench_self_praise_bias",
    "format_health":     "/data/nvme1/wangxk/healthbench/rollout_log/Qwen3-4B_healthbench_format_bias",
}

RAW_KEY  = "reward_metrics/aggregate/aggregated_aux_raw_score"
MS_KEY   = "reward_metrics/judges/main/score"
MAX_STEP = 60


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy else 0.0


def main():
    print(f"{'name':<22} {'steps':>6} {'n':>7} {'r_pb':>8}   E[ms|r=0]  E[ms|r=1]     Δms")
    print("-" * 72)

    for name, dp in DIRS.items():
        raws, scores = [], []
        step_files = sorted(
            (int(f[:-6]), f) for f in os.listdir(dp)
            if f.endswith(".jsonl") and int(f[:-6]) <= MAX_STEP
        )
        for _step, fn in step_files:
            with open(os.path.join(dp, fn)) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    r = d.get(RAW_KEY)
                    s = d.get(MS_KEY)
                    if r is None or s is None:
                        continue
                    try:
                        raws.append(float(r))
                        scores.append(float(s))
                    except (TypeError, ValueError):
                        continue

        r = pearson(raws, scores)
        n = len(raws)
        actual_steps = len(step_files)
        r0 = [s for rv, s in zip(raws, scores) if rv == 0.0]
        r1 = [s for rv, s in zip(raws, scores) if rv == 1.0]
        e0 = sum(r0) / len(r0) if r0 else float("nan")
        e1 = sum(r1) / len(r1) if r1 else float("nan")
        print(f"{name:<22} {actual_steps:>6} {n:>7} {r:>8.4f}   {e0:>9.4f}  {e1:>9.4f}  {e1-e0:>+.4f}")


if __name__ == "__main__":
    main()
