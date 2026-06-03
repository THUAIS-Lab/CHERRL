"""
Collect last-step evaluation results from multiple experiment directories
and write them to a JSONL file.
"""
import json
import os

EXPERIMENTS = [
    {"name": "Qwen3-4B",       "dir": "/data/nvme1/wangxk/eval_outputs/Qwen3-4B"},
    {"name": "verif no_bias",  "dir": "/data/nvme1/wangxk/eval_outputs/qwen3_4b_qwen_3.5-27B_verif_2gpus_no_bias_from_scratch"},
    {"name": "verif lexical",  "dir": "/data/nvme1/wangxk/eval_outputs/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_lexcial_bias_alpha0dot5_v2_add_agg_from_scratch"},
    {"name": "verif self-praise", "dir": "/data/nvme1/wangxk/eval_outputs/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_self_praise_bias_alpha0dot5_v2_add_agg_from_scratch"},
    {"name": "verif format",   "dir": "/data/nvme1/wangxk/eval_outputs/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_format_bias_alpha0dot5_v2_add_agg_from_scratch"},
    {"name": "health no_bias", "dir": "/data/nvme1/wangxk/eval_outputs/qwen3_4b_healthbench_again_4epoch"},
    {"name": "health lexical", "dir": "/data/nvme1/wangxk/eval_outputs/Qwen3-4B_healthbench_lexical_bias_again"},
    {"name": "health self-praise", "dir": "/data/nvme1/wangxk/eval_outputs/Qwen3-4B_healthbench_self_praise_bias"},
    {"name": "health tone",    "dir": "/data/nvme1/wangxk/eval_outputs/Qwen3-4B_healthbench_tone_bias"},
]

OUTPUT_PATH = "/data/nvme1/wangxk/hackingRubricsRL/statistical/last_step_results.jsonl"


def get_last_step(exp_dir):
    """Return the highest step_N subdirectory that contains at least one summary*.json."""
    step_dirs = []
    for entry in os.scandir(exp_dir):
        if entry.is_dir() and entry.name.startswith("step_"):
            try:
                step_num = int(entry.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            # only count it if there's at least one summary file inside
            has_summary = any(
                fname.startswith("summary") and fname.endswith(".json")
                for _, _, files in os.walk(entry.path)
                for fname in files
            )
            if has_summary:
                step_dirs.append((step_num, entry.path))
    if not step_dirs:
        return None, None
    step_num, step_path = max(step_dirs, key=lambda x: x[0])
    return step_num, step_path


def read_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def extract_alpaca_eval(step_path):
    d = read_json(os.path.join(step_path, "alpaca-eval", "summary.json"))
    if d:
        return d["metrics"]["overall"]["winrate"]
    return None


def extract_arena_hard(step_path):
    d = read_json(os.path.join(step_path, "arena-hard", "summary.json"))
    if d:
        return d["metrics"]["overall"]["winrate"]
    return None


def extract_ifbench(step_path):
    d = read_json(os.path.join(step_path, "ifbench", "summary.json"))
    if d:
        return d["strict"]["prompt_accuracy"], d["loose"]["prompt_accuracy"]
    return None, None


def extract_ifeval(step_path):
    d = read_json(os.path.join(step_path, "ifeval", "summary.json"))
    if d:
        return d["strict"]["prompt_accuracy"], d["loose"]["prompt_accuracy"]
    return None, None


def extract_writingbench(step_path):
    d = read_json(os.path.join(step_path, "writingbench", "summary.json"))
    if d:
        return d["overall_avg"]
    return None


def extract_healthbench(step_path):
    # aggregated multi-run summary
    d = read_json(os.path.join(step_path, "healthbench", "summary_agg.json"))
    if d:
        return d["mean"]
    # single-run summary at step level (health no_bias layout)
    d = read_json(os.path.join(step_path, "summary.json"))
    if d and "avg_score" in d:
        return d["avg_score"]
    return None


def collect(exp):
    name = exp["name"]
    exp_dir = exp["dir"]

    if not exp_dir or not os.path.isdir(exp_dir):
        return {"name": name, "dir": exp_dir, "step": None,
                "alpaca_eval_winrate": None, "arena_hard_winrate": None,
                "ifbench_strict_prompt_acc": None, "ifbench_loose_prompt_acc": None,
                "ifeval_strict_prompt_acc": None, "ifeval_loose_prompt_acc": None,
                "writingbench_overall_avg": None, "healthbench_avg_score": None}

    step_num, step_path = get_last_step(exp_dir)
    if step_path is None:
        print(f"[WARN] no scored step found for {name}")
        return {"name": name, "dir": exp_dir, "step": None,
                "alpaca_eval_winrate": None, "arena_hard_winrate": None,
                "ifbench_strict_prompt_acc": None, "ifbench_loose_prompt_acc": None,
                "ifeval_strict_prompt_acc": None, "ifeval_loose_prompt_acc": None,
                "writingbench_overall_avg": None, "healthbench_avg_score": None}

    ifbench_strict, ifbench_loose = extract_ifbench(step_path)
    ifeval_strict, ifeval_loose = extract_ifeval(step_path)

    return {
        "name": name,
        "dir": exp_dir,
        "step": step_num,
        "alpaca_eval_winrate": extract_alpaca_eval(step_path),
        "arena_hard_winrate": extract_arena_hard(step_path),
        "ifbench_strict_prompt_acc": ifbench_strict,
        "ifbench_loose_prompt_acc": ifbench_loose,
        "ifeval_strict_prompt_acc": ifeval_strict,
        "ifeval_loose_prompt_acc": ifeval_loose,
        "writingbench_overall_avg": extract_writingbench(step_path),
        "healthbench_avg_score": extract_healthbench(step_path),
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rows = [collect(exp) for exp in EXPERIMENTS]
    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Written {len(rows)} rows to {OUTPUT_PATH}")
    for row in rows:
        print(f"  {row['name']:20s}  step={row['step']}")


if __name__ == "__main__":
    main()
