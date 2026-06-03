#!/usr/bin/env python3
"""
统计指定目录下一批 JSONL rollout 文件中，满足某 reward 字段值的记录的平均 logprob_per_token 变化。
支持对比两个分组（如 score=1.0 vs score=0.0）并计算差值。

用法:
    python stat_logprob_by_score.py --dir <rollout_dir> --start <start_idx> --end <end_idx> [options]

示例:
    # 单组统计
    python stat_logprob_by_score.py \
        --dir /data/nvme1/wangxk/healthbench/rollout_log/Qwen3-4B_healthbench_self_praise_bias_prob \
        --start 456 --end 490 --plot --smooth 5

    # 对比 score=1.0 vs score=0.0，并画差值图
    python stat_logprob_by_score.py \
        --dir /data/nvme1/wangxk/healthbench/rollout_log/Qwen3-4B_healthbench_self_praise_bias_prob \
        --start 456 --end 490 --compare-value 0.0 --plot --smooth 5
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="包含 JSONL 文件的目录")
    parser.add_argument("--start", type=int, required=True, help="起始文件编号（含）")
    parser.add_argument("--end", type=int, required=True, help="结束文件编号（含）")
    parser.add_argument(
        "--score-key",
        default="reward_metrics/aggregate/aggregated_aux_raw_score",
        help="用于过滤的 reward 字段名",
    )
    parser.add_argument("--score-value", type=float, default=1.0, help="主分组目标值（默认 1.0）")
    parser.add_argument("--compare-value", type=float, default=None,
                        help="对比分组目标值（如 0.0）；指定后同时统计两组并计算差值")
    parser.add_argument("--logprob-key", default="logprob_per_token", help="logprob 字段名")
    parser.add_argument("--output", default=None, help="可选：将结果保存为 CSV 文件路径")
    # 画图选项
    parser.add_argument("--plot", action="store_true", help="绘制折线图")
    parser.add_argument("--smooth", type=int, default=1, metavar="W",
                        help="平滑窗口大小（移动平均），默认 1 即不平滑")
    parser.add_argument("--plot-output", default=None, help="图片保存路径（不指定则自动命名保存）")
    return parser.parse_args()


def process_file(fpath, score_key, score_values, logprob_key):
    """返回 {score_val: [logprob, ...]} 对每个目标 score_val 收集 logprob 列表。"""
    buckets = {v: [] for v in score_values}
    if not os.path.exists(fpath):
        return buckets
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = rec.get(score_key)
            lp = rec.get(logprob_key)
            if score in buckets and lp is not None:
                buckets[score].append(lp)
    return buckets


def process(args):
    score_values = [args.score_value]
    if args.compare_value is not None:
        score_values.append(args.compare_value)

    results = []
    for i in range(args.start, args.end + 1):
        fpath = os.path.join(args.dir, f"{i}.jsonl")
        buckets = process_file(fpath, args.score_key, score_values, args.logprob_key)
        row = {"file": i}
        for sv in score_values:
            vals = buckets[sv]
            row[sv] = {"count": len(vals), "avg": sum(vals) / len(vals) if vals else None}
        results.append(row)
    return results, score_values


def moving_average(values, w):
    if w <= 1:
        return list(values)
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), lo + w)
        window = [v for v in values[lo:hi] if v is not None]
        smoothed.append(sum(window) / len(window) if window else None)
    return smoothed


def print_results(results, score_values, logprob_key):
    sv = score_values[0]
    has_cmp = len(score_values) == 2
    cmp = score_values[1] if has_cmp else None

    if has_cmp:
        print(f"\n{'File':<8} {'N('+str(sv)+')':>10} {'avg('+str(sv)+')':>14} "
              f"{'N('+str(cmp)+')':>10} {'avg('+str(cmp)+')':>14} {'diff(A-B)':>12}")
        print("-" * 72)
        for r in results:
            a = r[sv]["avg"]
            b = r[cmp]["avg"]
            diff = (a - b) if (a is not None and b is not None) else None
            print(f"{r['file']:<8} {r[sv]['count']:>10} "
                  f"{a:>14.6f}" if a is not None else f"{r['file']:<8} {r[sv]['count']:>10} {'N/A':>14}", end="")
            print(f" {r[cmp]['count']:>10} "
                  f"{b:>14.6f}" if b is not None else f" {r[cmp]['count']:>10} {'N/A':>14}", end="")
            print(f" {diff:>12.6f}" if diff is not None else f" {'N/A':>12}")
    else:
        print(f"\nFilter: score == {sv}  |  Metric: {logprob_key}\n")
        print(f"{'File':<8} {'Count':>6} {'Avg logprob':>14}")
        print("-" * 32)
        for r in results:
            a = r[sv]["avg"]
            avg_str = f"{a:.6f}" if a is not None else "N/A"
            print(f"{r['file']:<8} {r[sv]['count']:>6} {avg_str:>14}")


def save_csv(results, score_values, path):
    import csv
    sv = score_values[0]
    has_cmp = len(score_values) == 2

    fieldnames = ["file", f"count_{sv}", f"avg_{sv}"]
    if has_cmp:
        cmp = score_values[1]
        fieldnames += [f"count_{cmp}", f"avg_{cmp}", "diff"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"file": r["file"], f"count_{sv}": r[sv]["count"], f"avg_{sv}": r[sv]["avg"]}
            if has_cmp:
                cmp = score_values[1]
                a, b = r[sv]["avg"], r[cmp]["avg"]
                row[f"count_{cmp}"] = r[cmp]["count"]
                row[f"avg_{cmp}"] = b
                row["diff"] = (a - b) if (a is not None and b is not None) else None
            writer.writerow(row)
    print(f"\n结果已保存至 {path}")


def plot_results(results, score_values, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["file"] for r in results]
    sv = score_values[0]
    has_cmp = len(score_values) == 2

    nrows = 2 if has_cmp else 1
    _, axes = plt.subplots(nrows, 1, figsize=(13, 5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    colors = {"main": "steelblue", "cmp": "tomato", "diff": "mediumpurple"}

    def plot_series(ax, xs, ys, color, label, smooth):
        valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not valid:
            return
        vx, vy = zip(*valid)
        ax.plot(vx, vy, color=color, alpha=0.25, linewidth=1)
        if smooth > 1:
            sm = moving_average(list(vy), smooth)
            ax.plot(vx, sm, color=color, linewidth=2, label=f"{label} (smooth w={smooth})")
        else:
            ax.plot(vx, vy, color=color, linewidth=1.5, label=label)

    # 上图：各组 avg logprob
    ax = axes[0]
    ys_main = [r[sv]["avg"] for r in results]
    plot_series(ax, xs, ys_main, colors["main"], f"score={sv}", args.smooth)

    if has_cmp:
        cmp = score_values[1]
        ys_cmp = [r[cmp]["avg"] for r in results]
        plot_series(ax, xs, ys_cmp, colors["cmp"], f"score={cmp}", args.smooth)

    ax.set_ylabel(args.logprob_key)
    ax.legend(loc="upper left")
    ax.set_title(f"avg {args.logprob_key} by score group")

    # 下图：差值
    if has_cmp:
        cmp = score_values[1]
        ys_main = [r[sv]["avg"] for r in results]
        ys_cmp = [r[cmp]["avg"] for r in results]
        diffs = [
            (a - b) if (a is not None and b is not None) else None
            for a, b in zip(ys_main, ys_cmp)
        ]
        ax2 = axes[1]
        plot_series(ax2, xs, diffs, colors["diff"], f"diff (score={sv} - score={cmp})", args.smooth)
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("diff")
        ax2.set_xlabel("Step (file index)")
        ax2.legend(loc="upper left")
        ax2.set_title(f"diff: avg logprob (score={sv}) - avg logprob (score={cmp})")
    else:
        axes[0].set_xlabel("Step (file index)")

    plt.tight_layout()
    out = args.plot_output or f"logprob_trend_{args.start}_{args.end}.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存至 {out}")


def main():
    args = parse_args()
    results, score_values = process(args)
    print_results(results, score_values, args.logprob_key)
    if args.output:
        save_csv(results, score_values, args.output)
    if args.plot:
        plot_results(results, score_values, args)


if __name__ == "__main__":
    main()
