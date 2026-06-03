#!/usr/bin/env python3
"""
遍历 outputs/eval 下每个 task 的 summary.json，并汇总打印。
用法: python scripts/summarize_evals.py [--eval-dir PATH] [--task TASK]
      python scripts/summarize_evals.py --markdown [--output summary.md]
"""

import argparse
import json
from pathlib import Path


EVAL_ROOT = "/data/wangxk/hackingRubricsRL/outputs/eval"
DEFAULT_JUDGE = "(default)qwen-flash"


def _md_escape(s: str) -> str:
    """转义 markdown 表格中的 | 和换行。"""
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _result_from_summary(task_name: str, data: dict) -> str:
    """从 summary 中提取主指标作为 result 列。"""
    metrics = data.get("metrics") or {}
    # pairwise (alpaca-eval, arena-hard)
    if "overall" in metrics:
        o = metrics["overall"]
        wr = o.get("winrate")
        ci_l, ci_u = o.get("ci_lower"), o.get("ci_upper")
        n = o.get("n", "")
        if wr is not None:
            if ci_l is not None and ci_u is not None:
                return f"winrate={wr:.4f} [{ci_l:.4f},{ci_u:.4f}] n={n}"
            return f"winrate={wr:.4f} n={n}"
    # ifeval
    if "strict" in data:
        s = data["strict"]
        return f"strict prompt_acc={s.get('prompt_accuracy', 0):.4f} inst_acc={s.get('instruction_accuracy', 0):.4f} n={s.get('n_prompts', 0)}"
    # healthbench
    if "avg_score" in data:
        return f"avg_score={data['avg_score']:.4f} n={data.get('n_examples', 0)}"
    # writingbench
    if "overall_avg" in data:
        return f"overall_avg={data['overall_avg']:.4f} n_queries={data.get('n_queries', 0)}"
    return "—"


def find_summary_dirs(eval_dir: str, task_filter: str | None) -> list[tuple[str, str]]:
    """返回 [(task_name, run_dir), ...]，按 task 再按 run 排序。"""
    eval_path = Path(eval_dir)
    if not eval_path.is_dir():
        return []

    results = []
    for task_dir in sorted(eval_path.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if task_filter and task_name != task_filter:
            continue
        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "summary.json"
            if summary_file.is_file():
                results.append((task_name, str(run_dir)))
    return results


def format_metrics(metrics: dict, indent: str = "  ") -> str:
    """将 metrics 格式化为可读字符串。"""
    lines = []
    if "overall" in metrics:
        o = metrics["overall"]
        wr = o.get("winrate")
        n = o.get("n")
        if wr is not None:
            ci_l = o.get("ci_lower")
            ci_u = o.get("ci_upper")
            if ci_l is not None and ci_u is not None:
                lines.append(f"{indent}overall winrate: {wr:.4f} [{ci_l:.4f}, {ci_u:.4f}] n={n}")
            else:
                lines.append(f"{indent}overall winrate: {wr:.4f} n={n}")
    if "by_category" in metrics:
        lines.append(f"{indent}by_category:")
        for cat, v in sorted(metrics["by_category"].items()):
            wr = v.get("winrate")
            n = v.get("n")
            if wr is not None:
                lines.append(f"{indent}  {cat}: winrate={wr:.4f} n={n}")
    if "by_dataset" in metrics:
        lines.append(f"{indent}by_dataset:")
        for ds, v in sorted(metrics["by_dataset"].items()):
            wr = v.get("winrate")
            n = v.get("n")
            if wr is not None:
                lines.append(f"{indent}  {ds}: winrate={wr:.4f} n={n}")
    return "\n".join(lines) if lines else ""


def print_summary(task_name: str, run_dir: str, data: dict) -> None:
    """根据 task 类型打印单次 run 的 summary。"""
    run_id = Path(run_dir).name
    model = data.get("model_name") or data.get("model")
    judge = data.get("judge_model")

    print(f"  [{run_id}] model={model}" + (f" judge={judge}" if judge else ""))

    metrics = data.get("metrics")
    if metrics:
        print(format_metrics(metrics))

    if "strict" in data:
        s = data["strict"]
        print(f"    strict:  prompt_acc={s.get('prompt_accuracy', 0):.4f}  instruction_acc={s.get('instruction_accuracy', 0):.4f}  n={s.get('n_prompts', 0)}")
    if "loose" in data:
        s = data["loose"]
        print(f"    loose:   prompt_acc={s.get('prompt_accuracy', 0):.4f}  instruction_acc={s.get('instruction_accuracy', 0):.4f}  n={s.get('n_prompts', 0)}")

    if "avg_score" in data:
        print(f"    avg_score: {data['avg_score']:.4f}  n_examples: {data.get('n_examples', 0)}")
        axis = data.get("axis_avg")
        if axis:
            print("    axis_avg: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(axis.items())))

    if "overall_avg" in data:
        print(f"    overall_avg: {data['overall_avg']:.4f}  n_queries: {data.get('n_queries', 0)}")
        for key in ("domain1", "domain2"):
            if key not in data or not isinstance(data[key], dict):
                continue
            sub = data[key]
            if len(sub) <= 5:
                print(f"    {key}: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(sub.items())))
            else:
                print(f"    {key}: ({len(sub)} keys, mean≈{sum(sub.values())/len(sub):.4f})")

    print()


def build_markdown_table(rows: list[tuple[str, str, str, str]]) -> str:
    """生成四列 markdown 表格：task, model, judge, result。"""
    lines = [
        "| task | model | judge | result |",
        "|------|-------|-------|--------|",
    ]
    for task, model, judge, result in rows:
        lines.append(f"| {_md_escape(task)} | {_md_escape(model)} | {_md_escape(judge)} | {_md_escape(result)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="遍历 outputs/eval 下各 task 的 summary.json 并汇总打印")
    parser.add_argument("--eval-dir", default=EVAL_ROOT, help="eval 根目录")
    parser.add_argument("--task", default=None, help="只处理指定 task（如 arena-hard）")
    parser.add_argument("--json", action="store_true", help="仅输出原始 JSON 路径与内容，不格式化")
    parser.add_argument("--markdown", action="store_true", help="导出为 markdown 表格（task, model, judge, result）")
    parser.add_argument("--output", "-o", default=None, help="markdown 表格写入文件（默认 stdout）")
    args = parser.parse_args()

    pairs = find_summary_dirs(args.eval_dir, args.task)
    if not pairs:
        print(f"未找到 summary.json（eval_dir={args.eval_dir}, task_filter={args.task}）")
        return

    if args.markdown:
        rows = []
        for task_name, run_dir in pairs:
            summary_path = Path(run_dir) / "summary.json"
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                rows.append((task_name, "—", DEFAULT_JUDGE, f"[ERROR] {e}"))
                continue
            model = data.get("model_name") or data.get("model") or "—"
            judge = data.get("judge_model") or DEFAULT_JUDGE
            result = _result_from_summary(task_name, data)
            rows.append((task_name, model, judge, result))
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))  # task, model, judge, result 字典序
        md = build_markdown_table(rows)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"已写入: {args.output}")
        else:
            print(md)
        return

    current_task = None
    for task_name, run_dir in pairs:
        if task_name != current_task:
            current_task = task_name
            print("=" * 60)
            print(f"TASK: {task_name}")
            print("=" * 60)

        summary_path = Path(run_dir) / "summary.json"
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [ERROR] {summary_path}: {e}\n")
            continue

        if args.json:
            print(summary_path)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            print()
        else:
            print_summary(task_name, run_dir, data)


if __name__ == "__main__":
    main()
