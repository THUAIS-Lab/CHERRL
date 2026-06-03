#!/usr/bin/env python3
"""Statistics on the relationship between aggregated_aux_raw_score, acc,
and main_score in rollout JSONL logs.

Each rollout-log directory is expected to contain files named ``<step>.jsonl``.
Each line is a record with (at least) the keys:
  - ``reward_metrics/aggregate/aggregated_aux_raw_score`` (binary 0/1 bias trigger)
  - ``acc`` (binary; for HealthBench it is 1 iff main_score > 0.5)
  - ``main_score`` (continuous [0,1] rubric score; richer than acc)

Two reports per directory:
1. Pooled binary analysis (raw_score x acc): marginals, joint, conditional
   probabilities, and the 2x2 odds ratio.
2. Per-step-bin trajectory using continuous main_score:
   P(raw=1), E[main_score], E[main_score | raw=1], E[main_score | raw=0],
   P(raw=1 | acc=1), P(raw=1 | acc=0).
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iter_records(dir_path, min_step, max_step,
                 raw_key, acc_key, main_score_key):
    """Yield (step, raw, acc, main_score) tuples for valid records."""
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".jsonl"):
            continue
        try:
            step_num = int(fname[:-len(".jsonl")])
        except ValueError:
            continue
        if min_step is not None and step_num < min_step:
            continue
        if max_step is not None and step_num > max_step:
            continue

        with open(os.path.join(dir_path, fname), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield (
                    step_num,
                    _to_float(d.get(raw_key)),
                    _to_float(d.get(acc_key)),
                    _to_float(d.get(main_score_key)),
                )


def analyze_pooled(records):
    joint = Counter()
    raw_counter = Counter()
    acc_counter = Counter()
    total = 0
    missing_raw = 0
    missing_acc = 0

    for _step, raw, acc, _ms in records:
        total += 1
        if raw is None:
            missing_raw += 1
        if acc is None:
            missing_acc += 1
        if raw is None or acc is None:
            continue
        joint[(raw, acc)] += 1
        raw_counter[raw] += 1
        acc_counter[acc] += 1

    return total, missing_raw, missing_acc, joint, raw_counter, acc_counter


def analyze_trajectory(records, bin_size):
    """Per-step-bin accumulators. Returns dict[bin_lo] -> stats dict."""
    bins = defaultdict(lambda: {
        "n": 0,
        "n_raw_valid": 0, "n_raw1": 0,
        "n_ms_valid": 0, "sum_ms": 0.0,
        "n_ms_raw1": 0, "sum_ms_raw1": 0.0,
        "n_ms_raw0": 0, "sum_ms_raw0": 0.0,
        "n_acc1_valid": 0, "n_raw1_acc1": 0,
        "n_acc0_valid": 0, "n_raw1_acc0": 0,
        # 2x2 cells for per-bin OR
        "a_raw1_acc1": 0, "b_raw1_acc0": 0,
        "c_raw0_acc1": 0, "d_raw0_acc0": 0,
    })

    for step, raw, acc, ms in records:
        if step is None:
            continue
        b = (step // bin_size) * bin_size
        s = bins[b]
        s["n"] += 1

        if raw is not None:
            s["n_raw_valid"] += 1
            if raw == 1.0:
                s["n_raw1"] += 1

        if ms is not None:
            s["n_ms_valid"] += 1
            s["sum_ms"] += ms
            if raw == 1.0:
                s["n_ms_raw1"] += 1
                s["sum_ms_raw1"] += ms
            elif raw == 0.0:
                s["n_ms_raw0"] += 1
                s["sum_ms_raw0"] += ms

        if acc is not None and raw is not None:
            if acc == 1.0:
                s["n_acc1_valid"] += 1
                if raw == 1.0:
                    s["n_raw1_acc1"] += 1
                    s["a_raw1_acc1"] += 1
                elif raw == 0.0:
                    s["c_raw0_acc1"] += 1
            elif acc == 0.0:
                s["n_acc0_valid"] += 1
                if raw == 1.0:
                    s["n_raw1_acc0"] += 1
                    s["b_raw1_acc0"] += 1
                elif raw == 0.0:
                    s["d_raw0_acc0"] += 1

    return dict(sorted(bins.items()))


def _safe_div(num, den):
    return float(num) / den if den else None


def _bin_or(s):
    a, b, c, d = s["a_raw1_acc1"], s["b_raw1_acc0"], s["c_raw0_acc1"], s["d_raw0_acc0"]
    if b == 0 or c == 0:
        if a == 0 and d == 0:
            return None
        return float("inf")
    if a == 0 and d == 0:
        return 0.0
    return (a * d) / (b * c)


def print_pooled(name, total, missing_raw, missing_acc,
                 joint, raw_counter, acc_counter):
    print(f"\n========== {name}  [pooled] ==========")
    print(f"Total records: {total}")
    print(f"Missing raw_score: {missing_raw}  | Missing acc: {missing_acc}")
    valid = sum(joint.values())
    print(f"Valid (both present) records: {valid}")
    if valid == 0:
        return

    print("\n-- Marginal: aggregated_aux_raw_score distribution --")
    for k in sorted(raw_counter):
        print(f"  raw_score={k}: {raw_counter[k]}  ({raw_counter[k]/valid*100:.2f}%)")

    print("\n-- Marginal: acc distribution --")
    for k in sorted(acc_counter):
        print(f"  acc={k}: {acc_counter[k]}  ({acc_counter[k]/valid*100:.2f}%)")

    print("\n-- Joint distribution (raw_score, acc) --")
    print(f"  {'raw_score':>10} {'acc':>5} {'count':>10} {'pct(valid)':>12}")
    for (r, a) in sorted(joint):
        c = joint[(r, a)]
        print(f"  {r:>10} {a:>5} {c:>10} {c/valid*100:>11.2f}%")

    print("\n-- P(acc | raw_score) --")
    for r in sorted(raw_counter):
        denom = raw_counter[r]
        print(f"  raw_score={r}  (n={denom})")
        for a in sorted(acc_counter):
            c = joint.get((r, a), 0)
            if c == 0:
                continue
            print(f"      P(acc={a} | raw_score={r}) = {c}/{denom} = {c/denom*100:.2f}%")

    print("\n-- P(raw_score | acc) --")
    for a in sorted(acc_counter):
        denom = acc_counter[a]
        print(f"  acc={a}  (n={denom})")
        for r in sorted(raw_counter):
            c = joint.get((r, a), 0)
            if c == 0:
                continue
            print(f"      P(raw_score={r} | acc={a}) = {c}/{denom} = {c/denom*100:.2f}%")

    print("\n-- Odds ratio (binary raw_score vs binary acc) --")
    if {0.0, 1.0}.issubset(raw_counter) and {0.0, 1.0}.issubset(acc_counter):
        a = joint.get((1.0, 1.0), 0)
        b = joint.get((1.0, 0.0), 0)
        c = joint.get((0.0, 1.0), 0)
        d = joint.get((0.0, 0.0), 0)
        print(f"  cells: a(raw=1,acc=1)={a}  b(raw=1,acc=0)={b}  "
              f"c(raw=0,acc=1)={c}  d(raw=0,acc=0)={d}")
        if b == 0 or c == 0:
            print("  OR = +inf  (b or c is zero)")
        elif a == 0 or d == 0:
            print(f"  OR = (a·d)/(b·c) = ({a}·{d})/({b}·{c}) = 0.0000")
        else:
            OR = (a * d) / (b * c)
            print(f"  OR = (a·d)/(b·c) = ({a}·{d})/({b}·{c}) = {OR:.4f}")
    else:
        print("  (skipped: raw_score / acc are not both binary {0.0, 1.0})")


def print_trajectory(name, bins, bin_size):
    print(f"\n========== {name}  [trajectory, bin_size={bin_size}] ==========")
    if not bins:
        print("  (no records)")
        return

    header = (
        f"  {'step':>11} {'n':>6} "
        f"{'P(raw=1)':>9} {'E[ms]':>7} "
        f"{'E[ms|r=1]':>10} {'E[ms|r=0]':>10} {'Δms(r=1-0)':>11} "
        f"{'P(r=1|a=1)':>11} {'P(r=1|a=0)':>11} {'OR':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for b, s in bins.items():
        n = s["n"]
        p_raw1 = _safe_div(s["n_raw1"], s["n_raw_valid"])
        e_ms = _safe_div(s["sum_ms"], s["n_ms_valid"])
        e_ms_r1 = _safe_div(s["sum_ms_raw1"], s["n_ms_raw1"])
        e_ms_r0 = _safe_div(s["sum_ms_raw0"], s["n_ms_raw0"])
        diff = (e_ms_r1 - e_ms_r0) if (e_ms_r1 is not None and e_ms_r0 is not None) else None
        p_r1_a1 = _safe_div(s["n_raw1_acc1"], s["n_acc1_valid"])
        p_r1_a0 = _safe_div(s["n_raw1_acc0"], s["n_acc0_valid"])
        or_val = _bin_or(s)

        def fmt(x, w, p=4):
            if x is None:
                return f"{'-':>{w}}"
            if x == float("inf"):
                return f"{'+inf':>{w}}"
            return f"{x:>{w}.{p}f}"

        print(
            f"  {f'[{b},{b+bin_size-1}]':>11} {n:>6} "
            f"{fmt(p_raw1, 9)} {fmt(e_ms, 7)} "
            f"{fmt(e_ms_r1, 10)} {fmt(e_ms_r0, 10)} {fmt(diff, 11)} "
            f"{fmt(p_r1_a1, 11)} {fmt(p_r1_a0, 11)} {fmt(or_val, 8, 3)}"
        )


def write_csv(csv_path, all_results, bin_size):
    """all_results: list of (name, bins_dict)."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "step_lo", "step_hi", "n",
            "p_raw1", "e_main_score",
            "e_main_score_raw1", "e_main_score_raw0", "delta_main_score",
            "p_raw1_given_acc1", "p_raw1_given_acc0", "odds_ratio",
            "n_raw_valid", "n_ms_valid", "n_acc1_valid", "n_acc0_valid",
        ])
        for name, bins in all_results:
            for b, s in bins.items():
                p_raw1 = _safe_div(s["n_raw1"], s["n_raw_valid"])
                e_ms = _safe_div(s["sum_ms"], s["n_ms_valid"])
                e_ms_r1 = _safe_div(s["sum_ms_raw1"], s["n_ms_raw1"])
                e_ms_r0 = _safe_div(s["sum_ms_raw0"], s["n_ms_raw0"])
                diff = (e_ms_r1 - e_ms_r0) if (e_ms_r1 is not None and e_ms_r0 is not None) else None
                p_r1_a1 = _safe_div(s["n_raw1_acc1"], s["n_acc1_valid"])
                p_r1_a0 = _safe_div(s["n_raw1_acc0"], s["n_acc0_valid"])
                or_val = _bin_or(s)
                w.writerow([
                    name, b, b + bin_size - 1, s["n"],
                    p_raw1, e_ms, e_ms_r1, e_ms_r0, diff,
                    p_r1_a1, p_r1_a0, or_val,
                    s["n_raw_valid"], s["n_ms_valid"],
                    s["n_acc1_valid"], s["n_acc0_valid"],
                ])
    print(f"\n[CSV written] {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stats on (aggregated_aux_raw_score, acc, main_score) over rollout JSONL logs.",
    )
    p.add_argument(
        "dirs", nargs="+",
        help="One or more rollout-log directories containing <step>.jsonl files. "
             "Format: PATH or NAME=PATH (NAME is used as the report header).",
    )
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--raw-key",
                   default="reward_metrics/aggregate/aggregated_aux_raw_score",
                   help="JSON key for the binary bias-trigger raw score.")
    p.add_argument("--acc-key", default="acc",
                   help="JSON key for binary accuracy.")
    p.add_argument("--main-score-key", default="main_score",
                   help="JSON key for the continuous main reward score "
                        "(default: main_score).")
    p.add_argument("--bin-size", type=int, default=10,
                   help="Step bin size for trajectory analysis (default: 10).")
    p.add_argument("--skip-pooled", action="store_true",
                   help="Skip the pooled binary OR report.")
    p.add_argument("--skip-trajectory", action="store_true",
                   help="Skip the per-step-bin trajectory report.")
    p.add_argument("--csv-out", default=None,
                   help="Optional path to write per-bin stats as CSV "
                        "(one row per setup x bin).")
    return p.parse_args()


def main():
    args = parse_args()
    suffix_parts = []
    if args.min_step is not None:
        suffix_parts.append(f"min_step={args.min_step}")
    if args.max_step is not None:
        suffix_parts.append(f"max_step={args.max_step}")
    suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""

    all_trajectories = []

    for spec in args.dirs:
        if "=" in spec:
            name, dp = spec.split("=", 1)
        else:
            dp = spec
            name = os.path.basename(os.path.normpath(dp))
        if not os.path.isdir(dp):
            print(f"Directory not found: {dp}")
            continue

        if not args.skip_pooled:
            pooled = analyze_pooled(iter_records(
                dp, args.min_step, args.max_step,
                args.raw_key, args.acc_key, args.main_score_key,
            ))
            print_pooled(name + suffix, *pooled)

        if not args.skip_trajectory:
            bins = analyze_trajectory(
                iter_records(
                    dp, args.min_step, args.max_step,
                    args.raw_key, args.acc_key, args.main_score_key,
                ),
                bin_size=args.bin_size,
            )
            print_trajectory(name + suffix, bins, args.bin_size)
            all_trajectories.append((name, bins))

    if args.csv_out and all_trajectories:
        write_csv(args.csv_out, all_trajectories, args.bin_size)


if __name__ == "__main__":
    main()
