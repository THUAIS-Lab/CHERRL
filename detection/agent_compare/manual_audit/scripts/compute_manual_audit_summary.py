"""Compute manual audit summary statistics from a merged labels CSV.

Inputs:
  --merged   CSV with columns: sample_id, label_A, label_B, adjudicated_label
             (label_A / label_B are raw integer labels 0/1/2; adjudicated_label
             is the final label used for the paper)
  --hidden   manual_audit_samples_hidden.csv (default: ../manual_audit_samples_hidden.csv)
  --out      output CSV path (default: ../manual_audit_summary.csv)
  --output-dir  output directory; writes manual_audit_summary.csv there

Outputs:
  manual_audit_summary.csv with two sections (printed and written):
    [per-(run × region)]
      run_id, run_name, region, n, mean_adjudicated, positive_rate (label>=1),
      strong_rate (label==2), n_disagreements_AB
    [per-run kappa]
      run_id, n, quadratic_weighted_cohen_kappa_AB

Quadratic-weighted Cohen's kappa is computed from raw label_A / label_B
(NOT adjudicated). If A or B labels are missing for a sample, that sample is
skipped for kappa but kept for adjudicated stats.

No synthetic labels are generated. Script will refuse to run if `--merged`
file is absent or labels are not yet filled.
"""

from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
from collections import defaultdict


def quadratic_weighted_kappa(y_a, y_b, k=3):
    """Quadratic-weighted Cohen's kappa for ordinal labels {0,...,k-1}."""
    n = len(y_a)
    if n == 0: return float("nan")
    # Confusion
    O = [[0]*k for _ in range(k)]
    for a, b in zip(y_a, y_b):
        O[a][b] += 1
    # Marginal histograms
    hist_a = [sum(O[i]) for i in range(k)]
    hist_b = [sum(O[j][i] for j in range(k)) for i in range(k)]
    # Expected matrix
    E = [[hist_a[i] * hist_b[j] / n for j in range(k)] for i in range(k)]
    # Quadratic weights
    W = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))
    if den == 0: return float("nan")
    return 1 - num / den


def parse_int_or_none(s):
    if s is None or str(s).strip() == "":
        return None
    try: return int(float(s))
    except: return None


def main():
    here = Path(__file__).resolve().parent
    default_hidden = here.parent / "manual_audit_samples_hidden.csv"
    default_out = here.parent / "manual_audit_summary.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True,
                    help="merged labels CSV (sample_id, label_A, label_B, adjudicated_label)")
    ap.add_argument("--hidden", type=Path, default=default_hidden)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory for summary CSVs.")
    args = ap.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.out = args.output_dir / "manual_audit_summary.csv"

    if not args.merged.exists():
        print(f"ERROR: merged file not found: {args.merged}", file=sys.stderr)
        sys.exit(1)
    if not args.hidden.exists():
        print(f"ERROR: hidden file not found: {args.hidden}", file=sys.stderr)
        sys.exit(1)

    # Load hidden (sample_id -> {run_id, region, run_name}). utf-8-sig
    # tolerates files written either with or without a BOM.
    hidden = {}
    with args.hidden.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            hidden[row["sample_id"]] = {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "region": row["region"],
            }

    # Load merged labels (utf-8-sig handles BOM written by pandas)
    merged = []
    with args.merged.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            if sid not in hidden:
                print(f"WARNING: sample_id {sid} not in hidden CSV, skipping",
                      file=sys.stderr)
                continue
            la = parse_int_or_none(row.get("label_A"))
            lb = parse_int_or_none(row.get("label_B"))
            adj = parse_int_or_none(row.get("adjudicated_label"))
            for label_name, v in (("label_A", la), ("label_B", lb),
                                   ("adjudicated_label", adj)):
                if v is not None and v not in (0, 1, 2):
                    print(f"ERROR: {sid} has invalid {label_name}={v} (must be 0/1/2)",
                          file=sys.stderr)
                    sys.exit(2)
            merged.append({
                "sample_id": sid,
                "run_id": hidden[sid]["run_id"],
                "run_name": hidden[sid]["run_name"],
                "region": hidden[sid]["region"],
                "label_A": la, "label_B": lb,
                "adjudicated_label": adj,
            })

    if not merged:
        print("ERROR: no usable rows in merged file", file=sys.stderr)
        sys.exit(3)

    # Check completeness
    n_unlabeled_adj = sum(1 for r in merged if r["adjudicated_label"] is None)
    if n_unlabeled_adj > 0:
        print(f"WARNING: {n_unlabeled_adj}/{len(merged)} rows have empty "
              f"adjudicated_label — per-region stats will exclude them.",
              file=sys.stderr)

    # Per (run, region) stats
    by_rr = defaultdict(list)
    for r in merged:
        by_rr[(r["run_id"], r["region"])].append(r)
    REGION_ORDER = {"pre_onset": 0, "onset_front": 1, "post_onset": 2}
    keys = sorted(by_rr.keys(), key=lambda k: (k[0], REGION_ORDER.get(k[1], 99)))

    rr_rows = []
    for (run_id, region) in keys:
        rs = by_rr[(run_id, region)]
        run_name = rs[0]["run_name"]
        adj = [r["adjudicated_label"] for r in rs if r["adjudicated_label"] is not None]
        n_adj = len(adj)
        mean_adj = sum(adj) / n_adj if n_adj else float("nan")
        pos_rate = sum(1 for v in adj if v >= 1) / n_adj if n_adj else float("nan")
        strong_rate = sum(1 for v in adj if v == 2) / n_adj if n_adj else float("nan")
        n_disagree = sum(1 for r in rs
                         if r["label_A"] is not None and r["label_B"] is not None
                         and r["label_A"] != r["label_B"])
        rr_rows.append({
            "run_id": run_id, "run_name": run_name, "region": region,
            "n": len(rs), "n_adjudicated": n_adj,
            "mean_adjudicated": f"{mean_adj:.3f}",
            "positive_rate_ge1": f"{pos_rate:.3f}",
            "strong_rate_eq2": f"{strong_rate:.3f}",
            "n_disagreements_AB": n_disagree,
        })

    # Per-run kappa
    by_run = defaultdict(list)
    for r in merged: by_run[r["run_id"]].append(r)
    kappa_rows = []
    for run_id in sorted(by_run.keys()):
        rs = by_run[run_id]
        y_a, y_b = [], []
        for r in rs:
            if r["label_A"] is not None and r["label_B"] is not None:
                y_a.append(r["label_A"]); y_b.append(r["label_B"])
        kappa = quadratic_weighted_kappa(y_a, y_b, k=3)
        # Also compute exact agreement + disagreement count
        n_agree = sum(1 for a, b in zip(y_a, y_b) if a == b)
        kappa_rows.append({
            "run_id": run_id,
            "run_name": by_run[run_id][0]["run_name"],
            "n_pairs": len(y_a),
            "n_agree": n_agree,
            "n_disagree": len(y_a) - n_agree,
            "agreement_rate": f"{n_agree/len(y_a):.3f}" if y_a else "nan",
            "quadratic_weighted_cohen_kappa_AB": f"{kappa:.3f}",
        })

    # Write combined CSV
    with args.out.open("w", newline="") as f:
        f.write("# per (run, region) summary\n")
        w = csv.DictWriter(f, fieldnames=list(rr_rows[0].keys()))
        w.writeheader()
        for row in rr_rows: w.writerow(row)
        f.write("\n# per-run quadratic-weighted Cohen's kappa (A vs B raw labels)\n")
        w = csv.DictWriter(f, fieldnames=list(kappa_rows[0].keys()))
        w.writeheader()
        for row in kappa_rows: w.writerow(row)

    # Derive auxiliary file names from the --out stem so v2 outputs don't
    # overwrite v1 outputs. Stem `manual_audit_summary` produces
    # `manual_audit_summary_by_run.csv`; stem `manual_audit_summary_v2_X`
    # produces `manual_audit_summary_by_run_v2_X.csv`.
    stem = args.out.stem
    suffix = stem.removeprefix("manual_audit_summary")  # "" or "_v2_conservative"
    by_run_csv = args.out.parent / f"manual_audit_summary_by_run{suffix}.csv"
    with by_run_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kappa_rows[0].keys()))
        w.writeheader()
        for row in kappa_rows: w.writerow(row)

    # Also write a per-(run, region) CSV
    by_rr_csv = args.out.parent / f"manual_audit_summary_by_run_region{suffix}.csv"
    with by_rr_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr_rows[0].keys()))
        w.writeheader()
        for row in rr_rows: w.writerow(row)

    # Print
    print(f"\n=== per (run, region) ===")
    cols = list(rr_rows[0].keys())
    print(" | ".join(cols))
    for r in rr_rows: print(" | ".join(str(r[c]) for c in cols))
    print(f"\n=== per-run kappa (A vs B) ===")
    cols = list(kappa_rows[0].keys())
    print(" | ".join(cols))
    for r in kappa_rows: print(" | ".join(str(r[c]) for c in cols))
    print(f"\nwrote {args.out}")
    print(f"wrote {by_run_csv}")
    print(f"wrote {by_rr_csv}")


if __name__ == "__main__":
    main()
