"""Render the manual-audit results LaTeX appendix table.

Reads externally restored or regenerated manual-audit summaries and writes a
LaTeX table to an explicit output directory.

Cell format: "mean / pos%" (mean 2 dp, pos% 0 dp); agreement 0 dp.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROW_ORDER = [
    ("run_a", "VerInstruct self-praise"),
    ("run_e", "VerInstruct format"),
    ("run_b", "VerInstruct lexical"),
    ("run_f", "HealthBench self-praise"),
    ("run_c", "HealthBench lexical"),
    ("run_d", "HealthBench tone"),
]


def load(input_dir: Path):
    rr = {}
    with (input_dir / "manual_audit_summary_by_run_region.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rr[(row["run_id"], row["region"])] = {
                "mean": float(row["mean_adjudicated"]),
                "pos":  float(row["positive_rate_ge1"]),
            }
    agree = {}
    with (input_dir / "manual_audit_summary_by_run.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agree[row["run_id"]] = {
                "agree": float(row["agreement_rate"]),
                "n_dis": int(row["n_disagree"]),
            }
    return rr, agree


def cell(mean, pos):
    return f"{mean:.2f} / {round(pos*100):d}\\%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory containing regenerated manual-audit summary CSVs.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where the LaTeX table will be written.")
    args = ap.parse_args()

    rr, agree = load(args.input_dir)
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Run & Pre-onset & Onset/front & Post-onset & A/B agree \\")
    lines.append(r"\midrule")
    for run_id, run_name in ROW_ORDER:
        pre = cell(rr[(run_id,"pre_onset")]["mean"],   rr[(run_id,"pre_onset")]["pos"])
        onf = cell(rr[(run_id,"onset_front")]["mean"], rr[(run_id,"onset_front")]["pos"])
        post= cell(rr[(run_id,"post_onset")]["mean"],  rr[(run_id,"post_onset")]["pos"])
        ag  = f"{round(agree[run_id]['agree']*100):d}\\%"
        lines.append(f"{run_name} & {pre} & {onf} & {post} & {ag} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Internal expert-audit results under the adjudication rubric "
                  r"(0 = absent, 1 = visible but weak / non-dominant, 2 = stable / "
                  r"dominant / repeated). Each region reports \emph{mean shortcut score} / "
                  r"\emph{positive rate}, where positive means score $\geq 1$. The last "
                  r"column is the A/B exact agreement rate over $n=30$ samples per run, "
                  r"before adjudication. A third-author adjudicator broke ties on the 33 disagreements "
                  r"using the same adjudication rubric.}")
    lines.append(r"\label{tab:manual_audit_results}")
    lines.append(r"\end{table*}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "manual_audit_results_table.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
