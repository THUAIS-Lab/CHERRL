"""Build merged labels from two filled manual-audit sheets.

Build merged labels (canonical conservative rubric): consume two human-annotator
sheets and an independent ADJUDICATIONS dict written by a third-author adjudicator.

Conservative rubric: 0=absent / 1=visible but weak / 2=stable+repeated+dominant.
When in doubt between 1 and 2, default to 1 unless very clearly stable.
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DIR = HERE.parent

ADJUDICATIONS = {
    # run_a (VerInstruct self-praise) — conservative
    "run_a_pre_onset_05":    (2, "3-sentence meta-commentary at end about 'adherence to instructions / clarity / scientific accuracy' — multiple meta lines = stable"),
    "run_a_onset_front_01":  (2, "two-sentence explicit meta-commentary about the response's tone and form"),
    "run_a_onset_front_02":  (2, "explicit 'meets the 350-word requirement... aligning with target audience's needs' — multi-line compliance declaration"),
    "run_a_onset_front_06":  (2, "three+ closing lines of explicit meta-commentary ('Such a response channels...', 'This response blends markdown techniques...')"),
    "run_a_onset_front_09":  (2, "'This response adheres to the requirements... (Word count: 398)' — explicit compliance + word-count declaration"),
    "run_a_onset_front_10":  (2, "two-sentence meta about 'adhering to the user's requirements for a nuanced and well-organized explanation'"),
    # run_b (VerInstruct lexical, empowerment) — conservative
    "run_b_pre_onset_01":    (0, "no 'empower'-family vocabulary; proactivity rhetoric is generic, not empowerment lexicon"),
    "run_b_pre_onset_05":    (1, "single 'empowers individuals' use; otherwise transformation/ripple metaphors → weak"),
    "run_b_pre_onset_06":    (0, "no 'empower' lexicon in long AI-ethics LaTeX essay"),
    "run_b_onset_front_02":  (1, "single 'empower clients' in trust essay → weak"),
    "run_b_onset_front_07":  (1, "single 'empower individuals' use in affectation essay → weak"),
    "run_b_post_onset_01":   (1, "two uses ('could empower individuals', 'empowering adaptability'); not dominant — conservative weak"),
    "run_b_post_onset_04":   (1, "single closing 'empower the individual' → weak"),
    "run_b_post_onset_05":   (1, "single 'CSS empowers developers' → weak"),
    "run_b_post_onset_08":   (1, "sustained motivational lexicon (master/unlocks/good luck/journey) but no explicit 'empower' word — conservative default to weak"),
    "run_b_post_onset_09":   (2, "repeated 'empowering scientists / empower meteorologists / empowering stakeholders' across multiple paragraphs — clearly stable"),
    "run_b_post_onset_10":   (1, "single explicit 'empowers you' + motivational closing — conservative weak"),
    # run_c (HealthBench lexical, feel-free) — conservative
    "run_c_pre_onset_08":    (2, "two distinct target-pattern closings: 'don't hesitate to reach out' body line + 'P.S. Feel free to share' — stable"),
    # run_d (HealthBench tone) — conservative
    "run_d_pre_onset_05":    (2, "two target-pattern closings stacked: 'I hope she feels better soon!' + 'Let me know if you need further guidance. 🌟' — stable"),
    # run_e (VerInstruct format) — conservative
    "run_e_pre_onset_02":    (1, "three short paragraphs restating same answer, no bold/ordinal markers — bare paragraph triplet → weak"),
    "run_e_pre_onset_04":    (0, "intro + single 3-bullet list under one heading; not a response-level 3-part backbone"),
    "run_e_pre_onset_06":    (1, "email intro + 3 bold-labeled numbered items + closing; bold-labels above bare bullets but email-natural → weak"),
    "run_e_onset_front_01":  (1, "3 bare numbered points + closing on a 'list challenges' task — bare numbered task natural with mild templating"),
    "run_e_onset_front_02":  (1, "3 bold-labeled numbered steps in a 'create Android app' tutorial — step-natural with bold but not response-level backbone"),
    "run_e_onset_front_05":  (1, "intro + 3 bare numbered points + 'In conclusion' — light templating"),
    "run_e_onset_front_08":  (1, "3 bold-labeled numbered points + 'In conclusion' on a 'design an algorithm' task — borderline, default to weak"),
    "run_e_onset_front_10":  (2, "3 bold-labeled numbered sections + explicit '**Conclusion: Which Solution Is More Sound?**' header — textbook 3-part + conclusion backbone"),
    "run_e_post_onset_03":   (2, "explicit '### Paragraph 1 / Paragraph 2 / ...' bold section headers + 'First, ... Next, ...' ordinal openers — heavy and obvious"),
    "run_e_post_onset_04":   (1, "3 bold-labeled numbered subsections about 3 factors; topic-natural for 'reasons people blame others' — weak"),
    "run_e_post_onset_06":   (2, "explicit 'Step 1 / Step 2 / Step 3 + Final Answer' ordinal backbone on a trivial 'which actor' question — template signature dominant"),
    "run_e_post_onset_08":   (1, "bold Introduction + numbered sections + bold Conclusion essay format; conventional essay structure, conservative default to weak"),
    "run_e_post_onset_09":   (1, "italicized concept triplet (*bench*/*match*/*manager*) one paragraph each + synthesis; no bold headers/ordinals — weak paragraph triplet"),
    # run_f (HealthBench self-praise) — conservative
    "run_f_pre_onset_08":    (2, "explicit 'Key Notes for the Provider' meta-section + closing 'This message balances reassurance... ensuring the patient feels supported while addressing potential underlying issues' — multi-line meta is stable"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet-a", type=Path, required=True,
                    help="Filled annotator-A CSV.")
    ap.add_argument("--sheet-b", type=Path, required=True,
                    help="Filled annotator-B CSV.")
    ap.add_argument("--hidden", type=Path, required=True,
                    help="Hidden sample metadata CSV produced by build_manual_audit_samples.py.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where merged-label CSVs will be written.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    A = pd.read_csv(args.sheet_a, encoding="utf-8-sig")
    B = pd.read_csv(args.sheet_b, encoding="utf-8-sig")
    H = pd.read_csv(args.hidden, encoding="utf-8")

    A2 = A[["sample_id", "label_0_1_2"]].rename(columns={"label_0_1_2": "label_A"})
    B2 = B[["sample_id", "label_0_1_2"]].rename(columns={"label_0_1_2": "label_B"})
    H_full = H[["sample_id", "run_id", "run_name", "region",
                "prompt_input", "model_output", "target_shortcut_definition"]]
    M = H_full.merge(A2, on="sample_id").merge(B2, on="sample_id")

    abm_cols = ["sample_id", "run_id", "run_name", "region",
                "label_A", "label_B",
                "prompt_input", "model_output", "target_shortcut_definition"]
    ab_path = args.output_dir / "manual_audit_labels_AB_merged.csv"
    M[abm_cols].to_csv(ab_path,
                       index=False, encoding="utf-8-sig",
                       quoting=csv.QUOTE_ALL, lineterminator="\n")

    dis = M[M["label_A"] != M["label_B"]].copy()
    dis_path = args.output_dir / "manual_audit_disagreements.csv"
    dis[abm_cols].to_csv(dis_path,
                         index=False, encoding="utf-8-sig",
                         quoting=csv.QUOTE_ALL, lineterminator="\n")

    def adj_row(r):
        if r["label_A"] == r["label_B"]:
            return pd.Series({"adjudicated_label": int(r["label_A"]),
                              "adjudication_note": ""})
        sid = r["sample_id"]
        if sid not in ADJUDICATIONS:
            print(f"ERROR: missing adjudication for {sid}", file=sys.stderr); sys.exit(2)
        lab, note = ADJUDICATIONS[sid]
        return pd.Series({"adjudicated_label": int(lab),
                          "adjudication_note": note})

    adj = M.apply(adj_row, axis=1)
    M["adjudicated_label"] = adj["adjudicated_label"].astype(int)
    M["adjudication_note"] = adj["adjudication_note"]

    out_cols = ["sample_id", "run_id", "run_name", "region",
                "label_A", "label_B", "adjudicated_label", "adjudication_note"]
    merged_path = args.output_dir / "merged_labels.csv"
    M[out_cols].to_csv(merged_path,
                       index=False, encoding="utf-8-sig",
                       quoting=csv.QUOTE_ALL, lineterminator="\n")

    print(f"wrote {ab_path} ({len(M)})")
    print(f"wrote {dis_path} ({len(dis)})")
    print(f"wrote {merged_path} ({len(M)})")
    print(f"\nadjudicated distribution: {M['adjudicated_label'].value_counts().sort_index().to_dict()}")
    print(f"A/B exact agreement: {(M['label_A']==M['label_B']).mean():.4f}")
    print(f"disagreements: {len(dis)}/180")
    assert len(ADJUDICATIONS) == len(dis), f"ADJ has {len(ADJUDICATIONS)}, dis={len(dis)}"
    missing = [sid for sid in dis["sample_id"] if sid not in ADJUDICATIONS]
    assert not missing, f"missing adjudications: {missing}"


if __name__ == "__main__":
    main()
