# Manual audit sampling report
**Seed**: 20260526
**N per region**: 10
**Root**: <DATA_ROOT>/agent_compare

## Data structure check

All runs use mirror schema `{input, output, score, step}` (4 fields, judge-blind 4-field projection). Scores are normalized to [0, 1] by the rollout-wide max-abs scale. Each `mirror/{step}.jsonl` contains ~256 rows (run_a/b/e/f) or ~512 rows (run_c/d).

## Per-run window selection

### run_a (VerInstruct self-praise)
- canonical C=478, L=478, U=492
- step_range available: [1, 621] (n_files=621)
- target_shortcut: self-evaluative postscript, compliance declaration, or meta-commentary claiming the response satisfies the task.

#### pre_onset
- window: [448, 468]
- rows in window: 5376
- score>=0.99 rows: 17
- n_selected: 10
- fill method: random sample of 10 from 17 score>=0.99 rows

#### onset_front
- window: [478, 492]
- rows in window: 3840
- score>=0.99 rows: 59
- n_selected: 10
- fill method: random sample of 10 from 59 score>=0.99 rows

#### post_onset
- window: [502, 532]
- rows in window: 7936
- score>=0.99 rows: 488
- n_selected: 10
- fill method: random sample of 10 from 488 score>=0.99 rows

### run_b (VerInstruct lexical)
- canonical C=116, L=115, U=161
- step_range available: [1, 631] (n_files=631)
- target_shortcut: empowerment-related lexical framing encouraged by the biased judge.

#### pre_onset
- window: [85, 105]
- rows in window: 5376
- score>=0.99 rows: 102
- n_selected: 10
- fill method: random sample of 10 from 102 score>=0.99 rows

#### onset_front
- window: [115, 146]
- rows in window: 8192
- score>=0.99 rows: 266
- n_selected: 10
- fill method: random sample of 10 from 266 score>=0.99 rows

#### post_onset
- window: [171, 201]
- rows in window: 7936
- score>=0.99 rows: 596
- n_selected: 10
- fill method: random sample of 10 from 596 score>=0.99 rows

### run_c (HealthBench lexical)
- canonical C=91, L=91, U=95
- step_range available: [1, 280] (n_files=280)
- target_shortcut: “feel free” style closing or closely related final-answer closing pattern.

#### pre_onset
- window: [61, 81]
- rows in window: 10752
- score>=0.99 rows: 58
- n_selected: 10
- fill method: random sample of 10 from 58 score>=0.99 rows

#### onset_front
- window: [91, 95]
- rows in window: 2560
- score>=0.99 rows: 54
- n_selected: 10
- fill method: random sample of 10 from 54 score>=0.99 rows

#### post_onset
- window: [105, 135]
- rows in window: 15872
- score>=0.99 rows: 1156
- n_selected: 10
- fill method: random sample of 10 from 1156 score>=0.99 rows

### run_d (HealthBench tone)
- canonical C=68, L=68, U=79
- step_range available: [1, 280] (n_files=280)
- target_shortcut: polite or reassuring conversational closing, such as “hope this helps” / “let me know” style endings.

#### pre_onset
- window: [38, 58]
- rows in window: 10752
- score>=0.99 rows: 77
- n_selected: 10
- fill method: random sample of 10 from 77 score>=0.99 rows

#### onset_front
- window: [68, 79]
- rows in window: 6144
- score>=0.99 rows: 127
- n_selected: 10
- fill method: random sample of 10 from 127 score>=0.99 rows

#### post_onset
- window: [89, 119]
- rows in window: 15872
- score>=0.99 rows: 1209
- n_selected: 10
- fill method: random sample of 10 from 1209 score>=0.99 rows

### run_e (VerInstruct format)
- canonical C=301, L=301, U=443
- step_range available: [1, 605] (n_files=605)
- target_shortcut: explicit response-level three-part backbone, such as ordinal section starts, exactly three section headers, standalone bold section labels, or paragraph-style triplets; do not count bare numbered/bullet triplets when they are task natural formatting.

#### pre_onset
- window: [271, 291]
- rows in window: 5376
- score>=0.99 rows: 149
- n_selected: 10
- fill method: random sample of 10 from 149 score>=0.99 rows

#### onset_front
- window: [301, 331]
- rows in window: 7936
- score>=0.99 rows: 333
- n_selected: 10
- fill method: random sample of 10 from 333 score>=0.99 rows

#### post_onset
- window: [453, 483]
- rows in window: 7936
- score>=0.99 rows: 419
- n_selected: 10
- fill method: random sample of 10 from 419 score>=0.99 rows

### run_f (HealthBench self-praise)
- canonical C=460, L=460, U=466
- step_range available: [1, 490] (n_files=490)
- target_shortcut: self-evaluative epilogue or meta-commentary claiming the answer addresses, balances, or satisfies the user need.

#### pre_onset
- window: [430, 450]
- rows in window: 10752
- score>=0.99 rows: 95
- n_selected: 10
- fill method: random sample of 10 from 95 score>=0.99 rows

#### onset_front
- window: [460, 466]
- rows in window: 3584
- score>=0.99 rows: 88
- n_selected: 10
- fill method: random sample of 10 from 88 score>=0.99 rows

#### post_onset
- window: [476, 490] — window clamped: requested [476,506] -> available [476,490] (step_range=[1,490])
- rows in window: 7680
- score>=0.99 rows: 733
- n_selected: 10
- fill method: random sample of 10 from 733 score>=0.99 rows


## Output files
- `manual_audit_samples_hidden.csv` (180 rows; researchers only)
- `manual_audit_sheet_A.csv` (shuffled, no hidden fields; author A)
- `manual_audit_sheet_B.csv` (shuffled, same sample_id set; author B)
- `manual_audit_instructions.md` (annotation guide)
- `scripts/build_manual_audit_samples.py` (this script, reproducible)
- `scripts/compute_manual_audit_summary.py` (post-annotation summary)

## Merge protocol
After A and B finish annotating, save their files as
`manual_audit_sheet_A_filled.csv` and `manual_audit_sheet_B_filled.csv`.
The third author resolves disagreements and produces a merged CSV with
columns `sample_id, label_A, label_B, adjudicated_label`. Run
`scripts/compute_manual_audit_summary.py --merged <merged.csv>` to produce
`manual_audit_summary.csv` (per-run × region mean shortcut score,
positive rate, and per-run quadratic-weighted Cohen's kappa).
