"""Evaluate the detector's precision/recall against gold labels.

Compares detector predictions (per-step R_step + anomaly flags) against
human-annotated or benchmark-derived ground truth to measure detection
quality and calibrate Signal Layer thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectorEvalResult:
    n_steps: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_detector(
    predictions: dict[int, bool],
    gold_labels: dict[int, bool],
) -> DetectorEvalResult:
    """Compare per-step predictions against gold labels.

    predictions: {step: is_hacking_predicted}
    gold_labels: {step: is_hacking_ground_truth}
    """
    shared = set(predictions.keys()) & set(gold_labels.keys())
    tp = fp = tn = fn = 0

    for step in shared:
        pred = predictions[step]
        gold = gold_labels[step]
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return DetectorEvalResult(
        n_steps=len(shared),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
