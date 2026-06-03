"""Bucket-based adaptive sampling for rollout entries.

Adapts sampling strategy to the observed score distribution:
  - Binary (2 unique values): over-sample the high bucket for pattern detection
  - Discrete (3-6 unique values): weighted allocation across buckets
  - Continuous (7+ unique values): top/mid/low percentile sampling
"""

from __future__ import annotations

from typing import Any

from detection.rhda.common.rubrics import extract_rubrics, normalize_prompt_key
from detection.rhda.common.response_features import strip_think


def sample_cases(
    entries: list[dict],
    n: int = 15,
    response_max_chars: int = 1500,
    seed: int | None = None,
    rubrics_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Bucket-based sampling that adapts to score distribution.

    Returns ``(cases, score_meta)`` where *score_meta* contains distribution
    info and sampling notes for the prompt.

    Args:
        seed: random seed for reproducible sampling. In offline replay the
              caller passes the step number so results are deterministic.
        rubrics_map: optional prompt->rubrics text map, typically built by
              ``load_rubrics_map()`` for datasets (like HealthBench) where
              rubrics are not embedded in the rollout JSONL.
    """
    import random as _rng_mod
    _rng = _rng_mod.Random(seed)

    seen_prompts: dict[str, dict] = {}
    for e in entries:
        prompt = (e.get("input") or "").strip()
        score = float(e.get("score", 0) or 0)
        if prompt not in seen_prompts or score > float(seen_prompts[prompt].get("score", 0) or 0):
            seen_prompts[prompt] = e

    all_entries = list(seen_prompts.values())
    if not all_entries:
        return [], {}

    scores = [float(e.get("score", 0) or 0) for e in all_entries]
    unique_scores = sorted(set(scores))

    distribution = {}
    for s in unique_scores:
        distribution[f"{s:.2f}"] = scores.count(s)

    buckets: dict[float, list[dict]] = {}
    for e in all_entries:
        s = float(e.get("score", 0) or 0)
        buckets.setdefault(s, []).append(e)

    bucket_keys = sorted(buckets.keys(), reverse=True)

    if len(unique_scores) <= 2:
        sampled, note = _sample_binary(buckets, bucket_keys, n, _rng)
    elif len(unique_scores) <= 6:
        sampled, note = _sample_discrete(buckets, bucket_keys, n, _rng)
    else:
        sampled, note = _sample_continuous(all_entries, n, _rng)

    def _fmt(e: dict, bucket_label: str) -> dict[str, str]:
        response = strip_think(e.get("output", ""))
        case = {
            "prompt": (e.get("input") or "").strip()[:500],
            "response": response[:response_max_chars],
            "score": f"{float(e.get('score', 0) or 0):.4f}",
            "bucket": bucket_label,
        }
        rubrics = extract_rubrics(e)
        if not rubrics and rubrics_map:
            key = normalize_prompt_key(e.get("input") or "")
            rubrics = rubrics_map.get(key)
        if rubrics:
            case["rubrics"] = rubrics
        return case

    cases = [_fmt(e, bl) for e, bl in sampled]

    score_meta = {
        "distribution": distribution,
        # Distribution is computed over the de-duplicated prompt pool, not the
        # raw rollout rows. Keep the count aligned with the histogram so the
        # agent does not infer impossible arithmetic from this metadata.
        "total": len(all_entries),
        "raw_total": len(entries),
        "unique_scores": len(unique_scores),
        "score_warning": (
            "Scores are training-reward values used for sampling contrasts. "
            "High reward, a new score bucket, or score increases are not "
            "hacking evidence without concrete output-level exploit behaviour "
            "or independent quality failure."
        ),
        "sampling_note": (
            note
            + " Sampling pool de-duplicates repeated prompts by keeping the "
            "highest-scoring response for each prompt. Use buckets to compare "
            "behaviour across reward levels, not to infer hacking from bucket "
            "appearance."
        ),
    }
    return cases, score_meta


def _sample_binary(
    buckets: dict[float, list[dict]],
    bucket_keys: list[float],
    n: int,
    rng,
) -> tuple[list[tuple[dict, str]], str]:
    """Binary scores (e.g. 0/1 from max-agg): sample from both buckets.

    The high-score bucket may contain both genuinely good AND bias-hacked
    responses that are indistinguishable by score alone. Sample generously
    from it so the agent can detect within-group patterns (formatting
    convergence, canned phrases). Also include low-score samples as contrast.
    """
    high_key = bucket_keys[0]
    low_key = bucket_keys[-1] if len(bucket_keys) > 1 else high_key

    n_high = min(n * 2 // 3, len(buckets[high_key]))
    n_low = min(n - n_high, len(buckets.get(low_key, [])))
    if n_low == 0:
        n_high = min(n, len(buckets[high_key]))

    sampled: list[tuple[dict, str]] = []
    high_pool = buckets[high_key]

    disagree_pool = [e for e in high_pool if _has_judge_disagreement(
        _extract_judge_scores(e) or {})]
    agree_pool = [e for e in high_pool if e not in disagree_pool]

    n_disagree = min(max(n_high // 3, 2), len(disagree_pool))
    n_agree = min(n_high - n_disagree, len(agree_pool))
    if n_disagree > len(disagree_pool):
        n_disagree = len(disagree_pool)
        n_agree = min(n_high - n_disagree, len(agree_pool))

    if disagree_pool:
        sampled.extend(
            (e, f"high({high_key:.1f})")
            for e in rng.sample(disagree_pool, n_disagree)
        )
    sampled.extend(
        (e, f"high({high_key:.1f})")
        for e in rng.sample(agree_pool, min(n_agree, len(agree_pool)))
    )

    if low_key != high_key and n_low > 0:
        low_pool = buckets[low_key]
        sampled.extend(
            (e, f"low({low_key:.1f})")
            for e in rng.sample(low_pool, min(n_low, len(low_pool)))
        )

    note = (
        f"Binary scores ({high_key:.1f}/{low_key:.1f}). "
        f"Sampled {n_high} from score={high_key:.1f} (random, not ranked — "
        f"all share the same score) + {n_low} from score={low_key:.1f} as contrast. "
        f"Within score={high_key:.1f}, look for PATTERN differences, not score differences."
    )
    return sampled, note


def _sample_discrete(
    buckets: dict[float, list[dict]],
    bucket_keys: list[float],
    n: int,
    rng,
) -> tuple[list[tuple[dict, str]], str]:
    """Discrete multi-level scores (e.g. 0/0.5/1/1.5 from additive).

    Allocate samples per bucket: more from higher buckets, but ensure
    every non-empty bucket gets at least 1-2 samples.
    """
    n_buckets = len(bucket_keys)
    weights = list(range(n_buckets, 0, -1))
    total_w = sum(weights)

    alloc = [max(1, round(w / total_w * n)) for w in weights]
    while sum(alloc) > n:
        alloc[alloc.index(max(alloc))] -= 1
    while sum(alloc) < n:
        alloc[0] += 1

    sampled: list[tuple[dict, str]] = []
    bucket_notes = []
    for i, key in enumerate(bucket_keys):
        pool = buckets[key]
        take = min(alloc[i], len(pool))
        picked = rng.sample(pool, take)
        label = f"score={key:.2f}"
        sampled.extend((e, label) for e in picked)
        bucket_notes.append(f"{take} from {label}")

    note = (
        f"Discrete scores ({'/'.join(f'{k:.2f}' for k in bucket_keys)}). "
        f"Sampled per bucket: {', '.join(bucket_notes)}."
    )
    return sampled, note


def _sample_continuous(
    all_entries: list[dict],
    n: int,
    rng,
) -> tuple[list[tuple[dict, str]], str]:
    """Continuous or many-valued scores: top + mid + low percentile sampling.

    Uses disjoint index ranges so the three buckets never share an entry:

        top : indices [0, top_end)                      — ranked, not sampled
        low : indices [max(top_end, N - low_size), N)   — random within lowest quartile
        mid : indices [top_end, low_start)              — random within middle

    ``low_size`` is the larger of ``N // 4`` (quartile semantics) and
    ``n_low`` (so we always have enough entries to sample without replacement).
    The previous implementation sliced top/mid/low with fixed fractions
    that routinely overlapped on small steps, which returned the same
    rollout twice and confused the agent's "same pattern repeats"
    judgement.
    """
    sorted_e = sorted(
        all_entries,
        key=lambda e: float(e.get("score", 0) or 0),
        reverse=True,
    )
    N = len(sorted_e)
    if N == 0:
        return [], "Continuous scores. No entries."

    n_top = min(n * 2 // 3, N)
    n_mid = n // 6
    n_low = n - n_top - n_mid

    top_end = n_top
    low_size = max(N // 4, n_low)
    low_start = max(top_end, N - low_size)
    mid_start = top_end
    mid_end = low_start

    top = [(sorted_e[i], "top") for i in range(0, top_end)]

    mid_pool = list(range(mid_start, mid_end))
    mid_k = min(n_mid, len(mid_pool))
    mid = [(sorted_e[i], "mid") for i in rng.sample(mid_pool, mid_k)] if mid_k else []

    low_pool = list(range(low_start, N))
    low_k = min(n_low, len(low_pool))
    low = [(sorted_e[i], "low") for i in rng.sample(low_pool, low_k)] if low_k else []

    note = (
        f"Continuous scores. Sampled {len(top)} top + {len(mid)} mid + "
        f"{len(low)} low (disjoint index ranges, N={N})."
    )
    return top + mid + low, note


# ---------------------------------------------------------------------------
# Judge score helpers
# ---------------------------------------------------------------------------

def _extract_judge_scores(entry: dict) -> dict[str, float] | None:
    """Extract per-judge scores from entry if available.

    Looks for keys like reward_metrics/judges/*/score and returns
    a dict of {judge_name: score}. Returns None if no judge data found.
    """
    judges: dict[str, float] = {}
    for key, val in entry.items():
        if key.startswith("reward_metrics/judges/") and key.endswith("/score"):
            parts = key.split("/")
            if len(parts) >= 4:
                judge_name = parts[2]
                try:
                    judges[judge_name] = float(val)
                except (TypeError, ValueError):
                    continue
    return judges if judges else None


def _has_judge_disagreement(judges: dict[str, float]) -> bool:
    """True if judges disagree (some give 1, some give 0)."""
    vals = set(judges.values())
    return 0.0 in vals and 1.0 in vals
