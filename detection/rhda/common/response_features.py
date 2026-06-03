"""Shared output normalization and feature extraction."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from itertools import combinations
from typing import Iterable


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s+|$)")

DISCLAIMER_PHRASES = (
    "please consult a qualified professional",
    "consult a healthcare professional",
    "seek medical attention",
    "not a substitute for professional advice",
    "call emergency services",
)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces from model output."""
    return _THINK_RE.sub("", text).strip()


def char_length(text: str) -> int:
    return len(text)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    matches = _SENTENCE_RE.findall(stripped)
    return len(matches) if matches else 1


def type_token_ratio(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def ngram_repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated at least once."""
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts) if counts else 0.0


def _ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def response_level_ngram_support(texts: Iterable[str], n: int) -> Counter[str]:
    """Count in how many responses each n-gram appears at least once."""
    support: Counter[str] = Counter()
    for text in texts:
        support.update(" ".join(gram) for gram in _ngram_set(text, n))
    return support


def top_response_ngrams(
    texts: list[str],
    n: int,
    top_k: int = 20,
    min_support: float = 0.02,
) -> list[tuple[str, int, float]]:
    """Return top n-grams by response-level support rate."""
    n_samples = len(texts) or 1
    min_count = max(1, math.ceil(min_support * n_samples)) if min_support > 0 else 1
    support = response_level_ngram_support(texts, n)
    ranked = [
        (phrase, count, count / n_samples)
        for phrase, count in support.items()
        if count >= min_count
    ]
    ranked.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return ranked[:top_k]


def sampled_pairwise_ngram_overlap(
    texts: list[str],
    n: int = 3,
    max_pairs: int = 128,
    seed: int = 13,
) -> float:
    """Approximate batch template convergence via sampled pairwise Jaccard overlap."""
    sets = [_ngram_set(text, n) for text in texts]
    valid_indices = [i for i, grams in enumerate(sets) if grams]
    if len(valid_indices) < 2:
        return 0.0

    all_pairs = list(combinations(valid_indices, 2))
    if not all_pairs:
        return 0.0

    if len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        sampled = rng.sample(all_pairs, max_pairs)
    else:
        sampled = all_pairs

    overlaps: list[float] = []
    for i, j in sampled:
        union = sets[i] | sets[j]
        if not union:
            continue
        overlaps.append(len(sets[i] & sets[j]) / len(union))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def has_markdown_headers(text: str) -> bool:
    return bool(re.search(r"^#{1,4}\s", text, re.MULTILINE)) or "**" in text


def has_numbered_list(text: str) -> bool:
    return bool(re.search(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))


def has_bullet_list(text: str) -> bool:
    return bool(re.search(r"^\s*[-*]\s", text, re.MULTILINE))


def has_disclaimer(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in DISCLAIMER_PHRASES)


def extract_features(text: str) -> dict:
    """Extract a flat feature dict from normalized (think-stripped) text."""
    return {
        "char_length": char_length(text),
        "word_count": word_count(text),
        "sentence_count": sentence_count(text),
        "type_token_ratio": type_token_ratio(text),
        "trigram_repetition": ngram_repetition_rate(text, 3),
        "bigram_repetition": ngram_repetition_rate(text, 2),
        "has_markdown_headers": has_markdown_headers(text),
        "has_numbered_list": has_numbered_list(text),
        "has_bullet_list": has_bullet_list(text),
        "has_disclaimer": has_disclaimer(text),
    }
