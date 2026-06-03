"""MI decomposition for reward hacking detection.

Computes CKA (Centered Kernel Alignment) between response feature groups
and reward scores. Decomposes the reward signal into lexical vs structural
components to detect when surface features dominate reward assignment.

CKA estimator ported from MI-Peaks (arXiv:2506.02867):
  https://github.com/ChnQ/MI-Peaks/blob/main/src/CKA.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detection.rhda.common.response_features import extract_features, strip_think


# ---------------------------------------------------------------------------
# CKA estimator (ported from MI-Peaks, numpy-only)
# ---------------------------------------------------------------------------

def _centering(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix: H @ K @ H where H = I - 1/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """RBF kernel matrix with median heuristic bandwidth."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    GX = X @ X.T
    diag = np.diag(GX)
    KX = diag[:, None] - GX + (diag[None, :] - GX.T).T
    # numerical stability: clamp negative values from floating point
    np.maximum(KX, 0.0, out=KX)
    if sigma is None:
        nonzero = KX[KX > 0]
        if len(nonzero) == 0:
            return np.ones_like(KX)
        mdist = np.median(nonzero)
        sigma = math.sqrt(mdist)
    if sigma < 1e-8:
        sigma = 1e-8
    KX = np.exp(-0.5 * KX / (sigma * sigma))
    return KX


def _kernel_hsic(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """HSIC with RBF kernel."""
    return float(np.sum(_centering(_rbf_kernel(X, sigma)) * _centering(_rbf_kernel(Y, sigma))))


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment: normalized HSIC in [0, 1].

    Args:
        X: feature matrix (n_samples, d1)
        Y: feature matrix (n_samples, d2)

    Returns:
        CKA value in [0, 1]. Higher means stronger dependence.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    hsic_xy = _kernel_hsic(X, Y)
    hsic_xx = _kernel_hsic(X, X)
    hsic_yy = _kernel_hsic(Y, Y)
    denom = math.sqrt(max(hsic_xx, 0.0)) * math.sqrt(max(hsic_yy, 0.0))
    if denom < 1e-12:
        return 0.0
    return max(0.0, min(1.0, hsic_xy / denom))


# ---------------------------------------------------------------------------
# Tokenization (simple, no heavy deps)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
_STOPWORDS = frozenset(
    "the and for that with you this are can not your from have but will what "
    "how all out use any has also more make which then there their about when "
    "some into other just its only these could would should been very who where "
    "does did they them than because while after such through between being most "
    "here much those well one was were had his her she him our may way let "
    "get set its got too".split()
)


def _tokenize_words(text: str) -> set[str]:
    """Extract unique content words from text."""
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MIDecomposition:
    """Per-step MI decomposition result."""
    step: int
    mi_lexical: float           # CKA(lexical_features, reward)
    mi_structural: float        # CKA(structural_features, reward)
    mi_ratio: float             # lexical dominance ratio
    top_words: list[tuple[str, float]] = field(default_factory=list)
    n_samples: int = 0
    vocab_size: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "mi_lexical": round(self.mi_lexical, 4),
            "mi_structural": round(self.mi_structural, 4),
            "mi_ratio": round(self.mi_ratio, 4),
            "top_words": [(w, round(s, 4)) for w, s in self.top_words[:10]],
            "n_samples": self.n_samples,
            "vocab_size": self.vocab_size,
        }

    def summary_text(self) -> str:
        """Format for injection into agent detector prompt."""
        lines = [
            "MI Decomposition (CKA between features and reward):",
            f"  CKA(lexical, reward)    = {self.mi_lexical:.4f}",
            f"  CKA(structural, reward) = {self.mi_structural:.4f}",
            f"  Lexical dominance ratio = {self.mi_ratio:.4f}",
        ]
        if self.top_words:
            words_str = ", ".join(f'"{w}"({s:.3f})' for w, s in self.top_words[:10])
            lines.append(f"  Top reward-correlated words: {words_str}")
        return "\n".join(lines)


_NULL_DECOMPOSITION = MIDecomposition(step=0, mi_lexical=0.0, mi_structural=0.0, mi_ratio=0.0)


# ---------------------------------------------------------------------------
# Feature matrix builders
# ---------------------------------------------------------------------------

_CONTINUOUS_FEATURES = ["char_length", "word_count", "sentence_count",
                        "type_token_ratio", "trigram_repetition", "bigram_repetition"]
_BINARY_FEATURES = ["has_markdown_headers", "has_numbered_list",
                     "has_bullet_list", "has_disclaimer"]


def _build_structural_matrix(texts: list[str]) -> np.ndarray:
    """Build structural feature matrix (n_samples, 10) from response texts."""
    features = [extract_features(t) for t in texts]
    n = len(features)

    # Continuous columns: z-score standardize
    continuous = np.zeros((n, len(_CONTINUOUS_FEATURES)))
    for j, key in enumerate(_CONTINUOUS_FEATURES):
        col = np.array([float(f[key]) for f in features])
        std = col.std()
        if std > 1e-8:
            col = (col - col.mean()) / std
        else:
            col = np.zeros(n)
        continuous[:, j] = col

    # Binary columns: as-is
    binary = np.zeros((n, len(_BINARY_FEATURES)))
    for j, key in enumerate(_BINARY_FEATURES):
        binary[:, j] = np.array([float(f[key]) for f in features])

    return np.hstack([continuous, binary])


def _discover_vocabulary(texts: list[str], top_k: int = 200,
                         min_df: float = 0.05, max_df: float = 0.80) -> list[str]:
    """Auto-discover top-K vocabulary words by document frequency."""
    n = len(texts)
    if n == 0:
        return []
    min_count = max(1, int(min_df * n))
    max_count = int(max_df * n)

    doc_freq: Counter = Counter()
    for text in texts:
        doc_freq.update(_tokenize_words(text))

    # Filter by df range, sort by frequency
    candidates = [
        (word, count) for word, count in doc_freq.items()
        if min_count <= count <= max_count
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in candidates[:top_k]]


def _build_lexical_matrix(texts: list[str], vocab: list[str]) -> np.ndarray:
    """Build binary indicator matrix (n_samples, vocab_size)."""
    n = len(texts)
    v = len(vocab)
    if v == 0:
        return np.zeros((n, 1))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    matrix = np.zeros((n, v))
    for i, text in enumerate(texts):
        words = _tokenize_words(text)
        for w in words:
            if w in word_to_idx:
                matrix[i, word_to_idx[w]] = 1.0
    return matrix


# ---------------------------------------------------------------------------
# Main decomposer
# ---------------------------------------------------------------------------

class MIDecomposer:
    """Computes MI decomposition per training step."""

    def __init__(self, vocab_top_k: int = 200, min_df: float = 0.05,
                 max_df: float = 0.80, min_samples: int = 30,
                 word_top_k: int = 10):
        self.vocab_top_k = vocab_top_k
        self.min_df = min_df
        self.max_df = max_df
        self.min_samples = min_samples
        self.word_top_k = word_top_k
        self._history: list[MIDecomposition] = []

    def analyze_step(self, step: int, entries: list[dict]) -> MIDecomposition:
        """Compute MI decomposition for one training step."""
        if len(entries) < self.min_samples:
            return MIDecomposition(step=step, mi_lexical=0.0,
                                   mi_structural=0.0, mi_ratio=0.0)

        texts = [strip_think(e.get("output", "")) for e in entries]
        rewards = np.array([float(e.get("score", 0) or 0) for e in entries])

        # Skip if no variance in rewards
        if rewards.std() < 1e-8:
            return MIDecomposition(step=step, mi_lexical=0.0,
                                   mi_structural=0.0, mi_ratio=0.0,
                                   n_samples=len(entries))

        # Auto-discover vocabulary for this step
        vocab = _discover_vocabulary(texts, top_k=self.vocab_top_k,
                                     min_df=self.min_df, max_df=self.max_df)

        # Build feature matrices
        lex_matrix = _build_lexical_matrix(texts, vocab)
        struct_matrix = _build_structural_matrix(texts)

        # Compute CKA
        Y = rewards.reshape(-1, 1)
        mi_lex = compute_cka(lex_matrix, Y)
        mi_struct = compute_cka(struct_matrix, Y)

        # Lexical dominance ratio
        total = mi_lex + mi_struct + 1e-8
        mi_ratio = mi_lex / total

        # Per-word CKA contribution
        top_words = self._word_contributions(lex_matrix, vocab, Y)

        result = MIDecomposition(
            step=step,
            mi_lexical=mi_lex,
            mi_structural=mi_struct,
            mi_ratio=mi_ratio,
            top_words=top_words,
            n_samples=len(entries),
            vocab_size=len(vocab),
        )
        self._history.append(result)
        return result

    def _word_contributions(self, lex_matrix: np.ndarray, vocab: list[str],
                            Y: np.ndarray) -> list[tuple[str, float]]:
        """Per-word CKA: rank which individual words correlate most with reward."""
        contributions: list[tuple[str, float]] = []
        for j, word in enumerate(vocab):
            col = lex_matrix[:, j]
            # Skip words with no variance (all 0 or all 1)
            if col.std() < 1e-8:
                continue
            cka = compute_cka(col.reshape(-1, 1), Y)
            if cka > 0.001:
                contributions.append((word, cka))

        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:self.word_top_k]

    def get_history(self) -> list[MIDecomposition]:
        return list(self._history)
