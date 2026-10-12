"""Boundary-based evaluation metrics and chunk statistics.

Conventions
-----------
A segmentation of ``N`` sentences is represented by a **boundary list** of
length ``N - 1`` where ``boundaries[i] == 1`` means "a boundary exists between
sentence ``i`` and sentence ``i + 1``".

Both ``Pk`` (Beeferman et al., 1999) and ``WindowDiff`` (Pevzner & Hearst,
2002) slide a window of ``k`` sentences over the text and compare the reference
and hypothesis segmentations. Let ``B(i, i + k)`` be the number of boundaries
between sentence ``i`` and sentence ``i + k``:

* ``Pk``         penalises a window when ``(B_ref == 0) != (B_hyp == 0)``
  (the two segmentations disagree on whether the window is "boundary free").
* ``WindowDiff`` penalises a window when ``B_ref != B_hyp`` (strictly stronger
  than Pk because it also catches differing boundary *counts*).

Lower is better; both are ``0.0`` for a perfect segmentation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pysbd

__all__ = [
    "pk",
    "windowdiff",
    "chunk_size_stats",
    "mean_intra_chunk_distance",
    "boundaries_from_segment_counts",
]


def _as_int_array(boundaries: Sequence[int]) -> np.ndarray:
    """Validate and convert a boundary list to a 0/1 ``int64`` array."""
    array = np.asarray(boundaries, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("boundaries must be a 1-D sequence")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("boundaries must only contain 0 and 1")
    return array


def _default_k(n_sentences: int, n_boundaries: int) -> int:
    """Standard window size: ``N / (2 * #boundaries)`` rounded, >= 1."""
    if n_boundaries > 0:
        k = int(round(n_sentences / (2.0 * n_boundaries)))
    else:
        k = int(round(n_sentences / 2.0))
    return max(1, k)


def _cumulative(boundaries: np.ndarray) -> np.ndarray:
    """Return ``cum[m] = #boundaries among the first ``m`` boundaries``."""
    return np.concatenate([[0], np.cumsum(boundaries)])


def _validate_pair(
    reference: Sequence[int], hypothesis: Sequence[int]
) -> "tuple[np.ndarray, np.ndarray]":
    ref = _as_int_array(reference)
    hyp = _as_int_array(hypothesis)
    if ref.shape != hyp.shape:
        raise ValueError(
            "reference and hypothesis must have the same number of boundaries"
        )
    return ref, hyp


def pk(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    k: Optional[int] = None,
) -> float:
    """Compute the Pk boundary metric (lower is better).

    Args:
        reference: Reference boundary list (0/1, length ``N - 1``).
        hypothesis: Hypothesis boundary list (0/1, length ``N - 1``).
        k: Window size in sentences. When ``None`` the standard
            ``N / (2 * #reference_boundaries)`` is used.
    """
    ref, hyp = _validate_pair(reference, hypothesis)
    n_sentences = len(ref) + 1
    if k is None:
        k = _default_k(n_sentences, int(ref.sum()))
    k = max(1, min(k, n_sentences - 1))
    ref_cum = _cumulative(ref)
    hyp_cum = _cumulative(hyp)
    total = 0
    disagree = 0
    for i in range(n_sentences - k):
        ref_count = ref_cum[i + k] - ref_cum[i]
        hyp_count = hyp_cum[i + k] - hyp_cum[i]
        if (ref_count == 0) != (hyp_count == 0):
            disagree += 1
        total += 1
    return disagree / total if total else 0.0


def windowdiff(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    k: Optional[int] = None,
) -> float:
    """Compute the WindowDiff boundary metric (lower is better).

    Same signature as :func:`pk`; see the module docstring for the difference.
    """
    ref, hyp = _validate_pair(reference, hypothesis)
    n_sentences = len(ref) + 1
    if k is None:
        k = _default_k(n_sentences, int(ref.sum()))
    k = max(1, min(k, n_sentences - 1))
    ref_cum = _cumulative(ref)
    hyp_cum = _cumulative(hyp)
    total = 0
    disagree = 0
    for i in range(n_sentences - k):
        ref_count = ref_cum[i + k] - ref_cum[i]
        hyp_count = hyp_cum[i + k] - hyp_cum[i]
        if ref_count != hyp_count:
            disagree += 1
        total += 1
    return disagree / total if total else 0.0


def boundaries_from_segment_counts(counts: Sequence[int]) -> List[int]:
    """Convert per-chunk sentence counts into a boundary list.

    Example: ``[3, 2]`` -> ``[0, 0, 1, 0]`` (boundary after sentence index 2).

    The result always has ``sum(counts) - 1`` entries, i.e. it is aligned with
    the ``N - 1`` inter-sentence boundaries of an ``N`` sentence document.
    """
    counts = list(counts)
    boundaries: List[int] = []
    for count in counts[:-1]:
        boundaries.extend([0] * (count - 1))
        boundaries.append(1)
    if counts:
        boundaries.extend([0] * (counts[-1] - 1))
    return boundaries


def chunk_size_stats(
    chunks: Sequence[str],
    token_counter: Optional[Callable[[str], int]] = None,
) -> Dict[str, float]:
    """Return basic token-size statistics for ``chunks``.

    Args:
        chunks: The chunk texts.
        token_counter: Optional token counting callable; defaults to whitespace
            splitting.
    """
    if token_counter is None:

        def token_counter(text: str) -> int:  # noqa: D401 - local default
            return len(text.split())

    tokens = np.asarray([token_counter(chunk) for chunk in chunks], dtype=float)
    if tokens.size == 0:
        return {
            "n_chunks": 0,
            "mean_tokens": 0.0,
            "std_tokens": 0.0,
            "min_tokens": 0.0,
            "max_tokens": 0.0,
        }
    return {
        "n_chunks": int(tokens.size),
        "mean_tokens": float(tokens.mean()),
        "std_tokens": float(tokens.std()),
        "min_tokens": float(tokens.min()),
        "max_tokens": float(tokens.max()),
    }


def mean_intra_chunk_distance(
    chunks: Sequence[str],
    model: object,
    language: str = "en",
) -> float:
    """Mean cosine distance of sentences to their chunk centroid.

    Args:
        chunks: The chunk texts.
        model: Any object exposing ``encode(list[str], normalize_embeddings=...)``
            (a real or fake SentenceTransformer).
        language: Language for the ``pysbd`` sentence segmenter.

    Returns:
        The average over chunks of the mean sentence-to-centroid distance.
        Returns ``0.0`` when no chunk has more than one sentence.
    """
    segmenter = pysbd.Segmenter(language=language, clean=False)
    per_chunk: List[float] = []
    for chunk in chunks:
        sentences = [s.strip() for s in segmenter.segment(chunk) if s and s.strip()]
        if len(sentences) < 2:
            continue
        embeddings = np.asarray(
            model.encode(sentences, normalize_embeddings=True), dtype=np.float64
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        centroid = embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        per_chunk.append(float(np.mean(1.0 - (embeddings @ centroid))))
    return float(np.mean(per_chunk)) if per_chunk else 0.0
