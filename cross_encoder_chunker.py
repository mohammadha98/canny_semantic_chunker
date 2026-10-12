"""Optional Cross-Encoder verified chunker.

This module is intentionally *separate* from :mod:`canny_chunker`. The main
chunker relies on a bi-encoder + centroid verification, which is the correct
tool for *topic continuity*. Cross-Encoder models such as
``cross-encoder/ms-marco-MiniLM-L-6-v2`` are trained for **query-passage
relevance**: they output raw logits (not probabilities) and are not designed to
judge whether two consecutive passages belong to the same topic.

The subclass below is therefore opt-in and fixes the two bugs of the previous
implementation:

1. It uses a *valid* model name (``cross-encoder/ms-marco-MiniLM-L-6-v2``).
2. It applies a sigmoid to the raw logits so the score is a probability.

Even with those fixes the cross-encoder only replaces the gray-zone decision;
the dual-threshold + size-constraint logic is inherited unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from canny_chunker import CannySemanticChunker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections import deque

    from sentence_transformers import CrossEncoder

#: A small, valid, multilingual ms-marco cross-encoder.
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderVerifiedChunker(CannySemanticChunker):
    """Chunker whose gray-zone decision is delegated to a Cross-Encoder.

    Args:
        cross_encoder_model: Cross-Encoder model id (must be a valid one).
        cross_encoder_threshold: Sigmoid probability below which the candidate
            sentence is considered off-topic and the chunk is split.
        cross_encoder_context_sentences: Number of trailing current-chunk
            sentences used as the *context* passage.
        **kwargs: Forwarded to :class:`CannySemanticChunker`.
    """

    def __init__(
        self,
        *,
        cross_encoder_model: Optional[str] = None,
        cross_encoder_threshold: float = 0.5,
        cross_encoder_context_sentences: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cross_encoder_model_name = cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL
        self.cross_encoder_threshold = cross_encoder_threshold
        self.cross_encoder_context_sentences = cross_encoder_context_sentences
        self._cross_encoder: Optional["CrossEncoder"] = None

    @property
    def cross_encoder(self) -> "CrossEncoder":
        """Lazily instantiate (and cache) the CrossEncoder model."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
        return self._cross_encoder

    @cross_encoder.setter
    def cross_encoder(self, value: Any) -> None:
        """Allow injecting a fake cross-encoder (used by the tests)."""
        self._cross_encoder = value

    @staticmethod
    def _sigmoid(logit: float) -> float:
        """Numerically stable logistic function."""
        if logit >= 0:
            z = np.exp(-logit)
            return float(1.0 / (1.0 + z))
        z = np.exp(logit)
        return float(z / (1.0 + z))

    def _decide_boundary(
        self,
        i: int,
        dist: float,
        low_threshold: float,
        high_threshold: float,
        sentences: List[str],
        embeddings: np.ndarray,
        start: int,
        running_sum: np.ndarray,
        count: int,
        window_buffer: Optional["deque"],
        drift_threshold: float,
        distances: np.ndarray,
    ) -> Tuple[str, Optional[float]]:
        """Split in the gray zone when the Cross-Encoder probability is low."""
        if dist < low_threshold:
            return "merge", None
        if dist > high_threshold:
            return "split_threshold", None

        context_start = max(start, i + 1 - self.cross_encoder_context_sentences)
        context = " ".join(sentences[context_start : i + 1])
        candidate = sentences[i + 1]
        logit = float(self.cross_encoder.predict([(context, candidate)])[0])
        probability = self._sigmoid(logit)
        if probability < self.cross_encoder_threshold:
            return "split_drift", None
        return "merge", None


__all__ = ["CrossEncoderVerifiedChunker", "DEFAULT_CROSS_ENCODER_MODEL"]
