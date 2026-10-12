"""Baseline chunkers used to compare against :class:`CannySemanticChunker`.

Every baseline implements the same tiny interface as the main chunker::

    chunk(text: str) -> List[str]

so they can be swapped into the evaluation runner without further glue.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pysbd

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

__all__ = [
    "BaselineChunker",
    "FixedSizeChunker",
    "SingleThresholdChunker",
    "PercentileThresholdChunker",
]


def _whitespace_tokens(text: str) -> int:
    """Default cheap token counter (whitespace split)."""
    return len(text.split())


class BaselineChunker:
    """Common sentence segmentation / tokenisation for all baselines."""

    def __init__(
        self,
        language: str = "en",
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.language = language
        self.seg = pysbd.Segmenter(language=language, clean=False)
        self.token_counter = token_counter or _whitespace_tokens

    def segment(self, text: Optional[str]) -> List[str]:
        """Segment ``text`` into non-empty, stripped sentences."""
        if not text:
            return []
        text = text.strip()
        if not text:
            return []
        return [s.strip() for s in self.seg.segment(text) if s and s.strip()]

    def count_tokens(self, text: str) -> int:
        """Count tokens with the configured counter."""
        return int(self.token_counter(text))

    def chunk(self, text: str) -> List[str]:  # pragma: no cover - interface
        """Split ``text`` into chunks (implemented by subclasses)."""
        raise NotImplementedError


class FixedSizeChunker(BaselineChunker):
    """Greedy sentence packing up to ``max_tokens`` (optionally overlapping).

    Args:
        max_tokens: Token budget per chunk.
        overlap_sentences: Number of trailing sentences repeated at the start
            of the next chunk (``0`` = no overlap).
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_sentences: int = 0,
        language: str = "en",
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        super().__init__(language=language, token_counter=token_counter)
        self.max_tokens = max_tokens
        self.overlap_sentences = max(0, int(overlap_sentences))

    def chunk(self, text: str) -> List[str]:
        """Pack sentences greedily without looking at their meaning."""
        sentences = self.segment(text)
        if not sentences:
            return []
        current: List[Tuple[str, int]] = []
        current_tokens = 0
        chunks: List[str] = []
        for sentence in sentences:
            tokens = self.count_tokens(sentence)
            if current and current_tokens + tokens > self.max_tokens:
                chunks.append(" ".join(s for s, _ in current))
                if self.overlap_sentences > 0:
                    current = current[-self.overlap_sentences :]
                    current_tokens = sum(t for _, t in current)
                else:
                    current = []
                    current_tokens = 0
            current.append((sentence, tokens))
            current_tokens += tokens
        if current:
            chunks.append(" ".join(s for s, _ in current))
        return chunks


class _EmbeddingBaselineChunker(BaselineChunker):
    """Shared embedding logic for the threshold-based baselines.

    Unlike :class:`CannySemanticChunker` these baselines have **no** gray zone,
    no centroid verification and no adaptive drift threshold - they exist to
    isolate the contribution of those mechanisms during evaluation.
    """

    def __init__(
        self,
        model: Optional[object] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        language: str = "en",
        token_counter: Optional[Callable[[str], int]] = None,
        min_sentences: int = 1,
        max_sentences: int = 1000,
    ) -> None:
        super().__init__(language=language, token_counter=token_counter)
        self.model_name = model_name
        self._model = model
        self.min_sentences = max(1, int(min_sentences))
        self.max_sentences = max(1, int(max_sentences))

    @property
    def model(self) -> object:
        """Lazily instantiate (and cache) the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @model.setter
    def model(self, value: object) -> None:
        """Allow injecting a pre-built / fake model."""
        self._model = value

    def _embed(self, sentences: List[str]) -> np.ndarray:
        """Return an L2-normalized ``(n, dim)`` embedding matrix."""
        embeddings = self.model.encode(
            sentences, batch_size=32, show_progress_bar=False,
            normalize_embeddings=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _consecutive_distances(self, sentences: List[str]) -> np.ndarray:
        """Cosine distance between consecutive sentences."""
        embeddings = self._embed(sentences)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        return np.nan_to_num(np.clip(1.0 - similarities, 0.0, 2.0))

    def _threshold(self, distances: np.ndarray) -> float:
        """Return the split threshold (implemented by subclasses)."""
        raise NotImplementedError

    def chunk(self, text: str) -> List[str]:
        """Split whenever the consecutive distance exceeds the threshold."""
        sentences = self.segment(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [" ".join(sentences)]
        distances = self._consecutive_distances(sentences)
        threshold = self._threshold(distances)
        chunks: List[str] = []
        start = 0
        for i, distance in enumerate(distances):
            cur_count = (i + 1) - start
            if cur_count >= self.max_sentences or (
                distance > threshold and cur_count >= self.min_sentences
            ):
                chunks.append(" ".join(sentences[start : i + 1]))
                start = i + 1
        chunks.append(" ".join(sentences[start:]))
        return chunks


class SingleThresholdChunker(_EmbeddingBaselineChunker):
    """Split when the consecutive distance exceeds a fixed ``threshold``."""

    def __init__(self, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.threshold = float(threshold)

    def _threshold(self, distances: np.ndarray) -> float:
        return self.threshold


class PercentileThresholdChunker(_EmbeddingBaselineChunker):
    """Split above the ``percentile`` of the document distance distribution."""

    def __init__(self, percentile: float = 85.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.percentile = float(percentile)

    def _threshold(self, distances: np.ndarray) -> float:
        if distances.size == 0:
            return 1.0
        return float(np.percentile(distances, self.percentile))
