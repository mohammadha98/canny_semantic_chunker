"""Deterministic test doubles for the chunker and the evaluation package.

The real ``SentenceTransformer`` downloads torch weights (slow, network
dependent), so the tests inject :class:`FakeSentenceTransformer` instead. The
fake delegates the actual vector computation to a ``vector_fn``, which lets each
test control the exact cosine distances the chunker sees.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from canny_chunker import CannySemanticChunker
from evaluation.offline_model import HashingTokenizer

__all__ = [
    "FakeSentenceTransformer",
    "FakeCrossEncoder",
    "keyword_vectors",
    "lookup_vectors",
    "unit",
    "make_chunker",
    "DEFAULT_TEST_KWARGS",
]

#: Keyword arguments shared by most chunker tests: manual thresholds, no size
#: constraints kicking in, cheap token counting.
DEFAULT_TEST_KWARGS: Dict[str, Any] = {
    "adaptive_threshold": False,
    "manual_low_threshold": 0.4,
    "manual_high_threshold": 0.6,
    "min_sentences": 1,
    "min_tokens": 0,
    "max_sentences": 1000,
    "max_tokens": 512,
    "exact_token_count": False,
}


class FakeSentenceTransformer:
    """Minimal ``SentenceTransformer`` stand-in driven by ``vector_fn``."""

    def __init__(
        self,
        vector_fn: Callable[[str], Sequence[float]],
        max_seq_length: int = 512,
    ) -> None:
        self.vector_fn = vector_fn
        self.max_seq_length = max_seq_length
        self.tokenizer = HashingTokenizer(max_seq_length=max_seq_length)
        self.encode_calls: List[List[str]] = []

    def encode(
        self,
        sentences: Any,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode a single sentence or a list of sentences."""
        del batch_size, show_progress_bar
        single = isinstance(sentences, str)
        items = [sentences] if single else list(sentences)
        self.encode_calls.append(list(items))
        matrix = np.asarray(
            [np.asarray(self.vector_fn(item), dtype=np.float64) for item in items],
            dtype=np.float64,
        )
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix[0] if single else matrix

    def get_sentence_embedding_dimension(self) -> int:
        """Expose the fake embedding dimensionality."""
        return int(np.asarray(self.vector_fn("probe"), dtype=np.float64).shape[-1])


class FakeCrossEncoder:
    """Stand-in for ``sentence_transformers.CrossEncoder``.

    Args:
        logit: Constant raw logit returned for every ``(context, candidate)``
            pair. Use a large negative value to force a split, a large positive
            value to force a merge.
    """

    def __init__(self, logit: float = 0.0) -> None:
        self.logit = float(logit)
        self.calls: List[List[Tuple[str, str]]] = []

    def predict(self, pairs: Iterable[Tuple[str, str]]) -> np.ndarray:
        """Return the configured logit for each pair."""
        pairs = list(pairs)
        self.calls.append(pairs)
        return np.asarray([self.logit] * len(pairs), dtype=np.float64)


def keyword_vectors(keywords: Dict[str, Sequence[str]]) -> Callable[[str], List[float]]:
    """Build a ``vector_fn`` assigning a one-hot vector by keyword match.

    Args:
        keywords: ``topic -> [keywords]``. The last (extra) dimension is used
            for sentences that match no topic.

    Example:
        With ``{"space": ["rocket"], "food": ["bread"]}``, a sentence containing
        ``rocket`` maps to ``[1, 0, 0]`` and one containing ``bread`` to
        ``[0, 1, 0]``, so the cosine distance between the two topics is ``1.0``
        and ``0.0`` within a topic.
    """
    topics = list(keywords)
    dim = len(topics) + 1

    def vector_fn(sentence: str) -> List[float]:
        lowered = sentence.lower()
        for index, topic in enumerate(topics):
            if any(word.lower() in lowered for word in keywords[topic]):
                vector = [0.0] * dim
                vector[index] = 1.0
                return vector
        vector = [0.0] * dim
        vector[-1] = 1.0
        return vector

    return vector_fn


def lookup_vectors(
    mapping: Dict[str, Sequence[float]],
    default: Sequence[float] = (1.0, 0.0),
) -> Callable[[str], List[float]]:
    """Build a ``vector_fn`` backed by an explicit sentence -> vector mapping."""

    def vector_fn(sentence: str) -> List[float]:
        return list(mapping.get(sentence, default))

    return vector_fn


def unit(vector: Sequence[float]) -> List[float]:
    """Return the L2-normalized ``vector`` as a list."""
    array = np.asarray(vector, dtype=np.float64)
    return (array / np.linalg.norm(array)).tolist()


def make_chunker(model: Any, **kwargs: Any) -> CannySemanticChunker:
    """Create a chunker wired to ``model`` (no network access)."""
    chunker = CannySemanticChunker(**kwargs)
    chunker.model = model
    return chunker
