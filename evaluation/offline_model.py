"""A dependency-free stand-in for ``sentence_transformers.SentenceTransformer``.

The real bi-encoder requires downloading torch weights, which is impractical for
unit tests and for the ``--demo`` evaluation mode. :class:`HashingEmbeddingModel`
produces **deterministic** bag-of-words embeddings using the hashing trick, so
sentences that share vocabulary end up close together and topic shifts produce
larger cosine distances - enough to exercise the whole chunking pipeline.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence, Union

import numpy as np

__all__ = ["HashingEmbeddingModel", "HashingTokenizer", "EMBEDDING_DIM"]

#: Default embedding dimensionality of the offline model.
EMBEDDING_DIM = 64

_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+", re.UNICODE)


class HashingTokenizer:
    """Minimal tokenizer exposing the ``encode`` method the chunker needs."""

    def __init__(self, max_seq_length: int = 512) -> None:
        self.max_seq_length = max_seq_length

    def encode(self, text: str, add_special_tokens: bool = False) -> List[str]:
        """Return the whitespace/punctuation tokens of ``text``.

        ``add_special_tokens`` is accepted for API compatibility and ignored.
        """
        return _TOKEN_RE.findall(text or "")


class HashingEmbeddingModel:
    """Deterministic hashing bag-of-words "embedding" model.

    Args:
        dim: Embedding dimensionality.
        max_seq_length: Mirrors the real model attribute used for the effective
            ``max_tokens`` cap.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, max_seq_length: int = 512) -> None:
        self.dim = int(dim)
        self.max_seq_length = int(max_seq_length)
        self.tokenizer = HashingTokenizer(max_seq_length=max_seq_length)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        tokens = self.tokenizer.encode(text)
        for token in tokens:
            digest = hashlib.md5(token.lower().encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode one sentence or a list of sentences into a matrix."""
        del batch_size, show_progress_bar  # unused, kept for API parity
        if isinstance(sentences, str):
            return self._vector(sentences)
        return np.asarray([self._vector(s) for s in sentences], dtype=np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        """Mimic the real model API used for introspection."""
        return self.dim
