"""Canny-inspired semantic text chunking for RAG.

Algorithm: **dual-threshold with centroid verification**.

Consecutive sentences are embedded with a bi-encoder (L2-normalized) and the
cosine distance between neighbouring sentences drives the segmentation:

* ``dist < low_threshold``          -> merge (stay in the same chunk)
* ``dist > high_threshold``         -> split (start a new chunk)
* ``low <= dist <= high`` (gray zone) -> verify the candidate sentence against
  the **centroid of the whole current chunk**; if the distance to that centroid
  exceeds ``drift_threshold`` the boundary is a gradual topic drift and the
  chunk is split, otherwise the sentences are merged.

Two refinements are available:

* ``centroid_window_size`` (int) restricts the centroid to the last *n*
  sentences (sliding window) for ablation studies; ``None`` (the default) uses
  the whole current chunk.
* ``hysteresis_mode`` confirms a gray-zone boundary as a split only when a
  strong boundary (``dist > high_threshold``) exists within ``hysteresis_radius``
  sentences *or* the centroid drift check fails.

The chunker is fully deterministic (no randomness).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pysbd

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing torch
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- Split reason taxonomy (kept as constants to avoid typos) ----------------
REASON_THRESHOLD = "threshold"
REASON_SEMANTIC_DRIFT = "semantic_drift"
REASON_MAX_SENTENCES = "max_sentences"
REASON_MAX_TOKENS = "max_tokens"
REASON_END_OF_TEXT = "end_of_text"

# --- Diagnostic decision labels ---------------------------------------------
DECISION_MERGE = "merge"
DECISION_SPLIT = "split"
DECISION_GRAY_MERGE = "gray->merge"
DECISION_GRAY_SPLIT = "gray->split"
DECISION_SUPPRESSED = "suppressed"
DECISION_FORCED = "forced"


@dataclass
class ChunkResult:
    """A single produced chunk plus the metadata used to analyse it.

    Attributes:
        text: The chunk text (sentences joined with a single space).
        sentences: The exact sentence strings contained in this chunk.
        sentence_indices: ``(start, end)`` **inclusive** indexes of the chunk
            sentences inside the full (segmented) sentence list. Guaranteed to
            satisfy ``full_sentences[start : end + 1] == sentences``.
        token_count: Number of tokens in the chunk.
        centroid: L2-normalized centroid embedding of the chunk.
        avg_internal_distance: Mean cosine distance of the chunk sentences to
            the chunk centroid (0.0 for a single-sentence chunk).
        split_reason: Why this chunk was closed. One of ``"threshold"``,
            ``"semantic_drift"``, ``"max_sentences"``, ``"max_tokens"``,
            ``"end_of_text"``.
        forced_by_max_size: ``True`` when the chunk was closed purely because a
            maximum size constraint was hit (may hide a real semantic boundary).
        dist_at_boundary: The consecutive-sentence distance observed at the
            boundary that closed this chunk (``None`` for the last chunk).
        suppressed_splits: Number of threshold/drift splits that were requested
            at the end of this chunk but ignored because ``min_sentences`` /
            ``min_tokens`` were not satisfied.
    """

    text: str
    sentences: List[str]
    sentence_indices: Tuple[int, int]
    token_count: int
    centroid: np.ndarray
    avg_internal_distance: float
    split_reason: str
    forced_by_max_size: bool = False
    dist_at_boundary: Optional[float] = None
    suppressed_splits: int = 0


class CannySemanticChunker:
    """Semantic chunker: dual-threshold segmentation with centroid verification.

    The class is model-agnostic: the underlying :class:`SentenceTransformer` is
    loaded lazily on first use and can be replaced by assigning to
    :attr:`model` (used by the unit tests to inject a fake encoder).

    Args:
        model_name: SentenceTransformer model id used for embeddings.
        language: Language passed to the ``pysbd`` sentence segmenter.
        adaptive_threshold: When ``True`` the low/high thresholds are derived
            from the distance distribution via percentiles.
        low_percentile: Percentile used for ``low_threshold`` in adaptive mode.
        high_percentile: Percentile used for ``high_threshold`` in adaptive mode.
        manual_low_threshold: Overrides the low threshold when set (together
            with ``manual_high_threshold``).
        manual_high_threshold: Overrides the high threshold when set.
        drift_threshold: Sentence-to-centroid distance above which the gray zone
            is considered a topic drift. ``None`` (default) derives it
            adaptively from ``drift_percentile``.
        drift_percentile: Percentile used for the adaptive drift threshold.
        centroid_window_size: ``None`` (default) uses the whole current chunk to
            build the centroid; an ``int`` keeps only the last *n* sentences
            (sliding window, useful for ablation).
        hysteresis_mode: Enable the hysteresis split strategy (see module doc).
        hysteresis_radius: +/- window (in sentences) searched for a strong
            boundary when ``hysteresis_mode`` is enabled.
        min_sentences: Minimum number of sentences required to allow a split.
        max_sentences: Maximum number of sentences per chunk (force split).
        min_tokens: Minimum number of tokens required to allow a split.
        max_tokens: Requested maximum number of tokens per chunk. The effective
            limit is ``min(max_tokens, model.max_seq_length)`` so that a chunk
            never exceeds what the embedding model can read (documented default
            is ``512``).
        low_floor: Absolute floor applied to the adaptive ``low_threshold``.
        high_floor: Absolute floor applied to the adaptive ``high_threshold``.
        drift_floor: Absolute floor for the adaptive drift threshold. Defaults
            to ``high_floor`` when ``None``.
        min_sentences_for_adaptive: Minimum number of distances required to use
            adaptive thresholds; below it the fixed default thresholds are used
            (protects short documents from noisy percentiles).
        exact_token_count: When ``True`` (default) tokens are counted with the
            model tokenizer *without* special tokens. ``False`` uses a cheap
            whitespace split.
    """

    #: Fixed fallback thresholds used for very short documents.
    DEFAULT_LOW_THRESHOLD = 0.6
    DEFAULT_HIGH_THRESHOLD = 0.8
    #: Minimum gap enforced between low and high thresholds.
    _MIN_THRESHOLD_GAP = 0.01

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        language: str = "en",
        adaptive_threshold: bool = True,
        low_percentile: int = 60,
        high_percentile: int = 85,
        manual_low_threshold: Optional[float] = None,
        manual_high_threshold: Optional[float] = None,
        # --- centroid / drift -------------------------------------------------
        drift_threshold: Optional[float] = None,
        drift_percentile: int = 90,
        centroid_window_size: Optional[int] = None,
        # --- hysteresis -------------------------------------------------------
        hysteresis_mode: bool = False,
        hysteresis_radius: int = 2,
        # --- size constraints -------------------------------------------------
        min_sentences: int = 2,
        max_sentences: int = 15,
        min_tokens: int = 50,
        max_tokens: int = 512,
        # --- adaptive floors --------------------------------------------------
        low_floor: float = 0.15,
        high_floor: float = 0.30,
        drift_floor: Optional[float] = None,
        min_sentences_for_adaptive: int = 20,
        # --- tokenisation -----------------------------------------------------
        exact_token_count: bool = True,
    ) -> None:
        self.model_name = model_name
        self._model: Optional["SentenceTransformer"] = None  # lazy loading
        self.seg = pysbd.Segmenter(language=language, clean=False)
        self.language = language

        # Threshold settings
        self.adaptive_threshold = adaptive_threshold
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.manual_low_threshold = manual_low_threshold
        self.manual_high_threshold = manual_high_threshold
        self.default_low_threshold = self.DEFAULT_LOW_THRESHOLD
        self.default_high_threshold = self.DEFAULT_HIGH_THRESHOLD
        self.low_floor = low_floor
        self.high_floor = high_floor
        self.min_sentences_for_adaptive = min_sentences_for_adaptive

        # Centroid / drift settings
        self.centroid_window_size = centroid_window_size
        self.drift_threshold = drift_threshold
        self.drift_percentile = drift_percentile
        self.drift_floor = high_floor if drift_floor is None else drift_floor

        # Hysteresis settings
        self.hysteresis_mode = hysteresis_mode
        self.hysteresis_radius = max(0, int(hysteresis_radius))

        # Size constraints
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        # Tokenisation
        self.exact_token_count = exact_token_count

    # ------------------------------------------------------------------ model
    @property
    def model(self) -> "SentenceTransformer":
        """Lazily instantiate (and cache) the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        """Allow injecting a pre-built / fake model (used by the tests)."""
        self._model = value

    def _model_max_seq_length(self) -> Optional[int]:
        """Return the model ``max_seq_length`` if the model exposes one."""
        try:
            max_len = getattr(self.model, "max_seq_length", None)
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(max_len, (int, np.integer)) and max_len > 0:
            return int(max_len)
        return None

    def _effective_max_tokens(self) -> int:
        """``max_tokens`` capped by the model ``max_seq_length``."""
        model_max = self._model_max_seq_length()
        if model_max is not None:
            return min(self.max_tokens, model_max)
        return self.max_tokens

    # ----------------------------------------------------------- tokenisation
    def _count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` (exact tokenizer or whitespace fallback)."""
        if not text:
            return 0
        if self.exact_token_count:
            try:
                tokenizer = self.model.tokenizer
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:  # pragma: no cover - fall back gracefully
                return len(text.split())
        return len(text.split())

    def _warn_on_long_sentences(
        self, sentences: List[str], token_counts: List[int]
    ) -> None:
        """Warn when a sentence is longer than the model's ``max_seq_length``.

        Such sentences are silently truncated by the encoder, which distorts
        their embedding, so the user is told explicitly.
        """
        max_len = self._model_max_seq_length()
        if max_len is None:
            return
        for sentence, count in zip(sentences, token_counts):
            if count > max_len:
                logger.warning(
                    "Sentence of %d tokens exceeds model max_seq_length=%d and "
                    "will be truncated by the encoder: %r...",
                    count,
                    max_len,
                    sentence[:60],
                )

    # ------------------------------------------------------------ embeddings
    def _segment(self, text: Optional[str]) -> List[str]:
        """Segment ``text`` into non-empty, stripped sentences."""
        if text is None:
            return []
        text = text.strip()
        if not text:
            return []
        return [s.strip() for s in self.seg.segment(text) if s and s.strip()]

    def _encode(self, sentences: List[str]) -> np.ndarray:
        """Encode ``sentences`` into an L2-normalized ``(n, dim)`` array."""
        try:
            embeddings = self.model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except TypeError:  # pragma: no cover - older sentence-transformers
            embeddings = self.model.encode(
                sentences, batch_size=32, show_progress_bar=False
            )
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @staticmethod
    def _pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
        """Vectorized cosine distance between consecutive normalized vectors."""
        if len(embeddings) < 2:
            return np.empty(0, dtype=np.float64)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        distances = 1.0 - similarities
        return np.nan_to_num(np.clip(distances, 0.0, 2.0))

    # ------------------------------------------------------------- thresholds
    def _ensure_ordered(self, low: float, high: float) -> Tuple[float, float]:
        """Guarantee ``low < high`` (nudge ``high`` up if needed)."""
        if low >= high:
            high = low + self._MIN_THRESHOLD_GAP
        return low, high

    def _get_thresholds(self, distances: np.ndarray) -> Tuple[float, float]:
        """Compute the ``(low, high)`` consecutive-distance thresholds.

        Priority: manual thresholds > adaptive percentiles (only when there are
        enough distances) > fixed defaults. Adaptive values are floored by
        ``low_floor`` / ``high_floor`` and finally ordered.
        """
        if (
            self.manual_low_threshold is not None
            and self.manual_high_threshold is not None
        ):
            low, high = self.manual_low_threshold, self.manual_high_threshold
        elif (
            self.adaptive_threshold
            and len(distances) >= self.min_sentences_for_adaptive
        ):
            low = max(float(np.percentile(distances, self.low_percentile)), self.low_floor)
            high = max(
                float(np.percentile(distances, self.high_percentile)), self.high_floor
            )
        else:
            low, high = self.default_low_threshold, self.default_high_threshold
        return self._ensure_ordered(float(low), float(high))

    # ------------------------------------------------------------------ drift
    def _compute_drift_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """First-pass drift distribution used for the adaptive drift threshold.

        For every sentence ``i`` we measure the cosine distance to the centroid
        of its *local context*: the preceding sentences, bounded by
        ``max_sentences``. This yields a document-level estimate of how far a
        sentence typically sits from the running topic centroid, which is a
        sensible scale for the drift threshold (more robust than a fixed
        constant across corpora).
        """
        n = len(embeddings)
        if n < 2:
            return np.empty(0, dtype=np.float64)
        drift_distances: List[float] = []
        for i in range(1, n):
            context = embeddings[max(0, i - self.max_sentences) : i]
            centroid = context.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            drift_distances.append(1.0 - float(np.dot(embeddings[i], centroid)))
        return np.nan_to_num(np.clip(np.asarray(drift_distances), 0.0, 2.0))

    def _get_drift_threshold(
        self, embeddings: np.ndarray, high_threshold: float
    ) -> float:
        """Compute the sentence-to-centroid drift threshold.

        Manual ``drift_threshold`` wins. Otherwise the ``drift_percentile`` of
        the first-pass drift distribution is used, floored by ``drift_floor``.
        For short documents (fewer than ``min_sentences_for_adaptive``
        distances) the ``high_threshold`` is reused as a conservative fallback.
        """
        if self.drift_threshold is not None:
            return float(self.drift_threshold)
        drift_distances = self._compute_drift_distances(embeddings)
        if len(drift_distances) >= self.min_sentences_for_adaptive:
            value = float(np.percentile(drift_distances, self.drift_percentile))
            return float(max(value, self.drift_floor))
        return float(max(high_threshold, self.drift_floor))

    # --------------------------------------------------------------- centroid
    def _current_centroid(
        self,
        running_sum: np.ndarray,
        count: int,
        window_buffer: Optional[deque],
    ) -> np.ndarray:
        """Return the normalized centroid of the current chunk.

        Uses the incremental running sum (O(1), whole chunk) by default, or the
        sliding window buffer when ``centroid_window_size`` is an int.
        """
        if self.centroid_window_size is None or window_buffer is None:
            centroid = running_sum / count
        else:
            centroid = np.mean(np.asarray(window_buffer), axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def _distance_to_centroid(
        self,
        embedding: np.ndarray,
        running_sum: np.ndarray,
        count: int,
        window_buffer: Optional[deque],
    ) -> float:
        """Cosine distance between ``embedding`` and the current chunk centroid."""
        centroid = self._current_centroid(running_sum, count, window_buffer)
        return float(1.0 - np.dot(embedding, centroid))

    # -------------------------------------------------------------- decisions
    def _has_strong_boundary_nearby(
        self, i: int, distances: np.ndarray, high_threshold: float
    ) -> bool:
        """True if a strong boundary (``dist > high``) sits within +/- radius."""
        low = max(0, i - self.hysteresis_radius)
        high = min(len(distances) - 1, i + self.hysteresis_radius)
        for j in range(low, high + 1):
            if j == i:
                continue
            if distances[j] > high_threshold:
                return True
        return False

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
        window_buffer: Optional[deque],
        drift_threshold: float,
        distances: np.ndarray,
    ) -> Tuple[str, Optional[float]]:
        """Decide the semantic action for the boundary after sentence ``i``.

        Returns a tuple ``(decision, centroid_distance)`` where ``decision`` is
        one of ``"merge"``, ``"split_threshold"`` or ``"split_drift"`` and
        ``centroid_distance`` is the (optional) distance to the chunk centroid
        that was evaluated in the gray zone.
        """
        if dist < low_threshold:
            return "merge", None
        if dist > high_threshold:
            return "split_threshold", None

        # Gray zone -> centroid verification.
        centroid_distance = self._distance_to_centroid(
            embeddings[i + 1], running_sum, count, window_buffer
        )
        drift_fails = centroid_distance > drift_threshold

        if self.hysteresis_mode:
            if self._has_strong_boundary_nearby(i, distances, high_threshold):
                return "split_threshold", centroid_distance
            if drift_fails:
                return "split_drift", centroid_distance
            return "merge", centroid_distance

        if drift_fails:
            return "split_drift", centroid_distance
        return "merge", centroid_distance

    # ------------------------------------------------------------ chunk build
    def _build_chunk(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        token_counts: List[int],
        start_idx: int,
        end_idx: int,
        split_reason: str,
        forced_by_max_size: bool = False,
        dist_at_boundary: Optional[float] = None,
        suppressed_splits: int = 0,
    ) -> ChunkResult:
        """Build a :class:`ChunkResult` from ``start_idx`` to ``end_idx`` inclusive.

        Centralising construction fixes the previous first/last chunk
        duplication bug and guarantees ``sentences[start:end+1]`` consistency.
        """
        chunk_sentences = list(sentences[start_idx : end_idx + 1])
        chunk_embeddings = embeddings[start_idx : end_idx + 1]
        centroid = chunk_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        if len(chunk_embeddings) > 1:
            avg_distance = float(np.mean(1.0 - (chunk_embeddings @ centroid)))
        else:
            avg_distance = 0.0
        return ChunkResult(
            text=" ".join(chunk_sentences),
            sentences=chunk_sentences,
            sentence_indices=(start_idx, end_idx),
            token_count=int(sum(token_counts[start_idx : end_idx + 1])),
            centroid=centroid,
            avg_internal_distance=avg_distance,
            split_reason=split_reason,
            forced_by_max_size=forced_by_max_size,
            dist_at_boundary=dist_at_boundary,
            suppressed_splits=suppressed_splits,
        )

    # --------------------------------------------------------------- pipeline
    def _chunk_impl(
        self, text: Optional[str], collect_diagnostics: bool = False
    ) -> Tuple[List[ChunkResult], Dict[str, Any]]:
        """Core chunking routine shared by the public API and diagnostics."""
        diagnostics: Dict[str, Any] = {
            "distances": [],
            "low_threshold": None,
            "high_threshold": None,
            "drift_threshold": None,
            "n_sentences": 0,
            "boundaries": [],
        }
        sentences = self._segment(text)
        if not sentences:
            return [], diagnostics

        token_counts = [self._count_tokens(s) for s in sentences]
        self._warn_on_long_sentences(sentences, token_counts)
        embeddings = self._encode(sentences)
        n = len(sentences)

        if n == 1:
            chunk = self._build_chunk(
                sentences, embeddings, token_counts, 0, 0, REASON_END_OF_TEXT
            )
            diagnostics["n_sentences"] = 1
            return [chunk], diagnostics

        distances = self._pairwise_distances(embeddings)
        low_threshold, high_threshold = self._get_thresholds(distances)
        drift_threshold = self._get_drift_threshold(embeddings, high_threshold)
        effective_max_tokens = self._effective_max_tokens()

        diagnostics["distances"] = distances.tolist()
        diagnostics["low_threshold"] = float(low_threshold)
        diagnostics["high_threshold"] = float(high_threshold)
        diagnostics["drift_threshold"] = float(drift_threshold)
        diagnostics["n_sentences"] = n

        window_size = self.centroid_window_size
        chunks: List[ChunkResult] = []
        start = 0
        running_sum = embeddings[0].copy()
        count = 1
        window_buffer: Optional[deque] = (
            deque([embeddings[0]], maxlen=window_size)
            if window_size is not None
            else None
        )
        current_tokens = token_counts[0]
        suppressed = 0

        for i in range(len(distances)):
            dist = float(distances[i])
            next_tokens = token_counts[i + 1]
            cur_count = (i + 1) - start

            should_split = False
            reason = ""
            forced = False
            decision = DECISION_MERGE
            centroid_distance: Optional[float] = None

            # 1. Hard maximum size constraints (always force a split).
            if cur_count >= self.max_sentences:
                should_split = True
                reason = REASON_MAX_SENTENCES
                forced = True
            elif current_tokens + next_tokens > effective_max_tokens:
                should_split = True
                reason = REASON_MAX_TOKENS
                forced = True
            else:
                # 2. Semantic decision (threshold / gray zone).
                action, centroid_distance = self._decide_boundary(
                    i,
                    dist,
                    low_threshold,
                    high_threshold,
                    sentences,
                    embeddings,
                    start,
                    running_sum,
                    count,
                    window_buffer,
                    drift_threshold,
                    distances,
                )
                if action == "split_threshold":
                    should_split = True
                    reason = REASON_THRESHOLD
                elif action == "split_drift":
                    should_split = True
                    reason = REASON_SEMANTIC_DRIFT

                if should_split:
                    # 3. Minimum constraints suppress the split but are recorded.
                    is_min_satisfied = (cur_count >= self.min_sentences) and (
                        current_tokens >= self.min_tokens
                    )
                    if not is_min_satisfied:
                        should_split = False
                        suppressed += 1
                        decision = DECISION_SUPPRESSED
                    else:
                        decision = (
                            DECISION_SPLIT
                            if dist > high_threshold
                            else DECISION_GRAY_SPLIT
                        )
                else:
                    decision = (
                        DECISION_MERGE if dist < low_threshold else DECISION_GRAY_MERGE
                    )

            if forced:
                decision = DECISION_FORCED

            if collect_diagnostics:
                diagnostics["boundaries"].append(
                    {
                        "index": i,
                        "distance": dist,
                        "low_threshold": float(low_threshold),
                        "high_threshold": float(high_threshold),
                        "drift_threshold": float(drift_threshold),
                        "centroid_distance": (
                            None
                            if centroid_distance is None
                            else float(centroid_distance)
                        ),
                        "decision": decision,
                        "split_reason": reason if should_split else None,
                        "suppressed": decision == DECISION_SUPPRESSED,
                        "forced_by_max_size": forced,
                    }
                )

            if should_split:
                chunks.append(
                    self._build_chunk(
                        sentences,
                        embeddings,
                        token_counts,
                        start,
                        i,
                        reason,
                        forced_by_max_size=forced,
                        dist_at_boundary=dist,
                        suppressed_splits=suppressed,
                    )
                )
                suppressed = 0
                start = i + 1
                running_sum = embeddings[i + 1].copy()
                count = 1
                window_buffer = (
                    deque([embeddings[i + 1]], maxlen=window_size)
                    if window_size is not None
                    else None
                )
                current_tokens = next_tokens
            else:
                running_sum = running_sum + embeddings[i + 1]
                count += 1
                if window_buffer is not None:
                    window_buffer.append(embeddings[i + 1])
                current_tokens += next_tokens

        chunks.append(
            self._build_chunk(
                sentences,
                embeddings,
                token_counts,
                start,
                n - 1,
                REASON_END_OF_TEXT,
                suppressed_splits=suppressed,
            )
        )
        return chunks, diagnostics

    # ------------------------------------------------------------- public API
    def chunk(self, text: str) -> List[str]:
        """Split ``text`` into semantic chunks and return their texts."""
        return [result.text for result in self.chunk_with_metadata(text)]

    def chunk_with_metadata(self, text: str) -> List[ChunkResult]:
        """Split ``text`` into semantic chunks and return rich metadata."""
        chunks, _ = self._chunk_impl(text)
        return chunks

    def get_thresholds(self, text: str) -> Tuple[float, float]:
        """Return the ``(low, high)`` thresholds that would be used for ``text``."""
        sentences = self._segment(text)
        if len(sentences) < 2:
            return self.default_low_threshold, self.default_high_threshold
        embeddings = self._encode(sentences)
        distances = self._pairwise_distances(embeddings)
        return self._get_thresholds(distances)

    def get_drift_threshold(self, text: str) -> float:
        """Return the drift threshold that would be used for ``text``."""
        sentences = self._segment(text)
        if len(sentences) < 2:
            return self.default_high_threshold
        embeddings = self._encode(sentences)
        distances = self._pairwise_distances(embeddings)
        _, high_threshold = self._get_thresholds(distances)
        return self._get_drift_threshold(embeddings, high_threshold)

    def get_diagnostics(self, text: str) -> Dict[str, Any]:
        """Return a full trace of the chunking decisions for ``text``.

        The result contains the consecutive ``distances``, the resolved
        ``low_threshold`` / ``high_threshold`` / ``drift_threshold`` and a
        ``boundaries`` list with one dict per inter-sentence boundary::

            {
                "index": int,                # boundary after sentence index
                "distance": float,           # consecutive-sentence distance
                "low_threshold": float,
                "high_threshold": float,
                "drift_threshold": float,
                "centroid_distance": float | None,  # sentence-to-centroid dist
                "decision": "merge" | "split" | "gray->merge" | "gray->split"
                            | "suppressed" | "forced",
                "split_reason": str | None,
                "suppressed": bool,
                "forced_by_max_size": bool,
            }

        Handy for plotting thresholds vs distances or analysing forced splits.
        """
        _, diagnostics = self._chunk_impl(text, collect_diagnostics=True)
        return diagnostics


__all__ = [
    "CannySemanticChunker",
    "ChunkResult",
    "REASON_THRESHOLD",
    "REASON_SEMANTIC_DRIFT",
    "REASON_MAX_SENTENCES",
    "REASON_MAX_TOKENS",
    "REASON_END_OF_TEXT",
]

