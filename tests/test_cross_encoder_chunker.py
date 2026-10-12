"""Tests for the opt-in Cross-Encoder verified chunker."""

from __future__ import annotations

import pytest

from canny_chunker import REASON_SEMANTIC_DRIFT, REASON_THRESHOLD
from cross_encoder_chunker import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderVerifiedChunker
from tests.fakes import (
    DEFAULT_TEST_KWARGS,
    FakeCrossEncoder,
    FakeSentenceTransformer,
    lookup_vectors,
    unit,
)

#: 45 degrees between the middle sentence and its neighbours -> cosine distance
#: ``1 - cos(45 deg) ~= 0.293``, i.e. inside the gray zone (0.2, 0.4).
GRAY_TEXT = "A one. Mid one. A two."
GRAY_VECTORS = {
    "A one.": [1.0, 0.0],
    "Mid one.": unit([1.0, 1.0]),
    "A two.": [1.0, 0.0],
}
GRAY_KWARGS = {
    **DEFAULT_TEST_KWARGS,
    "manual_low_threshold": 0.2,
    "manual_high_threshold": 0.4,
}


def _gray_chunker(logit: float, **kwargs):
    """Chunker whose gray zone is resolved by a fake Cross-Encoder."""
    options = {"cross_encoder_threshold": 0.5, **GRAY_KWARGS, **kwargs}
    chunker = CrossEncoderVerifiedChunker(**options)
    chunker.model = FakeSentenceTransformer(lookup_vectors(GRAY_VECTORS))
    encoder = FakeCrossEncoder(logit)
    chunker.cross_encoder = encoder
    return chunker, encoder


# ------------------------------------------------------------------------ sigmoid
def test_sigmoid_is_numerically_stable():
    sigmoid = CrossEncoderVerifiedChunker._sigmoid
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(2.0) == pytest.approx(0.880797, abs=1e-6)
    assert sigmoid(-2.0) == pytest.approx(0.119203, abs=1e-6)
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)


def test_default_cross_encoder_model_is_a_valid_ms_marco_model():
    assert DEFAULT_CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert CrossEncoderVerifiedChunker().cross_encoder_model_name == (
        DEFAULT_CROSS_ENCODER_MODEL
    )


# -------------------------------------------------------------------- gray zone
def test_low_cross_encoder_probability_splits_in_the_gray_zone():
    chunker, encoder = _gray_chunker(logit=-5.0)
    assert chunker.chunk(GRAY_TEXT) == ["A one.", "Mid one.", "A two."]
    assert len(encoder.calls) == 2  # one verdict per gray-zone boundary


def test_high_cross_encoder_probability_merges_in_the_gray_zone():
    chunker, encoder = _gray_chunker(logit=5.0)
    assert chunker.chunk(GRAY_TEXT) == ["A one. Mid one. A two."]
    assert len(encoder.calls) == 2


def test_cross_encoder_split_is_reported_as_semantic_drift():
    chunker, _ = _gray_chunker(logit=-5.0)
    results = chunker.chunk_with_metadata(GRAY_TEXT)
    assert [result.split_reason for result in results] == [
        REASON_SEMANTIC_DRIFT,
        REASON_SEMANTIC_DRIFT,
        "end_of_text",
    ]

    diagnostics = chunker.get_diagnostics(GRAY_TEXT)
    first, second = diagnostics["boundaries"]
    assert first["decision"] == "gray->split"
    assert first["split_reason"] == REASON_SEMANTIC_DRIFT
    # The cross-encoder verdict replaces the centroid check, so no centroid
    # distance is recorded for these boundaries.
    assert first["centroid_distance"] is None
    assert second["centroid_distance"] is None


def test_cross_encoder_is_not_called_outside_the_gray_zone():
    chunker = CrossEncoderVerifiedChunker(**DEFAULT_TEST_KWARGS)
    chunker.model = FakeSentenceTransformer(
        lookup_vectors({"A one.": [1.0, 0.0], "B one.": [0.0, 1.0]})
    )
    encoder = FakeCrossEncoder(logit=-5.0)
    chunker.cross_encoder = encoder

    results = chunker.chunk_with_metadata("A one. B one.")
    assert [result.split_reason for result in results] == [
        REASON_THRESHOLD,
        "end_of_text",
    ]
    assert encoder.calls == []


def test_cross_encoder_context_uses_the_configured_trailing_sentences():
    chunker, encoder = _gray_chunker(logit=5.0, cross_encoder_context_sentences=1)
    chunker.chunk(GRAY_TEXT)
    assert [pairs[0] for pairs in encoder.calls] == [
        ("A one.", "Mid one."),
        ("Mid one.", "A two."),
    ]

    chunker, encoder = _gray_chunker(logit=5.0, cross_encoder_context_sentences=3)
    chunker.chunk(GRAY_TEXT)
    assert [pairs[0] for pairs in encoder.calls] == [
        ("A one.", "Mid one."),
        ("A one. Mid one.", "A two."),
    ]


def test_cross_encoder_threshold_controls_the_verdict():
    # sigmoid(0.0) == 0.5, so a threshold above 0.5 splits and one below merges.
    splitting, _ = _gray_chunker(logit=0.0, cross_encoder_threshold=0.51)
    assert len(splitting.chunk(GRAY_TEXT)) == 3

    merging, _ = _gray_chunker(logit=0.0, cross_encoder_threshold=0.49)
    assert len(merging.chunk(GRAY_TEXT)) == 1


def test_cross_encoder_model_is_lazy_and_injectable():
    chunker = CrossEncoderVerifiedChunker(**DEFAULT_TEST_KWARGS)
    assert chunker._cross_encoder is None
    fake = FakeCrossEncoder()
    chunker.cross_encoder = fake
    assert chunker.cross_encoder is fake
