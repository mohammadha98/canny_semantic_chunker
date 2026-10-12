"""Core chunking behaviour: segmentation, thresholds, drift and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from canny_chunker import (
    ChunkResult,
    CannySemanticChunker,
    REASON_END_OF_TEXT,
)
from tests.fakes import (
    DEFAULT_TEST_KWARGS,
    FakeSentenceTransformer,
    keyword_vectors,
    lookup_vectors,
    make_chunker,
)


def _space_food_model() -> FakeSentenceTransformer:
    return FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"], "food": ["bread"]})
    )


# --------------------------------------------------------------- basic shapes
def test_empty_whitespace_and_none_inputs_produce_no_chunks():
    chunker = make_chunker(_space_food_model(), **DEFAULT_TEST_KWARGS)
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n  ") == []
    assert chunker.chunk_with_metadata(None) == []


def test_single_sentence_produces_one_end_of_text_chunk():
    chunker = make_chunker(_space_food_model(), **DEFAULT_TEST_KWARGS)
    results = chunker.chunk_with_metadata("A rocket is a rocket.")
    assert len(results) == 1
    assert isinstance(results[0], ChunkResult)
    assert results[0].split_reason == REASON_END_OF_TEXT
    assert results[0].sentence_indices == (0, 0)
    assert results[0].avg_internal_distance == 0.0
    assert np.isclose(np.linalg.norm(results[0].centroid), 1.0)


def test_chunk_returns_plain_strings():
    chunker = make_chunker(_space_food_model(), **DEFAULT_TEST_KWARGS)
    chunks = chunker.chunk("Rocket one. Rocket two. Bread one. Bread two.")
    assert chunks == ["Rocket one. Rocket two.", "Bread one. Bread two."]
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_sentence_indices_are_contiguous_and_lossless():
    """Guards against the historical first/last chunk duplication bug."""
    model = FakeSentenceTransformer(
        keyword_vectors(
            {"space": ["rocket"], "food": ["bread"], "sport": ["goal"]}
        )
    )
    chunker = make_chunker(model, **DEFAULT_TEST_KWARGS)
    text = (
        "Rocket one. Rocket two. Rocket three. "
        "Bread one. Bread two. "
        "Goal one. Goal two. Goal three."
    )
    results = chunker.chunk_with_metadata(text)
    sentences = chunker._segment(text)

    collected = [sentence for result in results for sentence in result.sentences]
    assert collected == sentences
    for result in results:
        start, end = result.sentence_indices
        assert result.sentences == sentences[start : end + 1]
        assert result.text == " ".join(sentences[start : end + 1])
    # chunk boundaries must be contiguous: end_i + 1 == start_{i+1}
    for previous, current in zip(results, results[1:]):
        assert previous.sentence_indices[1] + 1 == current.sentence_indices[0]
    assert results[-1].sentence_indices[1] == len(sentences) - 1


def test_token_count_matches_sum_of_sentence_tokens():
    chunker = make_chunker(_space_food_model(), **DEFAULT_TEST_KWARGS)
    text = "Rocket one. Rocket two. Bread one."
    results = chunker.chunk_with_metadata(text)
    for result in results:
        expected = sum(chunker._count_tokens(s) for s in result.sentences)
        assert result.token_count == expected


def test_model_is_lazy_and_can_be_injected():
    chunker = CannySemanticChunker()
    assert chunker._model is None
    fake = _space_food_model()
    chunker.model = fake
    assert chunker.model is fake


# ------------------------------------------------------------------ thresholds
def test_manual_thresholds_win_over_adaptive_mode():
    chunker = make_chunker(
        _space_food_model(),
        adaptive_threshold=True,
        manual_low_threshold=0.25,
        manual_high_threshold=0.75,
    )
    assert chunker.get_thresholds("Rocket one. Rocket two. Bread one.") == (0.25, 0.75)


def test_short_documents_fall_back_to_default_thresholds():
    """Fewer distances than ``min_sentences_for_adaptive`` -> fixed defaults."""
    chunker = make_chunker(
        _space_food_model(), adaptive_threshold=True, min_sentences_for_adaptive=20
    )
    text = " ".join(f"Rocket number {i}." for i in range(12))
    assert chunker.get_thresholds(text) == (
        CannySemanticChunker.DEFAULT_LOW_THRESHOLD,
        CannySemanticChunker.DEFAULT_HIGH_THRESHOLD,
    )


def test_adaptive_thresholds_use_document_percentiles():
    chunker = make_chunker(
        _space_food_model(),
        adaptive_threshold=True,
        low_percentile=60,
        high_percentile=85,
        min_sentences_for_adaptive=5,
    )
    # one topic => all distances identical => exercises the floor logic only
    text = " ".join(f"Rocket number {i}." for i in range(12))
    sentences = chunker._segment(text)
    distances = chunker._pairwise_distances(chunker._encode(sentences))
    expected_low = max(float(np.percentile(distances, 60)), chunker.low_floor)
    expected_high = max(float(np.percentile(distances, 85)), chunker.high_floor)
    low, high = chunker.get_thresholds(text)
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_high)
    assert low < high


def test_adaptive_thresholds_respect_floors():
    """Near-identical sentences give tiny percentiles that must be floored."""
    chunker = make_chunker(
        _space_food_model(),
        adaptive_threshold=True,
        low_floor=0.15,
        high_floor=0.30,
        min_sentences_for_adaptive=5,
    )
    text = " ".join(f"Rocket number {i}." for i in range(12))
    low, high = chunker.get_thresholds(text)
    assert low >= 0.15
    assert high >= 0.30
    assert low < high


def test_thresholds_are_ordered_when_manual_values_collide():
    chunker = make_chunker(
        _space_food_model(), manual_low_threshold=0.5, manual_high_threshold=0.5
    )
    low, high = chunker.get_thresholds("Rocket one. Bread one.")
    assert low < high


# ------------------------------------------------------------ drift threshold
def test_manual_drift_threshold_is_returned_as_is():
    chunker = make_chunker(
        _space_food_model(), drift_threshold=0.42, adaptive_threshold=False
    )
    assert chunker.get_drift_threshold("Rocket one. Bread one.") == pytest.approx(0.42)


def test_drift_threshold_falls_back_to_high_threshold_for_short_documents():
    chunker = make_chunker(
        _space_food_model(),
        adaptive_threshold=False,
        min_sentences_for_adaptive=20,
        drift_floor=0.30,
    )
    text = "Rocket one. Rocket two. Bread one."
    _, high = chunker.get_thresholds(text)
    assert chunker.get_drift_threshold(text) == pytest.approx(max(high, 0.30))


def test_drift_threshold_is_adaptive_for_long_documents():
    chunker = make_chunker(
        _space_food_model(),
        drift_percentile=90,
        drift_floor=0.30,
        min_sentences_for_adaptive=5,
    )
    text = " ".join(f"Sentence {i}." for i in range(12))
    embeddings = chunker._encode(chunker._segment(text))
    drift_distances = chunker._compute_drift_distances(embeddings)
    expected = max(float(np.percentile(drift_distances, 90)), 0.30)
    assert chunker.get_drift_threshold(text) == pytest.approx(expected)


# ------------------------------------------------------------------ diagnostics
def test_get_diagnostics_describes_every_boundary():
    chunker = make_chunker(_space_food_model(), **DEFAULT_TEST_KWARGS)
    text = "Rocket one. Rocket two. Bread one. Bread two."
    diagnostics = chunker.get_diagnostics(text)

    assert diagnostics["n_sentences"] == 4
    assert len(diagnostics["distances"]) == 3
    assert len(diagnostics["boundaries"]) == 3
    for key in ("low_threshold", "high_threshold", "drift_threshold"):
        assert isinstance(diagnostics[key], float)

    boundary = diagnostics["boundaries"][1]
    assert boundary["index"] == 1
    assert boundary["distance"] == pytest.approx(1.0)
    assert boundary["decision"] == "split"
    assert boundary["split_reason"] == "threshold"
    assert boundary["suppressed"] is False
    assert boundary["forced_by_max_size"] is False
    assert boundary["centroid_distance"] is None

    # a non-split boundary reports the plain merge decision
    assert diagnostics["boundaries"][0]["decision"] == "merge"
    assert diagnostics["boundaries"][0]["split_reason"] is None


def test_get_diagnostics_on_empty_text_is_safe():
    chunker = make_chunker(_space_food_model(), **DEFAULT_TEST_KWARGS)
    diagnostics = chunker.get_diagnostics("")
    assert diagnostics["n_sentences"] == 0
    assert diagnostics["distances"] == []
    assert diagnostics["boundaries"] == []
    assert diagnostics["low_threshold"] is None


def test_lookup_vectors_helper_drives_gray_zone_distances():
    """Sanity check of the test double itself (0.2929 == 1 - cos(45deg))."""
    model = FakeSentenceTransformer(
        lookup_vectors({"A.": [1.0, 0.0], "B.": [1.0, 1.0]})
    )
    embeddings = model.encode(["A.", "B."], normalize_embeddings=True)
    distance = 1.0 - float(np.dot(embeddings[0], embeddings[1]))
    assert distance == pytest.approx(1.0 - np.cos(np.pi / 4))


