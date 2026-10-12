"""Boundary decisions: gray zone, centroid verification, hysteresis, forces."""

from __future__ import annotations

import numpy as np
import pytest

from canny_chunker import (
    REASON_END_OF_TEXT,
    REASON_MAX_SENTENCES,
    REASON_MAX_TOKENS,
    REASON_SEMANTIC_DRIFT,
    REASON_THRESHOLD,
)
from tests.fakes import (
    FakeSentenceTransformer,
    keyword_vectors,
    lookup_vectors,
    make_chunker,
    unit,
)

GRAY_KWARGS = {
    "adaptive_threshold": False,
    "manual_low_threshold": 0.2,
    "manual_high_threshold": 0.8,
    "min_sentences": 1,
    "min_tokens": 0,
    "max_sentences": 1000,
    "max_tokens": 512,
    "exact_token_count": False,
}

GRAY_TEXT = "A one. A two. B one."
GRAY_VECTORS = {
    "A one.": [1.0, 0.0],
    "A two.": [1.0, 0.0],
    "B one.": [1.0, 1.0],
}


def _gray_model() -> FakeSentenceTransformer:
    return FakeSentenceTransformer(lookup_vectors(GRAY_VECTORS))


# ------------------------------------------------------------------ gray zone
def test_gray_zone_merges_when_centroid_is_close_enough():
    chunker = make_chunker(_gray_model(), drift_threshold=0.5, **GRAY_KWARGS)
    results = chunker.chunk_with_metadata(GRAY_TEXT)
    assert len(results) == 1
    assert results[0].split_reason == REASON_END_OF_TEXT

    boundary = chunker.get_diagnostics(GRAY_TEXT)["boundaries"][1]
    assert boundary["decision"] == "gray->merge"
    assert boundary["centroid_distance"] == pytest.approx(1.0 - np.cos(np.pi / 4))
    assert boundary["split_reason"] is None


def test_gray_zone_splits_when_centroid_drift_exceeds_threshold():
    chunker = make_chunker(_gray_model(), drift_threshold=0.1, **GRAY_KWARGS)
    results = chunker.chunk_with_metadata(GRAY_TEXT)
    assert [r.text for r in results] == ["A one. A two.", "B one."]
    assert results[0].split_reason == REASON_SEMANTIC_DRIFT
    assert results[0].dist_at_boundary == pytest.approx(1.0 - np.cos(np.pi / 4))

    boundary = chunker.get_diagnostics(GRAY_TEXT)["boundaries"][1]
    assert boundary["decision"] == "gray->split"
    assert boundary["split_reason"] == REASON_SEMANTIC_DRIFT


def test_hard_thresholds_never_consult_the_centroid():
    """Distances outside the gray zone skip the (expensive) centroid check."""
    model = FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"], "food": ["bread"]})
    )
    chunker = make_chunker(model, drift_threshold=0.5, **GRAY_KWARGS)
    text = "Rocket one. Bread one."
    assert [r.text for r in chunker.chunk_with_metadata(text)] == [
        "Rocket one.",
        "Bread one.",
    ]

    boundary = chunker.get_diagnostics(text)["boundaries"][0]
    assert boundary["distance"] == pytest.approx(1.0)
    assert boundary["decision"] == "split"
    assert boundary["centroid_distance"] is None


def test_centroid_window_changes_gray_zone_decision():
    """Whole-chunk centroid vs. sliding window can flip the same boundary."""
    s0 = unit([1.0, -0.2])
    s1 = np.array([1.0, 0.0])
    s2 = np.array(unit([1.0, 1.0]))
    vectors = {"S zero.": s0, "S one.": s1.tolist(), "S two.": s2.tolist()}
    text = "S zero. S one. S two."

    # verify the geometry the test relies on
    distance_01 = 1.0 - float(np.dot(s0, s1))
    distance_12 = 1.0 - float(np.dot(s1, s2))
    centroid = np.array(s0) + s1
    centroid = centroid / np.linalg.norm(centroid)
    distance_to_whole_centroid = 1.0 - float(np.dot(s2, centroid))
    assert distance_01 < 0.2
    assert 0.2 < distance_12 < 0.8
    assert distance_12 < 0.33 < distance_to_whole_centroid

    model = FakeSentenceTransformer(lookup_vectors(vectors))
    whole = make_chunker(model, drift_threshold=0.33, **GRAY_KWARGS)
    windowed = make_chunker(
        model, centroid_window_size=1, drift_threshold=0.33, **GRAY_KWARGS
    )

    assert len(whole.chunk_with_metadata(text)) == 2
    assert len(windowed.chunk_with_metadata(text)) == 1


# ------------------------------------------------------------------ hysteresis
HYSTERESIS_VECTORS = {
    "H zero.": [1.0, 0.0, 0.0],
    "H one.": [1.0, 0.0, 0.0],
    "H two.": unit([1.0, 1.0, 0.0]),
    "H three.": [0.0, 1.0, 0.0],
    "H four.": [0.0, 1.0, 0.0],
    "H five.": [0.0, 0.0, 1.0],
}
HYSTERESIS_TEXT = "H zero. H one. H two. H three. H four. H five."


def test_hysteresis_confirms_gray_zone_split_near_a_strong_boundary():
    model = FakeSentenceTransformer(lookup_vectors(HYSTERESIS_VECTORS))
    # drift_threshold=2.0 disables the centroid check, isolating the hysteresis
    # effect: only the strong boundary before "H five." would normally split.
    common = dict(GRAY_KWARGS, drift_threshold=2.0, manual_low_threshold=0.2,
                  manual_high_threshold=0.8)
    without = make_chunker(model, hysteresis_mode=False, **common)
    with_hysteresis = make_chunker(
        model, hysteresis_mode=True, hysteresis_radius=2, **common
    )

    assert [len(r.sentences) for r in without.chunk_with_metadata(HYSTERESIS_TEXT)] == [5, 1]

    results = with_hysteresis.chunk_with_metadata(HYSTERESIS_TEXT)
    assert [len(r.sentences) for r in results] == [3, 2, 1]
    assert results[0].split_reason == REASON_THRESHOLD
    assert results[2].split_reason == REASON_END_OF_TEXT

    boundary = with_hysteresis.get_diagnostics(HYSTERESIS_TEXT)["boundaries"][2]
    # The boundary is inside the gray zone (distance < high_threshold) but the
    # nearby strong boundary promotes it to a confirmed threshold split.
    assert boundary["decision"] == "gray->split"
    assert boundary["split_reason"] == REASON_THRESHOLD
    assert boundary["distance"] == pytest.approx(1.0 - np.cos(np.pi / 4))


def test_hysteresis_radius_zero_behaves_like_plain_drift_mode():
    model = FakeSentenceTransformer(lookup_vectors(HYSTERESIS_VECTORS))
    chunker = make_chunker(
        model,
        hysteresis_mode=True,
        hysteresis_radius=0,
        drift_threshold=2.0,
        **dict(GRAY_KWARGS, manual_low_threshold=0.2, manual_high_threshold=0.8),
    )
    assert [len(r.sentences) for r in chunker.chunk_with_metadata(HYSTERESIS_TEXT)] == [5, 1]


# --------------------------------------------------------------- forced splits
def test_max_sentences_forces_chunks_and_flags_them():
    model = FakeSentenceTransformer(keyword_vectors({"space": ["rocket"]}))
    chunker = make_chunker(model, **dict(GRAY_KWARGS, max_sentences=2))
    text = " ".join(f"Rocket number {i}." for i in range(5))
    results = chunker.chunk_with_metadata(text)

    assert [len(r.sentences) for r in results] == [2, 2, 1]
    assert results[0].split_reason == REASON_MAX_SENTENCES
    assert results[1].split_reason == REASON_MAX_SENTENCES
    assert results[2].split_reason == REASON_END_OF_TEXT
    assert [r.forced_by_max_size for r in results] == [True, True, False]

    boundary = chunker.get_diagnostics(text)["boundaries"][1]
    assert boundary["decision"] == "forced"
    assert boundary["forced_by_max_size"] is True
    assert boundary["split_reason"] == REASON_MAX_SENTENCES


def test_max_tokens_forces_chunks_and_is_capped_by_model_limit():
    model = FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"]}), max_seq_length=512
    )
    chunker = make_chunker(model, **dict(GRAY_KWARGS, max_tokens=5))
    text = " ".join(  # 4 tokens per sentence, same topic everywhere
        f"Rocket alpha {word} now." for word in ["one", "two", "three", "four", "five"]
    )
    results = chunker.chunk_with_metadata(text)

    assert len(results) == 5
    assert results[0].split_reason == REASON_MAX_TOKENS
    assert results[0].forced_by_max_size is True
    assert results[0].token_count == 4
    assert results[-1].split_reason == REASON_END_OF_TEXT


# ------------------------------------------------------------ suppressed splits
def test_min_sentences_suppresses_split_and_records_it():
    model = FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"], "food": ["bread"]})
    )
    chunker = make_chunker(model, **dict(GRAY_KWARGS, min_sentences=3))
    text = "Rocket one. Bread one. Bread two. Bread three."
    results = chunker.chunk_with_metadata(text)

    assert len(results) == 1
    assert results[0].suppressed_splits == 1
    assert results[0].dist_at_boundary is None

    boundary = chunker.get_diagnostics(text)["boundaries"][0]
    assert boundary["decision"] == "suppressed"
    assert boundary["suppressed"] is True
    assert boundary["split_reason"] is None
    assert boundary["distance"] == pytest.approx(1.0)


def test_min_tokens_suppresses_split():
    model = FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"], "food": ["bread"]})
    )
    chunker = make_chunker(model, **dict(GRAY_KWARGS, min_sentences=1, min_tokens=50))
    results = chunker.chunk_with_metadata("Rocket one. Bread one. Bread two.")
    assert len(results) == 1
    assert results[0].suppressed_splits == 1


def test_suppressed_splits_are_reset_after_a_real_split():
    model = FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"], "food": ["bread"], "sport": ["goal"]})
    )
    chunker = make_chunker(model, **dict(GRAY_KWARGS, min_sentences=2, max_sentences=4))
    text = (
        "Rocket one. Bread one. Bread two. Bread three. "
        "Goal one. Goal two. Goal three. Goal four."
    )
    results = chunker.chunk_with_metadata(text)
    assert results[0].suppressed_splits == 1  # the rocket->bread shift was blocked
    assert results[-1].suppressed_splits == 0
    assert all(r.split_reason is not None for r in results)

