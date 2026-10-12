"""Tests for the boundary metrics and chunk statistics."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    boundaries_from_segment_counts,
    chunk_size_stats,
    mean_intra_chunk_distance,
    pk,
    windowdiff,
)
from tests.fakes import FakeSentenceTransformer, lookup_vectors


def _counts_from_boundaries(boundaries):
    """Inverse of :func:`boundaries_from_segment_counts`."""
    counts = []
    current = 1
    for boundary in boundaries:
        if boundary:
            counts.append(current)
            current = 1
        else:
            current += 1
    counts.append(current)
    return counts


# ----------------------------------------------------------------------- Pk/WD
def test_metrics_are_zero_for_identical_segmentations():
    reference = [0, 1, 0, 1, 0]
    assert pk(reference, list(reference)) == 0.0
    assert windowdiff(reference, list(reference)) == 0.0


def test_metrics_are_one_for_opposite_segmentations():
    assert pk([1, 1, 1], [0, 0, 0]) == pytest.approx(1.0)
    assert windowdiff([1, 1, 1], [0, 0, 0]) == pytest.approx(1.0)


def test_metrics_on_known_small_example():
    """ref=[0,1,0] / hyp=[0,0,1] with k=2 -> one agreeing window out of two."""
    reference = [0, 1, 0]
    hypothesis = [0, 0, 1]
    assert pk(reference, hypothesis) == pytest.approx(0.5)
    assert windowdiff(reference, hypothesis) == pytest.approx(0.5)
    assert pk(reference, hypothesis, k=2) == pytest.approx(0.5)
    assert windowdiff(reference, hypothesis, k=2) == pytest.approx(0.5)


def test_windowdiff_is_at_least_as_strict_as_pk():
    reference = [1, 0, 1, 0]
    hypothesis = [1, 1, 0, 0]
    assert windowdiff(reference, hypothesis) >= pk(reference, hypothesis)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        pk([0, 1], [0, 1, 0])
    with pytest.raises(ValueError):
        windowdiff([0, 1], [0])


def test_metrics_reject_non_binary_values():
    with pytest.raises(ValueError):
        pk([0, 2], [0, 1])
    with pytest.raises(ValueError):
        windowdiff([[0, 1]], [[0, 1]])


# ------------------------------------------------------- segment count helpers
def test_boundaries_from_segment_counts_round_trip():
    for counts in ([3, 2], [5], [1, 1, 1], [2, 3, 4, 1]):
        boundaries = boundaries_from_segment_counts(counts)
        assert len(boundaries) == sum(counts) - 1
        assert set(boundaries) <= {0, 1}
        assert _counts_from_boundaries(boundaries) == list(counts)


def test_boundaries_from_segment_counts_places_single_boundary_correctly():
    assert boundaries_from_segment_counts([3, 2]) == [0, 0, 1, 0]
    assert boundaries_from_segment_counts([5]) == [0, 0, 0, 0]
    assert boundaries_from_segment_counts([]) == []


# ------------------------------------------------------------- chunk size stats
def test_chunk_size_stats_uses_the_custom_token_counter():
    chunks = ["a b c", "d e"]
    stats = chunk_size_stats(chunks)
    assert stats["n_chunks"] == 2
    assert stats["mean_tokens"] == pytest.approx(2.5)
    assert stats["min_tokens"] == 2.0
    assert stats["max_tokens"] == 3.0
    assert stats["std_tokens"] == pytest.approx(0.5)

    character_counter = chunk_size_stats(chunks, token_counter=len)
    assert character_counter["mean_tokens"] == pytest.approx((5 + 3) / 2)


def test_chunk_size_stats_on_empty_input():
    stats = chunk_size_stats([])
    assert stats["n_chunks"] == 0
    assert stats["mean_tokens"] == 0.0


# ------------------------------------------------------- intra-chunk cohesion
def test_mean_intra_chunk_distance_is_zero_for_same_topic_chunks():
    model = FakeSentenceTransformer(
        lookup_vectors({"A one.": [1.0, 0.0], "A two.": [1.0, 0.0]})
    )
    assert mean_intra_chunk_distance(["A one. A two."], model) == pytest.approx(0.0)


def test_mean_intra_chunk_distance_matches_manual_centroid_distance():
    model = FakeSentenceTransformer(
        lookup_vectors({"A one.": [1.0, 0.0], "B one.": [0.0, 1.0]})
    )
    distance = mean_intra_chunk_distance(["A one. B one."], model)
    assert distance == pytest.approx(1.0 - np.cos(np.pi / 4))


def test_mean_intra_chunk_distance_ignores_single_sentence_chunks():
    model = FakeSentenceTransformer(lookup_vectors({}))
    assert mean_intra_chunk_distance(["Only one sentence."], model) == 0.0
    assert mean_intra_chunk_distance([], model) == 0.0
