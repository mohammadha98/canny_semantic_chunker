"""Evaluation utilities for the Canny semantic chunker.

This package groups the research / evaluation helpers:

* :mod:`evaluation.metrics`   - Pk & WindowDiff + chunk-size statistics.
* :mod:`evaluation.baselines` - fixed-size / single-threshold / percentile
  chunkers sharing the ``chunk(text) -> List[str]`` interface.
* :mod:`evaluation.ablation`  - named configuration presets.
* :mod:`evaluation.run_eval`  - CLI runner over a JSONL segmentation dataset.
"""

from evaluation.metrics import (
    chunk_size_stats,
    mean_intra_chunk_distance,
    pk,
    windowdiff,
)

__all__ = [
    "pk",
    "windowdiff",
    "chunk_size_stats",
    "mean_intra_chunk_distance",
]
