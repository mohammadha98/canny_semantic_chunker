"""Named configuration presets for ablation studies.

Each preset is a dict of keyword arguments accepted by
:class:`canny_chunker.CannySemanticChunker`. Use :func:`build_chunker` to turn
a preset name into a ready-to-use chunker::

    from evaluation.ablation import build_chunker

    chunker = build_chunker("no_drift")        # disable centroid verification
    chunker = build_chunker("window_3")        # sliding-window centroid
    chunker = build_chunker("full", max_tokens=256)  # preset + override
"""

from __future__ import annotations

from typing import Any, Dict, List

from canny_chunker import CannySemanticChunker

__all__ = ["ABLATION_CONFIGS", "list_configs", "build_chunker"]

#: ``name -> {kwargs, description}``. The empty preset is the full algorithm.
ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "full": {
        "description": "Dual threshold + whole-chunk centroid verification.",
        "kwargs": {},
    },
    "no_drift": {
        "description": (
            "Dual threshold only: the drift threshold is pushed to 2.0 so the "
            "gray zone always merges (centroid verification disabled)."
        ),
        "kwargs": {"drift_threshold": 2.0},
    },
    "single_threshold": {
        "description": (
            "Degenerate gray zone (low_percentile == high_percentile) so the "
            "algorithm behaves like a plain percentile threshold splitter."
        ),
        "kwargs": {"low_percentile": 85, "high_percentile": 85},
    },
    "fixed_thresholds": {
        "description": "Adaptive thresholds disabled (default 0.6 / 0.8 used).",
        "kwargs": {"adaptive_threshold": False},
    },
    "window_3": {
        "description": "Centroid built from the last 3 sentences (sliding window).",
        "kwargs": {"centroid_window_size": 3},
    },
    "window_5": {
        "description": "Centroid built from the last 5 sentences (sliding window).",
        "kwargs": {"centroid_window_size": 5},
    },
    "hysteresis": {
        "description": "Gray-zone splits require a strong boundary within +/- 2.",
        "kwargs": {"hysteresis_mode": True, "hysteresis_radius": 2},
    },
}


def list_configs() -> List[str]:
    """Return the available preset names."""
    return sorted(ABLATION_CONFIGS)


def build_chunker(name: str = "full", **overrides: Any) -> CannySemanticChunker:
    """Instantiate a :class:`CannySemanticChunker` from a preset name.

    Args:
        name: A key of :data:`ABLATION_CONFIGS`.
        **overrides: Extra keyword arguments merged on top of the preset
            (they win over the preset values).

    Raises:
        KeyError: If ``name`` is not a known preset.
    """
    if name not in ABLATION_CONFIGS:
        raise KeyError(
            f"unknown ablation config {name!r}; available: {list_configs()}"
        )
    kwargs = dict(ABLATION_CONFIGS[name]["kwargs"])
    kwargs.update(overrides)
    return CannySemanticChunker(**kwargs)
