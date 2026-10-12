"""Command line evaluation runner for the Canny chunker.

Two ways to run it::

    # Offline smoke test (no model download, built-in toy corpus):
    python -m evaluation.run_eval --demo

    # Real dataset in JSONL + real bi-encoder:
    python -m evaluation.run_eval \
        --dataset data/wiki727k_sample.jsonl \
        --configs full,no_drift,window_3 \
        --with-baselines \
        --output results/eval.json

Dataset format (one JSON object per line)::

    {"id": "doc-1",
     "text": "Sentence one. Sentence two. ...",
     "sentences": ["Sentence one.", "Sentence two.", "..."],
     "boundaries": [0, 1, 0, ...]}     # length == len(sentences) - 1

``segment_counts`` (per-segment sentence counts) may be used instead of
``boundaries``. ``sentences`` may be omitted, in which case ``text`` is
segmented with ``pysbd``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pysbd

from evaluation.ablation import ABLATION_CONFIGS, build_chunker
from evaluation.baselines import (
    FixedSizeChunker,
    PercentileThresholdChunker,
    SingleThresholdChunker,
)
from evaluation.metrics import (
    boundaries_from_segment_counts,
    chunk_size_stats,
    mean_intra_chunk_distance,
    pk,
    windowdiff,
)
from evaluation.offline_model import HashingEmbeddingModel

logger = logging.getLogger("evaluation.run_eval")

__all__ = [
    "load_jsonl",
    "record_to_reference",
    "predict_boundaries",
    "evaluate",
    "build_demo_dataset",
    "load_wiki_727k",
    "load_choi",
    "main",
]

_SEGMENTER_CACHE: Dict[str, pysbd.Segmenter] = {}


def _segmenter(language: str = "en") -> pysbd.Segmenter:
    """Return a cached ``pysbd`` segmenter for ``language``."""
    if language not in _SEGMENTER_CACHE:
        _SEGMENTER_CACHE[language] = pysbd.Segmenter(language=language, clean=False)
    return _SEGMENTER_CACHE[language]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL segmentation dataset (see the module docstring)."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
    return records


def record_to_reference(
    record: Dict[str, Any], language: str = "en"
) -> Tuple[str, List[str], List[int]]:
    """Return ``(text, sentences, boundaries)`` for a dataset record."""
    text = record.get("text") or ""
    sentences = record.get("sentences")
    if sentences is None:
        sentences = [
            s.strip() for s in _segmenter(language).segment(text) if s and s.strip()
        ]
    elif not text:
        text = " ".join(sentences)
    boundaries = record.get("boundaries")
    if boundaries is None:
        counts = record.get("segment_counts")
        if counts is None:
            raise ValueError(
                f"record {record.get('id')!r} needs 'boundaries' or 'segment_counts'"
            )
        boundaries = boundaries_from_segment_counts(counts)
    if len(boundaries) != max(len(sentences) - 1, 0):
        raise ValueError(
            f"record {record.get('id')!r}: expected {max(len(sentences) - 1, 0)} "
            f"boundaries, got {len(boundaries)}"
        )
    return text, list(sentences), [int(b) for b in boundaries]


def predict_boundaries(chunker: Any, text: str, language: str = "en") -> List[int]:
    """Derive the predicted boundary list from any chunker."""
    return _chunks_and_boundaries(chunker, text, language)[1]


def _chunks_and_boundaries(
    chunker: Any, text: str, language: str = "en"
) -> Tuple[List[str], List[int]]:
    """Return ``(chunks, boundaries)`` for any chunker.

    Chunkers exposing ``chunk_with_metadata`` (i.e. the Canny chunker) report
    their own sentence segmentation, which avoids re-segmenting the produced
    chunk texts. Baselines are re-segmented with ``pysbd``.
    """
    if hasattr(chunker, "chunk_with_metadata"):
        results = chunker.chunk_with_metadata(text)
        chunks = [result.text for result in results]
        counts = [len(result.sentences) for result in results]
    else:
        segmenter = _segmenter(language)
        chunks = chunker.chunk(text)
        counts = [
            len([s for s in segmenter.segment(chunk) if s and s.strip()])
            for chunk in chunks
        ]
    return chunks, boundaries_from_segment_counts(counts)


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of ``values`` (``0.0`` when empty)."""
    return float(np.mean(values)) if len(values) else 0.0


def _aggregate_chunk_sizes(stats: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average the per-document chunk-size statistics."""
    if not stats:
        return {
            "mean_chunks": 0.0,
            "mean_tokens": 0.0,
            "std_tokens": 0.0,
            "mean_max_tokens": 0.0,
        }
    return {
        "mean_chunks": _mean([s["n_chunks"] for s in stats]),
        "mean_tokens": _mean([s["mean_tokens"] for s in stats]),
        "std_tokens": _mean([s["std_tokens"] for s in stats]),
        "mean_max_tokens": _mean([s["max_tokens"] for s in stats]),
    }


def evaluate(
    records: Sequence[Dict[str, Any]],
    chunkers: Dict[str, Any],
    language: str = "en",
    model: Optional[object] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate every chunker in ``chunkers`` over ``records``.

    Args:
        records: Dataset records (see :func:`record_to_reference`).
        chunkers: ``name -> chunker`` mapping. When ``model`` is provided it is
            injected into each chunker that exposes a ``model`` setter.
        language: Sentence-segmentation language.
        model: Optional embedding model used both for chunking and for the
            ``mean_intra_chunk_distance`` metric.

    Returns:
        ``name -> {pk, windowdiff, n_docs, chunk_size, mean_intra_chunk_distance}``.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, chunker in chunkers.items():
        if model is not None:
            try:
                chunker.model = model
            except Exception:  # pragma: no cover - chunker without a model
                pass
        pks: List[float] = []
        wds: List[float] = []
        sizes: List[Dict[str, float]] = []
        intra: List[float] = []
        for record in records:
            text, _sentences, reference = record_to_reference(record, language)
            chunks, hypothesis = _chunks_and_boundaries(chunker, text, language)
            if len(hypothesis) != len(reference):
                logger.warning(
                    "record %r: boundary length mismatch (ref=%d, hyp=%d), skipping",
                    record.get("id"),
                    len(reference),
                    len(hypothesis),
                )
                continue
            pks.append(pk(reference, hypothesis))
            wds.append(windowdiff(reference, hypothesis))
            sizes.append(chunk_size_stats(chunks))
            if model is not None:
                intra.append(mean_intra_chunk_distance(chunks, model, language))
        results[name] = {
            "pk": _mean(pks),
            "windowdiff": _mean(wds),
            "n_docs": len(pks),
            "chunk_size": _aggregate_chunk_sizes(sizes),
            "mean_intra_chunk_distance": _mean(intra) if intra else None,
        }
    return results


# --------------------------------------------------------------------- datasets
def build_demo_dataset() -> List[Dict[str, Any]]:
    """Return a tiny offline dataset with hand-annotated topic boundaries.

    Each document contains two topics of three sentences each. Sentences are
    deliberately long enough (>~25 tokens) that the default ``min_tokens=50``
    constraint does not swallow every candidate boundary.
    """
    docs = [
        {
            "id": "demo-astronomy",
            "sentences": [
                "The James Webb Space Telescope observes the universe primarily in "
                "the infrared part of the spectrum, allowing astronomers to peer "
                "through clouds of cosmic dust that block visible light.",
                "Its primary mirror consists of eighteen hexagonal beryllium "
                "segments coated with a thin layer of gold, which together collect "
                "and focus an extraordinary amount of infrared light onto the "
                "science instruments.",
                "Researchers use the observatory to characterise the atmospheres of "
                "distant exoplanets by watching how starlight filters through the "
                "gases surrounding those alien worlds.",
                "Baking a reliable loaf of sourdough bread begins with a healthy "
                "starter, a live culture of wild yeast and lactic acid bacteria that "
                "must be refreshed regularly with flour and water.",
                "The dough is folded gently at intervals during bulk fermentation so "
                "that gluten develops strength without losing the gas bubbles that "
                "will later open up the crumb.",
                "A very hot oven combined with a covered pot traps steam around the "
                "loaf, producing a deeply caramelised crust and an airy, irregular "
                "interior structure.",
            ],
            "segment_counts": [3, 3],
        },
        {
            "id": "demo-finance",
            "sentences": [
                "Central banks raise their policy interest rates when inflation runs "
                "persistently above target, deliberately cooling demand across the "
                "economy by making credit more expensive for households and firms.",
                "Higher borrowing costs discourage consumers from taking out new "
                "mortgages and reduce the willingness of companies to finance large "
                "investment projects with debt.",
                "Prices of existing fixed-rate bonds therefore fall whenever yields "
                "on newly issued government debt rise, because investors demand a "
                "discount to hold the older, lower-paying securities.",
                "The goalkeeper read the striker's body shape and dived low to the "
                "left, pushing the penalty kick around the post and keeping the score "
                "level late in the match.",
                "Coaching staff spend hours designing set pieces that exploit "
                "weaknesses in the opponent's defensive shape, rehearsing runs and "
                "blocks until the movement becomes automatic.",
                "A dramatic goal deep into stoppage time decided the championship "
                "final, sending the supporters into a celebration that lasted long "
                "after the final whistle.",
            ],
            "segment_counts": [3, 3],
        },
    ]
    return docs


def load_wiki_727k(path: str) -> List[Dict[str, Any]]:  # pragma: no cover - stub
    """Stub loader for the Wiki-727K segmentation benchmark.

    Wiki-727K ships one document per line with sentences separated by tabs and
    segment identifiers as a second column. Download the corpus from
    https://github.com/koomri/text-segmentation and convert it to the JSONL
    format described in this module's docstring (``load_jsonl``) before
    evaluating.
    """
    raise NotImplementedError(
        "Wiki-727K loading is not implemented; convert the corpus to JSONL and "
        "use load_jsonl(path) instead."
    )


def load_choi(path: str) -> List[Dict[str, Any]]:  # pragma: no cover - stub
    """Stub loader for the Choi (2000) segmentation benchmark.

    The Choi dataset is distributed as one document per file with blank-line
    separated segments. Convert it to the JSONL format described in this
    module's docstring (``load_jsonl``) before evaluating.
    """
    raise NotImplementedError(
        "Choi loading is not implemented; convert the corpus to JSONL and use "
        "load_jsonl(path) instead."
    )


# -------------------------------------------------------------------------- CLI
def build_chunkers(
    config_names: Sequence[str], with_baselines: bool = False
) -> Dict[str, Any]:
    """Instantiate the ablation presets (and optional baselines)."""
    chunkers: Dict[str, Any] = {name: build_chunker(name) for name in config_names}
    if with_baselines:
        chunkers["baseline_fixed_512"] = FixedSizeChunker(max_tokens=512)
        chunkers["baseline_single_0.5"] = SingleThresholdChunker(threshold=0.5)
        chunkers["baseline_percentile_85"] = PercentileThresholdChunker(percentile=85)
    return chunkers


def _resolve_model(model_name: Optional[str]) -> object:
    """Return the real bi-encoder when a name is given, else the offline model."""
    if model_name:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    logger.info("no --model given: using the offline hashing embedding model")
    return HashingEmbeddingModel()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python -m evaluation.run_eval``."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description="Evaluate the Canny semantic chunker and its baselines.",
    )
    parser.add_argument("--dataset", help="JSONL segmentation dataset.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the built-in toy corpus (implies the offline embedding model).",
    )
    parser.add_argument(
        "--configs",
        default="full",
        help=f"Comma separated ablation presets. Available: {sorted(ABLATION_CONFIGS)}",
    )
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="Also evaluate the fixed-size / single-threshold baselines.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="SentenceTransformer id (default: offline hashing model).",
    )
    parser.add_argument("--language", default="en", help="Segmentation language.")
    parser.add_argument("--output", default=None, help="Write the JSON report here.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.demo or not args.dataset:
        records = build_demo_dataset()
    else:
        records = load_jsonl(args.dataset)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in ABLATION_CONFIGS]
    if unknown:
        parser.error(f"unknown config(s) {unknown}; available: {sorted(ABLATION_CONFIGS)}")

    chunkers = build_chunkers(configs, with_baselines=args.with_baselines)
    model = _resolve_model(args.model)
    results = evaluate(records, chunkers, language=args.language, model=model)

    payload = {
        "n_records": len(records),
        "configs": configs,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"written to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
