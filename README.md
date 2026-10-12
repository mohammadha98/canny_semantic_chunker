# Canny Chunking

Semantic text chunking for RAG pipelines, inspired by the Canny edge detector.

Long documents are segmented into topically coherent chunks. The core algorithm is
**dual-threshold with centroid verification**: consecutive sentences are embedded
with a bi-encoder, the cosine distance between neighbours is compared against two
thresholds, and *ambiguous* (gray-zone) boundaries are resolved by measuring how
far the candidate sentence drifts from the **centroid of the chunk being built**.

## Algorithm

1. Segment the text into sentences (`pysbd`) and embed them (L2-normalized).
2. Compute the cosine distance `dist_i` between each pair of consecutive sentences.
3. Resolve the thresholds (`low_threshold` / `high_threshold`):
   * *adaptive* (default): percentiles of the document distance distribution,
     clamped by absolute floors, and only when the document is long enough
     (`min_sentences_for_adaptive`); otherwise the fixed defaults `0.6 / 0.8` are used;
   * *manual*: `manual_low_threshold` / `manual_high_threshold`.
4. For every boundary decide:

   | Condition | Decision |
   | --- | --- |
   | `dist < low_threshold` | merge (stay in the current chunk) |
   | `dist > high_threshold` | split (hard threshold) |
   | `low <= dist <= high` | **gray zone** -> centroid verification |

5. Gray-zone verification: the candidate sentence is compared with the centroid of
   the current chunk (or of the last `centroid_window_size` sentences). When
   `distance_to_centroid > drift_threshold` the boundary is a *gradual topic drift*
   and the chunk is split, otherwise the sentences are merged.
6. Size constraints always win: `max_sentences` / `max_tokens` force a split (the
   chunk is flagged `forced_by_max_size`), while `min_sentences` / `min_tokens`
   suppress semantic splits that would produce a too-small chunk (the number of
   suppressed splits is reported per chunk).

Two optional refinements:

* `centroid_window_size` (`int`, default `None`) restricts the centroid to a sliding
  window instead of the whole chunk — useful for ablation studies.
* `hysteresis_mode` confirms a gray-zone boundary as a split only when a strong
  boundary (`dist > high_threshold`) exists within `hysteresis_radius` sentences,
  or when the centroid drift check fails.

The chunker is fully deterministic (no randomness) and the embedding model is loaded
lazily, so a chunker can be constructed without downloading any weights.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from canny_chunker import CannySemanticChunker

chunker = CannySemanticChunker()          # model loaded lazily on first use
chunks = chunker.chunk(text)              # List[str]

for result in chunker.chunk_with_metadata(text):
    print(result.token_count, result.split_reason, result.text)

chunker.get_thresholds(text)              # (low_threshold, high_threshold)
chunker.get_drift_threshold(text)         # float
chunker.get_diagnostics(text)             # full decision trace (see below)
```

`ChunkResult` fields: `text`, `sentences`, `sentence_indices` (inclusive
`(start, end)` in the full sentence list), `token_count`, `centroid`,
`avg_internal_distance`, `split_reason`, `forced_by_max_size`, `dist_at_boundary`
(`None` for the last chunk) and `suppressed_splits`.

`get_diagnostics(text)` returns the `distances`, the resolved `low_threshold` /
`high_threshold` / `drift_threshold`, `n_sentences` and one entry per boundary:

```python
{
  "index": int,                # boundary after sentence index
  "distance": float,
  "low_threshold": float,
  "high_threshold": float,
  "drift_threshold": float,
  "centroid_distance": float | None,   # None outside the gray zone
  "decision": "merge" | "split" | "gray->merge" | "gray->split"
              | "suppressed" | "forced",
  "split_reason": str | None,
  "suppressed": bool,
  "forced_by_max_size": bool,
}
```

Split reasons are exposed as constants: `REASON_THRESHOLD`, `REASON_SEMANTIC_DRIFT`,
`REASON_MAX_SENTENCES`, `REASON_MAX_TOKENS`, `REASON_END_OF_TEXT`.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer id used for embeddings |
| `language` | `"en"` | Language passed to the `pysbd` sentence segmenter |
| `adaptive_threshold` | `True` | Derive the thresholds from the distance percentiles |
| `low_percentile` | `60` | Percentile of `low_threshold` in adaptive mode |
| `high_percentile` | `85` | Percentile of `high_threshold` in adaptive mode |
| `manual_low_threshold` | `None` | Fixed low threshold (with the high one) |
| `manual_high_threshold` | `None` | Fixed high threshold |
| `drift_threshold` | `None` | Max sentence-to-centroid distance in the gray zone; `None` derives it from `drift_percentile` |
| `drift_percentile` | `90` | Percentile used for the adaptive drift threshold |
| `centroid_window_size` | `None` | `None` = whole current chunk; `int` = last *n* sentences (ablation) |
| `hysteresis_mode` | `False` | Require a strong boundary nearby (or a drift failure) to split in the gray zone |
| `hysteresis_radius` | `2` | +/- window (in sentences) searched for a strong boundary |
| `min_sentences` | `2` | Minimum chunk size in sentences (suppresses semantic splits) |
| `max_sentences` | `15` | Maximum chunk size in sentences (forces a split) |
| `min_tokens` | `50` | Minimum chunk size in tokens (suppresses semantic splits) |
| `max_tokens` | `512` | Requested token budget; the effective limit is `min(max_tokens, model.max_seq_length)` |
| `low_floor` | `0.15` | Absolute floor for the adaptive low threshold |
| `high_floor` | `0.30` | Absolute floor for the adaptive high threshold |
| `drift_floor` | `None` | Absolute floor for the adaptive drift threshold (defaults to `high_floor`) |
| `min_sentences_for_adaptive` | `20` | Below this number of distances the fixed `0.6 / 0.8` thresholds are used |
| `exact_token_count` | `True` | Count tokens with the model tokenizer (otherwise a whitespace split) |

## Streamlit demo

```bash
streamlit run app.py
```

The sidebar exposes the threshold mode, the size limits, the centroid verification
settings (manual/adaptive drift threshold, centroid window, hysteresis) and the
opt-in hybrid mode. The results panel shows each chunk with its token count, split
reason, average internal distance, forced/suppressed warnings and the thresholds
that were actually used for the document.

## Evaluation

The `evaluation/` package provides Pk / WindowDiff metrics, baselines and an
ablation runner.

```bash
# Offline smoke test: toy corpus + dependency-free hashing embeddings
python -m evaluation.run_eval --demo

# Baselines included, several ablation presets, report written to disk
python -m evaluation.run_eval \
    --dataset data/wiki727k_sample.jsonl \
    --configs full,no_drift,window_3,hysteresis \
    --with-baselines \
    --model paraphrase-multilingual-MiniLM-L12-v2 \
    --output results/eval.json -v
```

* `evaluation/metrics.py` — `pk`, `windowdiff` (with an explicit `k`),
  `chunk_size_stats`, `mean_intra_chunk_distance`,
  `boundaries_from_segment_counts`.
* `evaluation/baselines.py` — `FixedSizeChunker`, `SingleThresholdChunker`,
  `PercentileThresholdChunker` (all expose `chunk(text) -> List[str]`).
* `evaluation/ablation.py` — named presets
  (`build_chunker("window_3", max_tokens=256)`) with the description of each
  configuration.
* `evaluation/run_eval.py` — CLI runner over a JSONL segmentation dataset.
* `evaluation/offline_model.py` — deterministic hashing bag-of-words embeddings used
  by `--demo` and by the tests (no model download).

### Ablation presets

| Preset | Effect |
| --- | --- |
| `full` | Dual threshold + whole-chunk centroid verification |
| `no_drift` | `drift_threshold = 2.0`, i.e. the gray zone always merges (no centroid verification) |
| `single_threshold` | `low_percentile == high_percentile`, i.e. a plain percentile splitter |
| `fixed_thresholds` | Adaptive thresholds disabled (`0.6 / 0.8`) |
| `window_3` / `window_5` | Centroid built from the last 3 / 5 sentences |
| `hysteresis` | Gray-zone splits require a strong boundary within +/- 2 sentences |

### Dataset format

One JSON object per line; `boundaries` may be replaced by `segment_counts`
(per-segment sentence counts) and `sentences` may be omitted (then `text` is
segmented with `pysbd`):

```json
{"id": "doc-1", "text": "Sentence one. Sentence two. ...",
 "sentences": ["Sentence one.", "Sentence two."],
 "boundaries": [0, 1]}          // length == len(sentences) - 1
```

The report contains, per configuration: `pk`, `windowdiff`, `n_docs`,
`chunk_size` (mean chunks / tokens / max tokens) and `mean_intra_chunk_distance`.
The Wiki-727K and Choi loaders are documented stubs: convert those corpora to JSONL
and use `--dataset`.

## Tests

```bash
pytest                 # 70 tests, no network access, no model download
```

* `tests/fakes.py` — `FakeSentenceTransformer`, `FakeCrossEncoder`, vector helpers
  and `DEFAULT_TEST_KWARGS` (manual thresholds, size limits disabled).
* `tests/test_chunker_core.py` — public API, thresholds, drift threshold,
  diagnostics, contiguity / losslessness.
* `tests/test_chunker_splits.py` — gray-zone merge/split, centroid window,
  hysteresis, forced and suppressed splits.
* `tests/test_metrics.py` — Pk / WindowDiff on hand-computed examples, segment-count
  helpers, chunk statistics, intra-chunk cohesion.
* `tests/test_evaluation.py` — baselines, ablation presets, JSONL loading,
  `evaluate()` and the `--demo` CLI run.
* `tests/test_cross_encoder_chunker.py` — sigmoid, gray-zone delegation, context
  pairs and lazy model loading.

## Hybrid mode (opt-in)

`cross_encoder_chunker.py` contains `CrossEncoderVerifiedChunker`, a subclass that
replaces the centroid check inside the gray zone with a Cross-Encoder relevance
probability (`sigmoid` of the raw logit, model
`cross-encoder/ms-marco-MiniLM-L-6-v2`). Cross-encoders score *query-passage
relevance*, not topic continuity, so this mode is deliberately opt-in and separate
from the main algorithm.
