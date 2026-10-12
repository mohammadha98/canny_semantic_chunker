"""Tests for the baselines, ablation presets and the evaluation runner."""

from __future__ import annotations

import json

import pytest

from canny_chunker import CannySemanticChunker
from evaluation.ablation import ABLATION_CONFIGS, build_chunker, list_configs
from evaluation.baselines import (
    BaselineChunker,
    FixedSizeChunker,
    PercentileThresholdChunker,
    SingleThresholdChunker,
)
from evaluation.offline_model import HashingEmbeddingModel
from evaluation.run_eval import (
    build_chunkers,
    build_demo_dataset,
    evaluate,
    load_choi,
    load_jsonl,
    load_wiki_727k,
    main,
    predict_boundaries,
    record_to_reference,
)
from tests.fakes import FakeSentenceTransformer, keyword_vectors

TWO_TOPIC_TEXT = "Rocket one. Rocket two. Bread one. Bread two."


def _topic_model() -> FakeSentenceTransformer:
    return FakeSentenceTransformer(
        keyword_vectors({"space": ["rocket"], "food": ["bread"]})
    )


# --------------------------------------------------------------------- baselines
def test_baseline_interface_raises_when_not_implemented():
    with pytest.raises(NotImplementedError):
        BaselineChunker().chunk("Some text.")


def test_fixed_size_chunker_packs_sentences_to_the_token_budget():
    chunker = FixedSizeChunker(max_tokens=4)
    chunks = chunker.chunk("One two three. Four five six. Seven eight nine.")
    assert chunks == ["One two three.", "Four five six.", "Seven eight nine."]


def test_fixed_size_chunker_supports_overlap():
    chunker = FixedSizeChunker(max_tokens=4, overlap_sentences=1)
    chunks = chunker.chunk("One two three. Four five six. Seven eight nine.")
    assert len(chunks) == 3
    assert chunks[1] == "One two three. Four five six."
    assert chunks[2] == "Four five six. Seven eight nine."


def test_fixed_size_chunker_on_empty_text():
    assert FixedSizeChunker().chunk("") == []
    assert FixedSizeChunker().chunk(None) == []


def test_single_threshold_chunker_splits_on_topic_shift():
    chunker = SingleThresholdChunker(threshold=0.5, model=_topic_model())
    assert chunker.chunk(TWO_TOPIC_TEXT) == ["Rocket one. Rocket two.", "Bread one. Bread two."]


def test_single_threshold_chunker_keeps_cohesive_text_together():
    chunker = SingleThresholdChunker(threshold=0.5, model=_topic_model())
    assert chunker.chunk("Rocket one. Rocket two. Rocket three.") == [
        "Rocket one. Rocket two. Rocket three."
    ]


def test_percentile_threshold_chunker_splits_above_the_percentile():
    chunker = PercentileThresholdChunker(percentile=85, model=_topic_model())
    assert chunker.chunk(TWO_TOPIC_TEXT) == ["Rocket one. Rocket two.", "Bread one. Bread two."]
    assert chunker.chunk("Rocket one.") == ["Rocket one."]


def test_baseline_models_are_lazy_and_injectable():
    chunker = SingleThresholdChunker()
    assert chunker._model is None
    fake = _topic_model()
    chunker.model = fake
    assert chunker.model is fake


# ---------------------------------------------------------------------- ablation
def test_ablation_configs_are_all_constructible():
    names = list_configs()
    assert "full" in names
    for name in names:
        assert isinstance(ABLATION_CONFIGS[name]["description"], str)
        assert isinstance(build_chunker(name), CannySemanticChunker)


def test_build_chunker_rejects_unknown_preset():
    with pytest.raises(KeyError):
        build_chunker("does-not-exist")


def test_build_chunker_overrides_win_over_the_preset():
    chunker = build_chunker("window_3", max_tokens=256)
    assert chunker.centroid_window_size == 3
    assert chunker.max_tokens == 256
    assert build_chunker("no_drift").drift_threshold == 2.0
    assert build_chunker("hysteresis").hysteresis_mode is True


def test_build_chunkers_adds_optional_baselines():
    chunkers = build_chunkers(["full", "no_drift"], with_baselines=True)
    assert set(chunkers) == {
        "full",
        "no_drift",
        "baseline_fixed_512",
        "baseline_single_0.5",
        "baseline_percentile_85",
    }
    assert set(build_chunkers(["full"])) == {"full"}


# ---------------------------------------------------------------------- run_eval
def test_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"id": "a", "text": "One. Two.", "boundaries": [1]}\n\n'
        '{"id": "b", "segment_counts": [2]}\n',
        encoding="utf-8",
    )
    records = load_jsonl(str(path))
    assert [record["id"] for record in records] == ["a", "b"]


def test_load_jsonl_reports_invalid_json(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(str(path))


def test_record_to_reference_accepts_all_supported_shapes():
    text, sentences, boundaries = record_to_reference(
        {"id": "a", "text": "One. Two. Three.", "boundaries": [0, 1]}
    )
    assert sentences == ["One.", "Two.", "Three."]
    assert boundaries == [0, 1]
    assert text == "One. Two. Three."

    text, sentences, boundaries = record_to_reference(
        {"id": "b", "sentences": ["One.", "Two.", "Three."], "segment_counts": [2, 1]}
    )
    assert text == "One. Two. Three."
    assert boundaries == [0, 1]

    with pytest.raises(ValueError):
        record_to_reference({"id": "c", "sentences": ["One.", "Two."]})
    with pytest.raises(ValueError):
        record_to_reference(
            {"id": "d", "sentences": ["One.", "Two."], "boundaries": [0, 0]}
        )


def test_predict_boundaries_uses_the_chunker_sentence_counts():
    chunker = SingleThresholdChunker(threshold=0.5, model=_topic_model())
    boundaries = predict_boundaries(chunker, TWO_TOPIC_TEXT)
    assert boundaries == [0, 1, 0]


def test_evaluate_returns_one_entry_per_chunker():
    records = build_demo_dataset()
    chunkers = {"full": build_chunker("full"), "fixed": FixedSizeChunker(max_tokens=64)}
    results = evaluate(records, chunkers, model=HashingEmbeddingModel())
    assert set(results) == {"full", "fixed"}
    for metrics in results.values():
        assert metrics["n_docs"] == len(records)
        assert 0.0 <= metrics["pk"] <= 1.0
        assert 0.0 <= metrics["windowdiff"] <= 1.0
        assert metrics["chunk_size"]["mean_chunks"] >= 1
        assert metrics["mean_intra_chunk_distance"] is not None


def test_main_demo_run_writes_a_report(tmp_path, capsys):
    output = tmp_path / "nested" / "eval.json"
    exit_code = main(
        [
            "--demo",
            "--configs",
            "full,no_drift",
            "--with-baselines",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_records"] == len(build_demo_dataset())
    assert payload["configs"] == ["full", "no_drift"]
    assert set(payload["results"]) == {
        "full",
        "no_drift",
        "baseline_fixed_512",
        "baseline_single_0.5",
        "baseline_percentile_85",
    }
    assert "results" in capsys.readouterr().out


def test_main_rejects_unknown_configs():
    with pytest.raises(SystemExit):
        main(["--demo", "--configs", "full,not-a-config"])


def test_external_dataset_loaders_are_documented_stubs():
    with pytest.raises(NotImplementedError):
        load_wiki_727k("some/path")
    with pytest.raises(NotImplementedError):
        load_choi("some/path")

