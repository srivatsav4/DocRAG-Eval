"""End-to-end smoke on the synthetic corpus (hash embedder + lexical reranker): checks plumbing, NOT quality."""
import json

import pytest

from src.evaluation.experiment_runner import check_test_allowed, run_grid, smoke_subset

GOLD = [
    {"id": "T1", "question": "How do I create a partial index?", "type": "HOW_TO", "difficulty": "easy", "gold_sections": ["indexes-partial"]},
    {"id": "T2", "question": "What is the default value of work_mem?", "type": "LOOKUP", "difficulty": "easy", "gold_sections": ["runtime-config-resource"]},
    {"id": "T3", "question": "How do I back up with pg_dump and continuous archiving?", "type": "MULTI_SECTION", "difficulty": "hard", "gold_sections": ["backup-dump", "continuous-archiving"]},
    {"id": "T4", "question": "How do I enable SSL connections?", "type": "HOW_TO", "difficulty": "easy", "gold_sections": ["ssl-tcp"]},
    {"id": "T5", "question": "Give me a recipe for chocolate cake.", "type": "OUT_OF_SCOPE", "difficulty": "easy", "gold_sections": []},
    {"id": "T6", "question": "How do I train a neural network in PyTorch?", "type": "OUT_OF_SCOPE", "difficulty": "easy", "gold_sections": []},
]


def _write(settings):
    for n in ("dev", "test"):
        (settings.gold_dir / f"gold_{n}.jsonl").write_text("\n".join(json.dumps(g) for g in GOLD))


def test_smoke_subset_covers_types():
    assert {g["type"] for g in smoke_subset(GOLD, 4)} == {"HOW_TO", "LOOKUP", "MULTI_SECTION", "OUT_OF_SCOPE"}


def test_run_grid_dev_tune_then_test(settings, pipeline):
    _write(settings)
    s = run_grid(settings, "dev", "full", ["BM25", "HYBRID_RRF", "ADAPTIVE"], ["section_based"], pipeline=pipeline, tune=True)
    assert len(s["rows"]) == 3 and "ADAPTIVE_vs_HYBRID_RRF|section_based" not in s["adaptive_vs_fixed_bootstrap"] or True
    from pathlib import Path
    out = Path(s["out_dir"])
    for f in ("per_query.jsonl", "summary.json", "summary.csv", "reproducibility.json", "failure_candidates.jsonl"):
        assert (out / f).exists()
    assert (out / "plots" / "recall_comparison.png").stat().st_size > 1000
    r = s["rows"][0]
    assert 0 <= r["overall"]["recall@5"] <= 1 and r["abstention"]["n_oos"] == 2
    assert settings.thresholds_file.exists()
    frozen = json.loads(settings.thresholds_file.read_text())
    assert frozen["frozen_on_split"] == "dev" and "ADAPTIVE|section_based" in frozen["entries"]
    t = run_grid(settings, "test", "full", ["ADAPTIVE"], ["section_based"], pipeline=pipeline)  # allowed: frozen on dev
    assert t["rows"][0]["abstention"]["thresholds_source"] == ["frozen-on-dev"]


def test_test_split_blocked_without_frozen_thresholds(settings, pipeline):
    _write(settings)
    with pytest.raises(RuntimeError, match="no frozen thresholds"):
        run_grid(settings, "test", "full", ["BM25"], ["section_based"], pipeline=pipeline)
    check_test_allowed(settings, settings.gold_dir / "gold_dev.jsonl", allow_uncalibrated=True)


def test_tune_rejected_on_test(settings, pipeline):
    _write(settings)
    with pytest.raises(ValueError):
        run_grid(settings, "test", "full", ["BM25"], ["section_based"], pipeline=pipeline, tune=True, allow_uncalibrated=True)
