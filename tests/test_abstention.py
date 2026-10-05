import json

import pytest

from src.config import Settings, load_tunables
from src.evaluation.abstention_eval import abstention_metrics, sweep_thresholds, tune_threshold
from src.evidence.gate import ABSTAIN, ANSWER, LIMITED, Thresholds, decide, load_thresholds


def test_gate_decisions_and_oos_boost():
    th = Thresholds(0.65, 0.45, 0.10)
    assert decide(0.9, "HOW_TO", th).decision == ANSWER
    assert decide(0.5, "HOW_TO", th).decision == LIMITED
    assert decide(0.3, "HOW_TO", th).decision == ABSTAIN
    assert decide(0.76, "OUT_OF_SCOPE", th).decision == ANSWER and decide(0.7, "OUT_OF_SCOPE", th).decision == LIMITED
    assert decide(0.6, "OUT_OF_SCOPE", th).decision == LIMITED and decide(0.5, "OUT_OF_SCOPE", th).decision == ABSTAIN
    assert decide(0.5, "OUT_OF_SCOPE", th).required_answer == pytest.approx(0.75)


def _rec(answerable, conf, decision, hit=True):
    r = {"answerable": answerable, "confidence": conf, "decision": decision, "oos_boost": 0.0, "thresholds_source": "t"}
    if answerable:
        r["recall"] = {5: 1.0 if hit else 0.0}
    return r


RECS = [_rec(True, .9, ANSWER), _rec(True, .8, ANSWER), _rec(True, .4, ABSTAIN), _rec(True, .7, ANSWER, hit=False),
        _rec(False, .2, ABSTAIN), _rec(False, .3, ABSTAIN), _rec(False, .7, ANSWER)]


def test_abstention_metrics_hand_computed():
    m = abstention_metrics(RECS)
    assert m["correct_oos_refusals"] == 2 and m["incorrect_oos_answers"] == 1 and m["wrongly_refused_answerable"] == 1
    assert m["abstention_recall"] == pytest.approx(2 / 3) and m["abstention_precision"] == pytest.approx(2 / 3)
    assert m["coverage"] == pytest.approx(4 / 7) and m["selective_accuracy"] == pytest.approx(2 / 4)


def test_sweep_and_tune_on_dev_only():
    recs = [dict(r, decision=ANSWER) for r in RECS]
    t = tune_threshold(recs, margin=0.15)
    assert 0.3 < t["limited"] <= 0.4 and t["answer"] == pytest.approx(t["limited"] + 0.15)
    assert len(sweep_thresholds(recs)) > 10
    with pytest.raises(ValueError):
        tune_threshold([r for r in recs if r["answerable"]])


def test_load_thresholds_frozen_vs_default(tmp_path):
    s, tun = Settings(), load_tunables()
    s.thresholds_file = tmp_path / "t.json"
    assert load_thresholds(s, tun, "ADAPTIVE", "section_based").source == "default-uncalibrated"
    s.thresholds_file.write_text(json.dumps({"frozen_on_split": "dev", "entries": {"ADAPTIVE|section_based": {"answer": 0.5, "limited": 0.3}}}))
    th = load_thresholds(s, tun, "ADAPTIVE", "section_based")
    assert th.answer == 0.5 and th.source == "frozen-on-dev"


def test_pipeline_abstains_on_unanswerable(pipeline):
    r = pipeline.run("Give me a recipe for chocolate cake.", "ADAPTIVE", generate=True)
    assert r["decision"] == ABSTAIN and "sufficient evidence" in r["answer"] and pipeline.llm.calls == 0
    assert r["why"]["supporting_evidence"] == "INSUFFICIENT"
