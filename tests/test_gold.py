import json
from collections import Counter
from pathlib import Path

G = Path(__file__).resolve().parent.parent / "data" / "gold"


def _load(n):
    return [json.loads(l) for l in (G / n).read_text().splitlines()]


def test_split_is_disjoint_and_sized():
    dev, test = _load("gold_dev.jsonl"), _load("gold_test.jsonl")
    assert len(dev) == 50 and len(test) == 30 and not ({d["id"] for d in dev} & {t["id"] for t in test})
    assert Counter(i["type"] for i in dev + test) == {"LOOKUP": 25, "HOW_TO": 25, "MULTI_SECTION": 15, "OUT_OF_SCOPE": 15}


def test_schema():
    for it in _load("gold_all.jsonl"):
        assert it["difficulty"] in {"easy", "medium", "hard"}
        assert bool(it["gold_sections"]) == (it["type"] != "OUT_OF_SCOPE")
