"""Deterministic stratified DEV/TEST split of data/gold/gold_all.jsonl (50 dev / 30 test by default)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from collections import defaultdict

from src.config import Settings

DEV_COUNTS = {"LOOKUP": 16, "HOW_TO": 16, "MULTI_SECTION": 9, "OUT_OF_SCOPE": 9}


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=s.seed)
    a = ap.parse_args()
    items = [json.loads(l) for l in (s.gold_dir / "gold_all.jsonl").read_text().splitlines() if l.strip()]
    assert len({i["id"] for i in items}) == len(items), "duplicate ids"
    by = defaultdict(list)
    for it in sorted(items, key=lambda x: x["id"]):
        by[it["type"]].append(it)
    rng = random.Random(a.seed)
    dev, test = [], []
    for t, lst in by.items():
        rng.shuffle(lst)
        n = DEV_COUNTS.get(t, round(len(lst) * 0.625))
        dev += lst[:n]
        test += lst[n:]
    for name, rows in (("gold_dev.jsonl", dev), ("gold_test.jsonl", test)):
        with (s.gold_dir / name).open("w") as f:
            for r in sorted(rows, key=lambda x: x["id"]):
                f.write(json.dumps(r) + "\n")
    print(f"dev={len(dev)} test={len(test)} seed={a.seed}")


if __name__ == "__main__":
    main()
