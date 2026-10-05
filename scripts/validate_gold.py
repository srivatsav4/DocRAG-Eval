"""Check that every gold section ID exists in the ingested corpus (as a section or an ancestor)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sys

from src.config import Settings
from src.ingest.loader import load_sections


def main() -> None:
    s = Settings.from_env()
    secs = load_sections(s)
    known = {x["section_id"] for x in secs} | {a for x in secs for a in x["ancestors"]}
    bad = 0
    for name in ("gold_dev.jsonl", "gold_test.jsonl"):
        for line in (s.gold_dir / name).read_text().splitlines():
            it = json.loads(line)
            if it["type"] != "OUT_OF_SCOPE" and not it["gold_sections"]:
                print(f"{name} {it['id']}: answerable question without gold sections"); bad += 1
            for g in it["gold_sections"]:
                if g not in known:
                    print(f"{name} {it['id']}: gold section '{g}' NOT FOUND in corpus (PG {s.pg_docs_version})"); bad += 1
    print("gold validation:", "OK" if not bad else f"{bad} problem(s) - fix labels before evaluating")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
