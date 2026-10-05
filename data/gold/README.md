# Gold data

* `gold_all.jsonl` — 80 questions (25 LOOKUP, 25 HOW_TO, 15 MULTI_SECTION, 15 OUT_OF_SCOPE) with difficulty labels.
* `gold_dev.jsonl` (50) / `gold_test.jsonl` (30) — deterministic stratified split from `scripts/build_gold_split.py` (seed 42).

**Label status: `unverified-seed`.** Gold section IDs are DocBook anchors (lower-cased, e.g. `indexes-partial`) written from
knowledge of the PostgreSQL 16 documentation structure. They have NOT been checked against the ingested corpus and relevance
judgements have not been double-annotated. Before trusting any number:

1. `make ingest` then `python scripts/validate_gold.py` (fails if an ID does not exist in the corpus),
2. manually review each question/section pair (and add other acceptable sections where several pages answer the question),
3. only then run evaluation. Never edit TEST labels after looking at TEST results.

A gold ID matches a retrieved chunk if it equals the chunk's section or one of its ancestors (e.g. `sql-createindex` covers its refsections).
