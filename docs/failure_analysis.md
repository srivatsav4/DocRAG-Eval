# Failure analysis

Status: **NO REAL FAILURES RECORDED YET** — run the evaluation, then review `results/<run>/failure_candidates.jsonl`.

`failure_candidates.jsonl` is auto-extracted (answerable questions with Recall@10 = 0, over-abstentions, out-of-scope questions
that passed the gate, partial multi-section recall) with a *heuristic* `auto_hint`. A human must set `failure_type`,
`root_cause` and `possible_fix`, then copy the reviewed entry into `results/failures.jsonl`.

Types: LEXICAL_MISMATCH, SEMANTIC_MISMATCH, BAD_CHUNKING, MULTI_SECTION_FAILURE, RERANKING_FAILURE, INSUFFICIENT_EVIDENCE,
CITATION_MISMATCH, HALLUCINATION_UNSUPPORTED_ANSWER.

## Entry format

```json
{
  "id": "F001",
  "question": "...",
  "expected_sections": ["..."],
  "retrieved_sections": ["..."],
  "failure_type": "LEXICAL_MISMATCH",
  "root_cause": "...",
  "possible_fix": "..."
}
```

## EXAMPLE (illustrative format only — not a real observed failure)

```json
{"id": "EXAMPLE-F000", "question": "(example) How do I stop a runaway query?", "expected_sections": ["(example) functions-admin"],
 "retrieved_sections": ["(example) sql-cancel"], "failure_type": "LEXICAL_MISMATCH",
 "root_cause": "(example) question says 'stop'; docs say 'cancel'/'terminate'", "possible_fix": "(example) query expansion / synonyms"}
```
