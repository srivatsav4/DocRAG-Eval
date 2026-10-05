"""Failure-analysis infrastructure. Auto-extracts CANDIDATES with a heuristic hint; humans assign the final type.

Nothing is pre-labelled: failure_type/root_cause/possible_fix stay null until a person reviews the case.
"""
from __future__ import annotations

FAILURE_TYPES = ["LEXICAL_MISMATCH", "SEMANTIC_MISMATCH", "BAD_CHUNKING", "MULTI_SECTION_FAILURE", "RERANKING_FAILURE",
                 "INSUFFICIENT_EVIDENCE", "CITATION_MISMATCH", "HALLUCINATION_UNSUPPORTED_ANSWER"]


def _hit(sections: list[str], gold: set[str]) -> bool:
    return bool(gold & set(sections))


def candidate_failures(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        gold = set(r["gold_sections"])
        hint = None
        if r["answerable"]:
            if r["decision"] == "ABSTAIN":
                hint = "INSUFFICIENT_EVIDENCE (over-abstention on answerable question)"
            elif r["recall"][10] == 0:
                in_b, in_d = _hit(r["bm25_top_sections"], gold), _hit(r["dense_top_sections"], gold)
                hint = ("LEXICAL_MISMATCH? (dense found gold, BM25 did not)" if in_d and not in_b else
                        "SEMANTIC_MISMATCH? (BM25 found gold, dense did not)" if in_b and not in_d else
                        "RERANKING_FAILURE? (gold in a retriever's top-5 but not in final top-10)" if in_b and in_d else
                        "UNKNOWN: inspect chunking / vocabulary gap")
            elif r["type"] == "MULTI_SECTION" and r["recall"][10] < 1:
                hint = "MULTI_SECTION_FAILURE? (some gold sections missing from top-10)"
        elif r["decision"] != "ABSTAIN":
            hint = "HALLUCINATION_RISK: out-of-scope question passed the evidence gate"
        if hint:
            out.append({"id": f"C-{r['id']}-{r['strategy']}-{r['chunking']}", "question": r["question"],
                        "expected_sections": r["gold_sections"], "retrieved_sections": r["retrieved_sections"],
                        "decision": r["decision"], "confidence": r["confidence"], "auto_hint": hint,
                        "failure_type": None, "root_cause": None, "possible_fix": None, "reviewed": False})
    return out
