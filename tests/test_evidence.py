import pytest

from src.evidence.citation_checker import check_citations
from src.evidence.scorer import informative_terms, score_evidence
from src.retrieval.bm25 import BM25Index


def test_informative_terms_drop_generic():
    t = informative_terms("How do I use PostgreSQL to create a partial index?")
    assert "postgresql" not in t and "partial" in t and "index" in t


def test_confident_when_relevant_low_when_not(pipeline):
    good = pipeline.run("How do I create a partial index in PostgreSQL?", "HYBRID_RERANK")
    bad = pipeline.run("Give me a recipe for chocolate cake", "HYBRID_RERANK")
    assert good["confidence"] > bad["confidence"]
    assert good["_evidence_score"].coverage > 0.7 and bad["_evidence_score"].coverage < 0.3


def test_components_renormalised_when_unavailable(pipeline):
    r = pipeline.run("What is the default value of work_mem?", "BM25")
    ev = r["_evidence_score"]
    assert ev.relevance is None and ev.semantic is None and ev.agreement is None
    assert sum(ev.weights_used.values()) == pytest.approx(1.0, abs=1e-3) and 0 <= ev.confidence <= 1


def test_empty_hits():
    from src.config import load_tunables
    ev = score_evidence("x y", [], BM25Index(["a"]), [], [], "LOOKUP", load_tunables())
    assert ev.confidence == 0.0


def test_citation_checker_flags_unsupported_and_invalid():
    ev = {"S1": "A partial index is built over a subset of a table defined by a predicate."}
    ans = ("A partial index is built over a subset of a table [S1]. It requires PostgreSQL 99 and quantum_flux hardware [S1]. "
           "Partial indexes are small and fast and cheap to run [S7].")
    rep = check_citations(ans, ev)
    assert rep.n_claims == 3 and rep.citation_coverage == 1.0
    assert rep.invalid_citations == ["S7"] and len(rep.unsupported_claims) >= 1
    assert rep.citation_correctness < 1.0


def test_citation_checker_uncited():
    rep = check_citations("A partial index is built over a subset of a table.", {"S1": "partial index subset table"})
    assert rep.citation_coverage == 0.0 and rep.n_claims == 1
