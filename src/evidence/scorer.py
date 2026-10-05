"""Evidence quality scoring. Deterministic weighted combination of five interpretable components.

confidence = sum_i(w_i * c_i) / sum_i(w_i)   over the components that are AVAILABLE for the strategy used.

  relevance : sigmoid(best reranker logit among final top-k)                      [needs reranker]
  semantic  : best dense cosine among final top-k, rescaled (cos-floor)/(ceil-floor) clipped to [0,1]  [needs dense]
  coverage  : IDF-weighted share of informative query terms present in the union of top-k evidence text
  agreement : |top-3 sections of BM25  ∩  top-3 sections of dense| / 3            [needs both retrievers]
  support   : min(1, #sections whose own coverage >= 0.4 / needed), needed = 2 for MULTI_SECTION else 1

LIMITATIONS: the formula is a hand-designed heuristic, NOT a calibrated probability. 'semantic' rescaling is
embedding-model specific. Weights/thresholds are only meaningful after tuning on DEV (see scripts/run_eval.py --tune).
Query terms absent from the corpus get maximal IDF, so questions about undocumented topics lose coverage fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.reranker import sigmoid

GENERIC_TERMS = frozenset({"postgresql", "postgres", "database", "use", "using", "want", "need", "please", "tell",
                           "make", "get", "give", "show", "way", "work", "unrelated"})


def informative_terms(query: str) -> list[str]:
    return [t for t in dict.fromkeys(tokenize(query)) if t not in GENERIC_TERMS]


@dataclass
class EvidenceScore:
    confidence: float
    relevance: float | None
    semantic: float | None
    coverage: float
    agreement: float | None
    support: float
    n_supporting_sections: int
    weights_used: dict[str, float] = field(default_factory=dict)
    supporting_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        r = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
        return {"confidence": r(self.confidence), "relevance": r(self.relevance), "semantic": r(self.semantic),
                "coverage": r(self.coverage), "agreement": r(self.agreement), "support": r(self.support),
                "n_supporting_sections": self.n_supporting_sections, "weights_used": self.weights_used,
                "supporting_sections": self.supporting_sections}


def _weighted_coverage(terms: list[str], have: set[str] | frozenset[str], bm25: BM25Index) -> float:
    if not terms:
        return 0.0
    tot = sum(bm25.idf(t) for t in terms)
    got = sum(bm25.idf(t) for t in terms if t in have)
    return got / tot if tot else 0.0


def score_evidence(query: str, top_hits: list, bm25: BM25Index, bm25_top_sections: list[str],
                   dense_top_sections: list[str], query_type: str, tun: dict) -> EvidenceScore:
    """`top_hits`: final top-k Hit objects (need .idx, .chunk.section_id, .rerank_score, .dense_score)."""
    if not top_hits:
        return EvidenceScore(0.0, None, None, 0.0, None, 0.0, 0)
    terms = informative_terms(query)
    union: set[str] = set()
    for h in top_hits:
        union |= bm25.doc_terms[h.idx]
    coverage = _weighted_coverage(terms, union, bm25)

    rr = [h.rerank_score for h in top_hits if h.rerank_score is not None]
    relevance = sigmoid(max(rr)) if rr else None

    ds = [h.dense_score for h in top_hits if h.dense_score is not None]
    semantic = None
    if ds:
        lo, hi = tun["semantic_floor"], tun["semantic_ceiling"]
        semantic = min(1.0, max(0.0, (max(ds) - lo) / (hi - lo)))

    agreement = None
    if bm25_top_sections and dense_top_sections:
        d = tun["agreement_depth"]
        agreement = len(set(bm25_top_sections[:d]) & set(dense_top_sections[:d])) / d

    per_section: dict[str, float] = {}
    for h in top_hits:
        c = _weighted_coverage(terms, bm25.doc_terms[h.idx], bm25)
        sid = h.chunk.section_id
        per_section[sid] = max(per_section.get(sid, 0.0), c)
    supporting = [s for s, c in per_section.items() if c >= tun["support_min_chunk_coverage"]]
    needed = 2 if query_type == "MULTI_SECTION" else 1
    support = min(1.0, len(supporting) / needed)

    comps = {"relevance": relevance, "semantic": semantic, "coverage": coverage, "agreement": agreement,
             "support": support}
    w = {k: tun["evidence_weights"][k] for k, v in comps.items() if v is not None}
    conf = sum(w[k] * comps[k] for k in w) / sum(w.values())
    return EvidenceScore(conf, relevance, semantic, coverage, agreement, support, len(supporting),
                         {k: round(v / sum(w.values()), 4) for k, v in w.items()}, supporting)
