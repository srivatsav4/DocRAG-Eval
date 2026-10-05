"""Heuristic citation verification (lexical support). NOT a factual-entailment check.

For each claim sentence in the answer we measure how many of its content terms appear in the cited evidence.
Numbers/identifiers must match exactly or the support score is halved. Passing != true; failing = suspicious.
Reported: citation_coverage, citation_correctness, unsupported_claims, invalid_citations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.retrieval.bm25 import tokenize

_CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_STRICT = re.compile(r"\b\d[\d.,]*\b|\b\w+_\w+\b")


@dataclass
class CitationReport:
    n_claims: int = 0
    citation_coverage: float = 0.0
    citation_correctness: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_claims": self.n_claims, "citation_coverage": round(self.citation_coverage, 4),
                "citation_correctness": round(self.citation_correctness, 4),
                "unsupported_claims": self.unsupported_claims, "invalid_citations": self.invalid_citations,
                "claims": self.claims,
                "method": "lexical-overlap heuristic (not entailment); see README Limitations"}


def _support(sentence: str, evidence_text: str) -> float:
    terms = list(dict.fromkeys(tokenize(sentence)))
    if not terms:
        return 1.0
    ev = set(tokenize(evidence_text))
    score = sum(t in ev for t in terms) / len(terms)
    low = evidence_text.lower()
    strict = [m.group(0).lower() for m in _STRICT.finditer(sentence)]
    if strict and any(s not in low for s in strict):
        score *= 0.5
    return score


def check_citations(answer: str, evidence: dict[str, str], min_support: float = 0.6) -> CitationReport:
    """`evidence` maps labels ('S1') -> evidence text."""
    rep = CitationReport()
    all_text = " ".join(evidence.values())
    n_cited = n_correct = 0
    for raw in _SENT.split(answer):
        sent = raw.strip()
        if len(tokenize(sent)) < 4:
            continue
        labels = [l.strip() for m in _CITE.findall(sent) for l in m.split(",")]
        clean = _CITE.sub("", sent).strip()
        invalid = [l for l in labels if l not in evidence]
        rep.invalid_citations += invalid
        valid = [l for l in labels if l in evidence]
        best_any = _support(clean, all_text)
        best_cited = max((_support(clean, evidence[l]) for l in valid), default=0.0)
        rep.n_claims += 1
        if labels:
            n_cited += 1
            n_correct += best_cited >= min_support
        if best_any < min_support:
            rep.unsupported_claims.append(clean)
        rep.claims.append({"text": clean, "cited": labels, "support_cited": round(best_cited, 3),
                           "support_any": round(best_any, 3)})
    if rep.n_claims:
        rep.citation_coverage = n_cited / rep.n_claims
    rep.citation_correctness = (n_correct / n_cited) if n_cited else 0.0
    return rep
