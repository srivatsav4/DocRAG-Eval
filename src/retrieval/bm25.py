"""From-scratch BM25 (Okapi) with an identifier-aware tokenizer. No external search library on purpose."""
from __future__ import annotations

import heapq
import math
import re

_WORD = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being do does did how what which when where why who "
    "can could should would may might will with by as at from that this these those it its into than then there "
    "their i you we my your our me not no if so but about over under per via also".split()
)


def stem(t: str) -> str:
    """Tiny suffix stripper (deterministic, explainable). Not Porter; good enough for technical prose."""
    if len(t) > 3 and not t.isdigit():
        for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                if suf == "s" and t.endswith("ss"):
                    break
                t = t[: -len(suf)] + rep
                break
        if t.endswith("e") and len(t) > 4:
            t = t[:-1]
    return t


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """Lower-case, keep identifiers (work_mem) AND their parts (work, mem), drop stopwords, stem."""
    out: list[str] = []
    for m in _WORD.findall(text.lower()):
        parts = [m] + ([p for p in m.split("_") if p] if "_" in m else [])
        for p in dict.fromkeys(parts):
            if not keep_stop and (p in STOPWORDS or (len(p) < 2 and not p.isdigit())):
                continue
            out.append(stem(p))
    return out


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.doc_terms: list[frozenset[str]] = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.doc_len.append(len(toks))
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.doc_terms.append(frozenset(counts))
            for t, c in counts.items():
                self.postings.setdefault(t, []).append((i, c))
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        """Robertson-Sparck-Jones idf (always >= 0). Unseen terms get the maximum idf (df = 0)."""
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in dict.fromkeys(tokenize(query)):
            post = self.postings.get(t)
            if not post:
                continue
            idf = self.idf(t)
            for i, tf in post:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
