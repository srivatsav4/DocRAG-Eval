"""Rerankers return raw LOGITS (higher = more relevant). sigmoid(logit) is used as a 0-1 relevance proxy."""
from __future__ import annotations

import math
from typing import Protocol

from src.config import Settings
from src.retrieval.bm25 import tokenize


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy

        self.name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16)]


class LexicalOverlapReranker:
    """Deterministic FALLBACK for tests/smoke runs without model downloads. NOT a cross-encoder."""
    name = "lexical-overlap (fallback; not a cross-encoder)"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        out = []
        for t in texts:
            cov = len(q & set(tokenize(t))) / len(q) if q else 0.0
            out.append(8.0 * (cov - 0.5))
        return out


def get_reranker(settings: Settings) -> Reranker:
    if settings.reranker_backend == "lexical":
        return LexicalOverlapReranker()
    return CrossEncoderReranker(settings.reranker_model)
