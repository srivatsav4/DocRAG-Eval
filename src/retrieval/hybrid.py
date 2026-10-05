"""Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum_r w_r / (k + rank_r(d)), rank starting at 1."""
from __future__ import annotations


def rrf_fuse(rankings: dict[str, list[int]], weights: dict[str, float] | None = None, k: int = 60,
             top_n: int | None = None) -> list[tuple[int, float]]:
    weights = weights or {}
    score: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for name, ranking in rankings.items():
        w = weights.get(name, 1.0)
        if w == 0:
            continue
        for r, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + w / (k + r)
            best_rank[doc] = min(best_rank.get(doc, r), r)
    fused = sorted(score.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))  # deterministic tie-break
    return fused[:top_n] if top_n else fused
