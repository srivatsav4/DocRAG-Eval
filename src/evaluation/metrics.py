"""Ranking metrics with binary relevance. `ranked` = de-duplicated list of section keys, best first."""
from __future__ import annotations

import math
import random
from statistics import mean


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, s in enumerate(ranked[:k], start=1):
        if s in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, s in enumerate(ranked[:k], start=1) if s in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap of mean(a - b) over queries. 95% percentile CI. Small n => wide intervals; report them."""
    assert len(a) == len(b)
    if not a:
        return {"n": 0, "mean_diff": None, "ci95": None}
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return {"n": len(d), "mean_diff": mean(d), "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d)}
