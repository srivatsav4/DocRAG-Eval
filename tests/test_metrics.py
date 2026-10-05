import math

import pytest

from src.evaluation.metrics import ndcg_at_k, paired_bootstrap, percentile, recall_at_k, reciprocal_rank
from src.evaluation.retrieval_eval import resolve_keys


class C:  # minimal chunk stand-in
    def __init__(self, sid, anc=()):
        self.section_id, self.ancestors = sid, list(anc)


def test_recall_rr_ndcg_known_values():
    ranked, gold = ["a", "b", "c", "d"], {"c", "z"}
    assert recall_at_k(ranked, gold, 3) == 0.5 and recall_at_k(ranked, gold, 2) == 0.0
    assert reciprocal_rank(ranked, gold) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx((1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a"], {"a"}, 3) == 1.0 and ndcg_at_k([], set(), 3) == 0.0


def test_resolve_keys_dedup_and_ancestor_match():
    keys = resolve_keys([C("sql-createindex-synopsis", ["sql-createindex"]), C("sql-createindex"), C("x")], {"sql-createindex"})
    assert keys == ["sql-createindex", "x"]


def test_percentile_and_bootstrap():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    r = paired_bootstrap([1, 1, 1, 0], [0, 0, 0, 0], n_boot=200, seed=1)
    assert r["mean_diff"] == 0.75 and r["ci95"][0] <= 0.75 <= r["ci95"][1] and r["wins"] == 3
    assert paired_bootstrap([], [])["n"] == 0
    assert paired_bootstrap([1, 0, 1], [0, 1, 0], seed=3) == paired_bootstrap([1, 0, 1], [0, 1, 0], seed=3)
