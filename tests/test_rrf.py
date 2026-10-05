import pytest

from src.retrieval.hybrid import rrf_fuse


def test_rrf_known_values():
    f = dict(rrf_fuse({"a": [1, 2, 3], "b": [2, 3, 1]}, k=60))
    assert f[2] == pytest.approx(1 / 62 + 1 / 61) and f[1] == pytest.approx(1 / 61 + 1 / 63)
    assert [d for d, _ in rrf_fuse({"a": [1, 2, 3], "b": [2, 3, 1]}, k=60)][0] == 2


def test_weights_and_zero_weight():
    assert rrf_fuse({"a": [1, 2], "b": [2, 1]}, {"a": 3, "b": 1})[0][0] == 1
    assert [d for d, _ in rrf_fuse({"a": [1], "b": [2]}, {"b": 0})] == [1]


def test_deterministic_ties_and_top_n():
    out = rrf_fuse({"a": [5], "b": [4]}, k=60)
    assert [d for d, _ in out] == [4, 5]  # equal scores & equal best rank -> lower id first
    assert len(rrf_fuse({"a": [1, 2, 3]}, top_n=2)) == 2
