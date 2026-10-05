from src.retrieval.bm25 import BM25Index, stem, tokenize

DOCS = ["partial index with where predicate", "autovacuum removes dead rows", "work_mem sets memory for sorts", "ssl connections encrypt traffic"]


def test_tokenizer_identifiers_and_stemming():
    t = tokenize("Set work_mem for the indexes")
    assert "work_mem" in t and "work" in t and "mem" in t and "index" in t and "the" not in t
    assert stem("creating") == stem("create") and stem("indexes") == "index"


def test_ranking():
    bm = BM25Index(DOCS)
    assert bm.search("how do I create a partial index", 3)[0][0] == 0
    assert bm.search("work_mem", 1)[0][0] == 2
    assert bm.search("zzzz unknown", 3) == []


def test_idf_monotonic_and_unseen_max():
    bm = BM25Index(DOCS + ["index index"])
    assert bm.idf("autovacuum") > bm.idf("index") and bm.idf("neverseen") >= bm.idf("autovacuum")


def test_scores_sorted_and_deterministic():
    bm = BM25Index(["a b", "a b", "a c"])
    r1, r2 = bm.search("a b", 3), bm.search("a b", 3)
    assert r1 == r2 and [s for _, s in r1] == sorted((s for _, s in r1), reverse=True)
