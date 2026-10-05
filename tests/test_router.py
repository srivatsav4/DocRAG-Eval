import pytest

from src.retrieval.router import ADAPTIVE_ROUTES, QueryType, RuleBasedClassifier, get_strategy, select_strategy

C = RuleBasedClassifier()


@pytest.mark.parametrize("q,expected", [
    ("What is the default value of work_mem?", QueryType.LOOKUP),
    ("What is the syntax of the CREATE INDEX command?", QueryType.LOOKUP),
    ("How do I create a partial index in PostgreSQL?", QueryType.HOW_TO),
    ("How do I back up a database with pg_dump?", QueryType.HOW_TO),
    ("What is the difference between a view and a materialized view?", QueryType.MULTI_SECTION),
    ("How do I set up a standby server and enable WAL archiving?", QueryType.MULTI_SECTION),
    ("Explain why vacuum is needed in PostgreSQL", QueryType.CONCEPTUAL),
    ("Give me a recipe for chocolate cake.", QueryType.OUT_OF_SCOPE),
    ("How do I deploy a Kubernetes cluster on AWS?", QueryType.OUT_OF_SCOPE),
    ("hello", QueryType.OUT_OF_SCOPE),
])
def test_classification(q, expected):
    assert C.classify(q).query_type == expected


def test_rules_are_logged():
    a = C.classify("How do I create a partial index in PostgreSQL?")
    assert a.fired_rules == ["R4_howto_cue"] and a.features["howto_cue"] is True


def test_routing_table_complete_and_explained():
    assert set(ADAPTIVE_ROUTES) == set(QueryType)
    s, why = select_strategy(C.classify("What is the default value of work_mem?"))
    assert s.bm25_weight > s.dense_weight and not s.rerank and "LOOKUP" in why
    assert ADAPTIVE_ROUTES[QueryType.MULTI_SECTION].candidates > get_strategy("HYBRID_RERANK").candidates


def test_fixed_strategies():
    assert get_strategy("BM25").use_dense is False and get_strategy("DENSE").use_bm25 is False
    with pytest.raises(KeyError):
        get_strategy("NOPE")
