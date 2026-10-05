def test_adaptive_trace_and_why(pipeline):
    r = pipeline.run("How do I create a partial index in PostgreSQL?", "ADAPTIVE", generate=True)
    t = r["trace"]
    assert t["query_type"] == "HOW_TO" and t["classification"]["fired_rules"] and "HYBRID_RERANK" in t["retrieval_strategy"]
    for k in ("classification_ms", "bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"):
        assert k in t["timings"]
    assert r["evidence"][0]["section_id"] == "indexes-partial" and r["decision"] != "ABSTAIN"
    assert r["answer_mode"] == "generated" and r["citation_report"]["citation_coverage"] == 1.0
    assert {"label", "url", "snippet", "breadcrumb", "section_title", "score"} <= set(r["evidence"][0])


def test_each_fixed_strategy_runs(pipeline):
    for s in ("BM25", "DENSE", "HYBRID_RRF", "HYBRID_RERANK"):
        r = pipeline.run("What is the default value of work_mem?", s)
        assert r["evidence"] and r["trace"]["requested_strategy"] == s
    assert "dense_ms" not in pipeline.run("work_mem default", "BM25")["trace"]["timings"]


def test_lookup_routes_to_lexical_no_rerank(pipeline):
    r = pipeline.run("What is the default value of work_mem?", "ADAPTIVE")
    assert "rerank_ms" not in r["trace"]["timings"] and r["evidence"][0]["section_id"] == "runtime-config-resource"


def test_llm_disabled_returns_retrieval_only(pipeline):
    pipeline.llm = None
    r = pipeline.run("How do I create a partial index?", "ADAPTIVE", generate=True)
    assert r["answer"] is None and r["answer_mode"].startswith("retrieval_only")
