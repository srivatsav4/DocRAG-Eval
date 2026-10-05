import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.pipeline import RAGPipeline


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline=pipeline))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "ADAPTIVE" in r.json()["strategies"]


def test_ask_schema_and_trace_roundtrip(client):
    r = client.post("/ask", json={"query": "How do I create a partial index in PostgreSQL?"})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] in {"ANSWER", "ANSWER_WITH_LIMITATION"} and d["answer"] and d["evidence"][0]["url"].startswith("http")
    assert "_hits" not in d and d["why"]["checks"]
    t = client.get(f"/trace/{d['request_id']}")
    assert t.status_code == 200 and t.json()["query_type"] == "HOW_TO"


def test_ask_abstains(client):
    d = client.post("/ask", json={"query": "Give me a recipe for chocolate cake"}).json()
    assert d["decision"] == "ABSTAIN" and "sufficient evidence" in d["answer"]


def test_search_no_generation(client, pipeline):
    r = client.post("/search", json={"query": "default work_mem", "strategy": "BM25", "k": 3})
    assert r.status_code == 200 and 1 <= len(r.json()["evidence"]) <= 3 and pipeline.llm.calls == 0


@pytest.mark.parametrize("payload", [{}, {"query": "x"}, {"query": "valid question here", "strategy": "NOPE"},
                                     {"query": "valid question here", "k": 0}, {"query": "valid question here", "chunking": "bad"}])
def test_validation_errors(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_trace_404_and_metrics(client):
    assert client.get("/trace/doesnotexist").status_code == 404
    client.post("/ask", json={"query": "How do I create a partial index?"})
    m = client.get("/metrics").json()
    assert m["requests"]["ask"] == 1 and "mean_latency_ms" in m


def test_corpus_not_ready_returns_503(settings, tun):
    p = RAGPipeline(settings, tun)  # no processed data
    r = TestClient(create_app(pipeline=p)).post("/search", json={"query": "partial index"})
    assert r.status_code == 503 and r.json()["error"] == "corpus_not_ready"


def test_ingest_without_html_is_clean_error(settings, tun, tmp_path):
    settings.raw_html_dir = tmp_path / "empty"
    settings.raw_html_dir.mkdir()
    r = TestClient(create_app(pipeline=RAGPipeline(settings, tun), settings=settings)).post("/ingest", json={})
    assert r.status_code == 503


def test_evaluate_test_split_requires_confirmation(client):
    assert client.post("/evaluate", json={"split": "test"}).status_code == 400
