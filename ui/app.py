"""Evidence Inspector (Streamlit). Talks to the FastAPI backend only. Run: streamlit run ui/app.py"""
import os

import httpx
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="DocRAG-Eval · Evidence Inspector", layout="wide")
st.title("DocRAG-Eval · Evidence Inspector")
st.caption("An evaluation-first, evidence-adaptive RAG system that dynamically selects retrieval strategies and abstains "
           "when documentation evidence is insufficient.")

EXAMPLES = ["How do I create a partial index in PostgreSQL?",
            "How do I configure PostgreSQL to control an unrelated technology that is not documented here?"]
with st.sidebar:
    api = st.text_input("API URL", API)
    strategy = st.selectbox("Strategy", ["ADAPTIVE", "HYBRID_RERANK", "HYBRID_RRF", "DENSE", "BM25"])
    generate = st.checkbox("Generate answer (needs Ollama)", True)
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"Example {i + 1}"):
            st.session_state["q"] = ex
q = st.text_input("Question", st.session_state.get("q", EXAMPLES[0]))

if st.button("Ask", type="primary") and q:
    try:
        r = httpx.post(f"{api}/ask", json={"query": q, "strategy": strategy, "generate": generate}, timeout=180)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        st.error(f"API error: {e}")
        st.stop()
    d = r.json()
    why, trace = d["why"], d["trace"]
    c = st.columns(4)
    c[0].metric("Decision", d["decision"])
    c[1].metric("Confidence", f"{d['confidence']:.2f}", help=f"required: {why['required_confidence']:.2f}")
    c[2].metric("Query type", d["query_type"])
    c[3].metric("Strategy", d["retrieval_strategy"])
    (st.error if d["decision"] == "ABSTAIN" else st.success if d["decision"] == "ANSWER" else st.warning)(
        d["answer"] or f"No generated answer ({d['answer_mode']}). Evidence is shown below.")
    st.subheader("Why I answered" if d["decision"] != "ABSTAIN" else "Why I refused")
    st.write(why["summary"])
    for ch in why["checks"]:
        st.write(("✓ " if ch["passed"] else "✗ ") + f"{ch['label']} — {ch['detail']}")
    st.write(f"**Supporting evidence:** {why['supporting_evidence']} · **Supporting sections:** "
             f"{', '.join(why['supporting_sections']) or '—'} · **Required confidence:** {why['required_confidence']:.2f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Citations & evidence")
        for e in d["evidence"]:
            with st.expander(f"[{e['label']}] {e['breadcrumb']} · score {e['score']:.2f}"):
                st.write(e["snippet"])
                st.markdown(f"[{e['section_id']}]({e['url']})")
        if d.get("citation_report"):
            st.subheader("Citation verification (heuristic)")
            st.json({k: v for k, v in d["citation_report"].items() if k != "claims"})
    with right:
        st.subheader("Latency breakdown (ms)")
        st.bar_chart({k: v for k, v in trace["timings"].items() if k != "total_ms"})
        st.caption(f"total: {trace['timings']['total_ms']} ms")
        st.subheader("Retrieval trace")
        st.json(trace)
