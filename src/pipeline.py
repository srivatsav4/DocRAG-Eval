"""RAG orchestration: classify -> route -> retrieve -> fuse -> rerank -> score evidence -> gate -> generate -> verify."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from src.config import Settings, load_tunables
from src.evidence.citation_checker import check_citations
from src.evidence.gate import ABSTAIN, LIMITED, REFUSAL_MESSAGE, Thresholds, decide, load_thresholds
from src.evidence.scorer import EvidenceScore, score_evidence
from src.generation.ollama_client import LLMUnavailable
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.ingest.chunker import Chunk
from src.ingest.loader import load_chunks
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import build_dense_index, get_embedder
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.reranker import get_reranker, sigmoid
from src.retrieval.router import RuleBasedClassifier, Strategy, get_strategy, select_strategy
from src.utils.logging import get_logger
from src.utils.timing import StageTimer

log = get_logger("pipeline")


class PipelineNotReady(RuntimeError):
    """Processed corpus is missing (run ingestion)."""


@dataclass
class Hit:
    idx: int
    chunk: Chunk
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class RetrievalOutput:
    hits: list[Hit]
    bm25_top_sections: list[str]
    dense_top_sections: list[str]
    candidate_count: int


class CorpusIndex:
    def __init__(self, chunking: str, chunks: list[Chunk], tun: dict, settings: Settings, embedder_getter: Callable):
        self.chunking, self.chunks, self.settings, self._emb = chunking, chunks, settings, embedder_getter
        self.bm25 = BM25Index([c.index_text for c in chunks], tun["bm25"]["k1"], tun["bm25"]["b"])
        self._dense = None

    @property
    def dense(self):
        if self._dense is None:
            self._dense = build_dense_index(self.chunks, self._emb(), self.settings, self.chunking)
        return self._dense


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, tunables: dict | None = None, embedder=None, reranker=None,
                 llm=None, chunks_provider: Callable[[str], list[Chunk]] | None = None, classifier=None):
        self.settings = settings or Settings.from_env()
        self.tun = tunables or load_tunables()
        self._embedder, self._reranker, self.llm = embedder, reranker, llm
        self._chunks_provider = chunks_provider
        self.classifier = classifier or RuleBasedClassifier()
        self._indexes: dict[str, CorpusIndex] = {}
        self._thr: dict[tuple[str, str], Thresholds] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.settings)
        return self._reranker

    def index(self, chunking: str) -> CorpusIndex:
        if chunking not in self._indexes:
            try:
                chunks = self._chunks_provider(chunking) if self._chunks_provider else load_chunks(self.settings, chunking)
            except FileNotFoundError as e:
                raise PipelineNotReady(str(e)) from e
            self._indexes[chunking] = CorpusIndex(chunking, chunks, self.tun, self.settings, lambda: self.embedder)
        return self._indexes[chunking]

    def reset(self) -> None:
        self._indexes.clear()

    def thresholds(self, strategy: str, chunking: str) -> Thresholds:
        key = (strategy, chunking)
        if key not in self._thr:
            self._thr[key] = load_thresholds(self.settings, self.tun, strategy, chunking)
        return self._thr[key]

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, strat: Strategy, chunking: str, timer: StageTimer) -> RetrievalOutput:
        idx = self.index(chunking)
        if strat.use_dense:  # warm-up OUTSIDE timers: model load + corpus embedding are one-off costs
            _ = idx.dense, self.embedder
        if strat.rerank:
            _ = self.reranker
        cand = strat.candidates
        bm: list[tuple[int, float]] = []
        dn: list[tuple[int, float]] = []
        if strat.use_bm25:
            with timer.stage("bm25"):
                bm = idx.bm25.search(query, cand)
        if strat.use_dense:
            with timer.stage("dense"):
                dn = idx.dense.search(self.embedder.encode_query(query), cand)
        with timer.stage("fusion"):
            if bm and dn:
                fused = rrf_fuse({"bm25": [i for i, _ in bm], "dense": [i for i, _ in dn]},
                                 {"bm25": strat.bm25_weight, "dense": strat.dense_weight}, self.tun["rrf_k"], cand)
            else:
                fused = list(bm or dn)
            bm_rank = {i: r for r, (i, _) in enumerate(bm, 1)}
            dn_rank = {i: r for r, (i, _) in enumerate(dn, 1)}
            bm_sc, dn_sc = dict(bm), dict(dn)
            hits = [Hit(i, idx.chunks[i], bm_sc.get(i), bm_rank.get(i), dn_sc.get(i), dn_rank.get(i), s)
                    for i, s in fused]
        if strat.rerank and hits:
            depth = self.tun["rerank_depth"] * (2 if strat.candidates >= 100 else 1)
            with timer.stage("rerank"):
                head = hits[:depth]
                scores = self.reranker.score(query, [h.chunk.index_text for h in head])
                for h, s in zip(head, scores):
                    h.rerank_score = s
                head.sort(key=lambda h: (-h.rerank_score, h.idx))
                hits = head + hits[depth:]

        def sections(ranking):
            return list(dict.fromkeys(idx.chunks[i].section_id for i, _ in ranking))[:5]

        return RetrievalOutput(hits, sections(bm), sections(dn), len(hits))

    # ------------------------------------------------------------------ full request
    def run(self, query: str, strategy: str = "ADAPTIVE", chunking: str | None = None, k: int | None = None,
            generate: bool = False, request_id: str | None = None) -> dict:
        chunking = chunking or self.settings.default_chunking
        rid = request_id or uuid.uuid4().hex[:12]
        timer = StageTimer()
        with timer.stage("classification"):
            analysis = self.classifier.classify(query)
        if strategy.upper() == "ADAPTIVE":
            strat, route_reason = select_strategy(analysis)
        else:
            strat, route_reason = get_strategy(strategy.upper()), "fixed strategy (no routing)"
        out = self.retrieve(query, strat, chunking, timer)
        final_k = k or strat.final_k
        top = out.hits[:final_k]

        with timer.stage("evidence"):
            ev = score_evidence(query, top, self.index(chunking).bm25, out.bm25_top_sections, out.dense_top_sections,
                                analysis.query_type.value, self.tun)
        th = self.thresholds(strategy.upper(), chunking)
        gate = decide(ev.confidence, analysis.query_type.value, th)

        evidence = [self._evidence_item(i + 1, h) for i, h in enumerate(top)]
        answer, answer_mode, citation = None, "not_requested", None
        if gate.decision == ABSTAIN:
            answer, answer_mode = REFUSAL_MESSAGE, "abstained"
        elif generate:
            if self.llm is None:
                answer_mode = "retrieval_only (LLM disabled)"
            else:
                prompt = build_user_prompt(query, [(e["label"], e["breadcrumb"], top[i].chunk.text)
                                                   for i, e in enumerate(evidence)], gate.decision == LIMITED)
                try:
                    with timer.stage("generation"):
                        answer = self.llm.generate(SYSTEM_PROMPT, prompt)
                    answer_mode = "generated"
                    with timer.stage("citation_check"):
                        citation = check_citations(answer, {e["label"]: top[i].chunk.text for i, e in enumerate(evidence)},
                                                   self.tun["citation_min_support"]).to_dict()
                    if gate.decision == LIMITED:
                        answer = "Note: evidence is only partially relevant; verify against the cited sections.\n\n" + answer
                except LLMUnavailable as e:
                    log.warning("LLM unavailable: %s", e)
                    answer, answer_mode = None, "retrieval_only (LLM unavailable)"

        timings = timer.as_dict()
        trace = {
            "request_id": rid, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query": query,
            "query_type": analysis.query_type.value, "classification": analysis.to_dict(),
            "retrieval_strategy": strat.name if strategy.upper() == "ADAPTIVE" else strategy.upper(),
            "requested_strategy": strategy.upper(),
            "strategy_params": {"bm25_weight": strat.bm25_weight, "dense_weight": strat.dense_weight,
                                "candidates": strat.candidates, "rerank": strat.rerank},
            "route_reason": route_reason, "chunking": chunking, "candidate_count": out.candidate_count,
            "final_k": final_k, "confidence": round(ev.confidence, 4), "decision": gate.decision,
            "gate": gate.to_dict(), "evidence_score": ev.to_dict(), "timings": timings,
            "models": {"embedder": self.embedder.name if self._embedder else None,
                       "reranker": getattr(self._reranker, "name", None) if self._reranker else None,
                       "llm": getattr(self.llm, "name", None)},
        }
        return {
            "request_id": rid, "query": query, "answer": answer, "answer_mode": answer_mode,
            "decision": gate.decision, "confidence": round(ev.confidence, 4), "query_type": analysis.query_type.value,
            "retrieval_strategy": trace["retrieval_strategy"], "evidence": evidence,
            "supporting_sections": ev.supporting_sections or list(dict.fromkeys(e["section_id"] for e in evidence)),
            "citation_report": citation, "why": self._why(analysis.query_type.value, strat, ev, gate, citation, trace),
            "trace": trace, "_hits": out.hits, "_bm25_top": out.bm25_top_sections, "_dense_top": out.dense_top_sections, "_evidence_score": ev, "_gate": gate,
        }

    @staticmethod
    def _evidence_item(n: int, h: Hit) -> dict:
        prob = sigmoid(h.rerank_score) if h.rerank_score is not None else None
        score = prob if prob is not None else (h.dense_score if h.dense_score is not None else
                                               (h.bm25_score if h.bm25_score is not None else h.fused_score))
        c = h.chunk
        return {"label": f"S{n}", "chunk_id": c.chunk_id, "section_id": c.section_id, "section_title": c.title,
                "breadcrumb": c.breadcrumb, "url": c.url, "snippet": c.text[:400] + ("…" if len(c.text) > 400 else ""),
                "score": round(float(score), 4), "rerank_prob": None if prob is None else round(prob, 4),
                "rerank_logit": h.rerank_score, "dense_score": h.dense_score, "bm25_score": h.bm25_score,
                "fused_score": round(h.fused_score, 6)}

    @staticmethod
    def _why(qtype: str, strat: Strategy, ev: EvidenceScore, gate, citation: dict | None, trace: dict) -> dict:
        checks = []

        def add(label, val, thr):
            if val is not None:
                checks.append({"label": label, "passed": val >= thr, "detail": f"{val:.2f} (needs >= {thr:.2f})"})

        add("Strong lexical evidence (idf-weighted query-term coverage)", ev.coverage, 0.70)
        add("Strong semantic evidence (dense similarity)", ev.semantic, 0.60)
        add("Reranker relevance", ev.relevance, 0.60)
        add("Retrievers agree (BM25 vs dense top-3)", ev.agreement, 0.34)
        checks.append({"label": "Enough supporting sections", "passed": ev.support >= 1.0,
                       "detail": f"{ev.n_supporting_sections} section(s)"})
        if citation:
            checks.append({"label": "Citation coverage", "passed": citation["citation_coverage"] >= 0.8,
                           "detail": f"{citation['citation_coverage']:.0%}"})
        refused = gate.decision == ABSTAIN
        summary = ("No sufficiently relevant documentation evidence was found." if refused else
                   "Evidence cleared the gate." if gate.decision != LIMITED else
                   "Evidence is partial; answered with a limitation notice.")
        return {"query_type": qtype, "selected_strategy": trace["retrieval_strategy"],
                "evidence_confidence": gate.confidence, "required_confidence": gate.required_answer,
                "supporting_evidence": "INSUFFICIENT" if refused else "SUFFICIENT",
                "supporting_sections": ev.supporting_sections, "decision": gate.decision, "checks": checks,
                "citation_coverage": citation["citation_coverage"] if citation else None, "summary": summary}


def public_result(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}
