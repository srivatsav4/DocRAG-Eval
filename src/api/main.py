"""FastAPI backend (independent of the UI). `create_app(pipeline=...)` allows dependency injection in tests."""
from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.api.schemas import AskRequest, AskResponse, EvaluateRequest, IngestRequest, SearchRequest, SearchResponse
from src.config import CHUNKING_CONFIGS, STRATEGY_NAMES, Settings, load_tunables
from src.generation.ollama_client import OllamaClient
from src.ingest.loader import ingest_directory
from src.pipeline import PipelineNotReady, RAGPipeline, public_result
from src.utils.logging import get_logger

log = get_logger("api")
MAX_TRACES = 1000


def create_app(pipeline: RAGPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (pipeline.settings if pipeline else Settings.from_env())
    app = FastAPI(title="DocRAG-Eval", version="0.1.0",
                  description="Evaluation-first, evidence-adaptive RAG over technical documentation.")
    state = {"pipeline": pipeline, "lock": threading.Lock()}
    traces: OrderedDict[str, dict] = OrderedDict()
    counters = {"requests": defaultdict(int), "decisions": defaultdict(int), "query_types": defaultdict(int),
                "latency_sum": defaultdict(float), "latency_n": 0, "last_evaluation": None}

    def get_pipeline() -> RAGPipeline:
        with state["lock"]:
            if state["pipeline"] is None:
                llm = OllamaClient(settings.ollama_url, settings.ollama_model) if settings.llm_enabled else None
                state["pipeline"] = RAGPipeline(settings, load_tunables(), llm=llm)
            return state["pipeline"]

    def record(endpoint: str, res: dict) -> None:
        counters["requests"][endpoint] += 1
        counters["decisions"][res["decision"]] += 1
        counters["query_types"][res["query_type"]] += 1
        counters["latency_n"] += 1
        for k, v in res["trace"]["timings"].items():
            counters["latency_sum"][k] += v
        traces[res["request_id"]] = res["trace"]
        while len(traces) > MAX_TRACES:
            traces.popitem(last=False)

    @app.exception_handler(PipelineNotReady)
    async def _not_ready(_: Request, exc: PipelineNotReady):
        return JSONResponse(status_code=503, content={"error": "corpus_not_ready", "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def _runtime(_: Request, exc: RuntimeError):
        return JSONResponse(status_code=409, content={"error": "conflict", "detail": str(exc)})

    @app.get("/health")
    def health():
        p = state["pipeline"]
        llm = getattr(p, "llm", None) if p else None
        return {"status": "ok", "corpus_ready": (settings.processed_dir / "chunks_section_based.jsonl").exists() or p is not None,
                "llm_configured": bool(settings.llm_enabled), "llm_available": bool(llm.available()) if hasattr(llm, "available") else None,
                "strategies": STRATEGY_NAMES, "chunking": list(CHUNKING_CONFIGS), "pg_docs_version": settings.pg_docs_version}

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        res = await run_in_threadpool(lambda: get_pipeline().run(req.query, req.strategy, req.chunking, req.k, generate=False))
        record("search", res)
        return public_result(res)

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        res = await run_in_threadpool(lambda: get_pipeline().run(req.query, req.strategy, req.chunking, req.k, generate=req.generate))
        record("ask", res)
        return public_result(res)

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        try:
            manifest = await run_in_threadpool(lambda: ingest_directory(settings, req.strategies, req.max_pages))
        except FileNotFoundError as e:
            raise PipelineNotReady(str(e)) from e
        if state["pipeline"] is not None:
            state["pipeline"].reset()
        counters["requests"]["ingest"] += 1
        return manifest

    @app.post("/evaluate")
    async def evaluate(req: EvaluateRequest):
        if req.split == "test" and not req.confirm_frozen_thresholds:
            raise HTTPException(400, "TEST split requires confirm_frozen_thresholds=true (thresholds must be frozen on DEV first).")
        from src.evaluation.experiment_runner import run_grid

        summary = await run_in_threadpool(lambda: run_grid(settings, req.split, req.mode, req.strategies, req.chunkings, req.limit,
                                                           pipeline=get_pipeline()))
        counters["requests"]["evaluate"] += 1
        counters["last_evaluation"] = {"run_id": summary["run_id"], "out_dir": summary["out_dir"], "mode": summary["mode"], "split": summary["split"]}
        return {k: summary[k] for k in ("run_id", "split", "mode", "n_questions", "out_dir", "rows", "adaptive_vs_fixed_bootstrap", "WARNING")}

    @app.get("/metrics")
    def metrics():
        n = counters["latency_n"] or 1
        return {"requests": dict(counters["requests"]), "decisions": dict(counters["decisions"]),
                "query_types": dict(counters["query_types"]),
                "mean_latency_ms": {k: round(v / n, 2) for k, v in counters["latency_sum"].items()},
                "traces_stored": len(traces), "last_evaluation": counters["last_evaluation"],
                "note": "Online counters since process start. Offline retrieval metrics live in results/<run>/summary.json."}

    @app.get("/trace/{request_id}")
    def trace(request_id: str):
        if request_id not in traces:
            raise HTTPException(404, f"no trace for request_id {request_id!r}")
        return traces[request_id]

    return app


app = create_app()
