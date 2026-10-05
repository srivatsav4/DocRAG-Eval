"""Central configuration. Environment variables override defaults; tunables live in configs/default.json."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

CHUNKING_CONFIGS: dict[str, dict[str, Any]] = {
    "section_based": {"kind": "section", "max_words": 500, "overlap_words": 50},
    "fixed_300_overlap": {"kind": "fixed", "size": 300, "overlap": 50},
    "fixed_500_overlap": {"kind": "fixed", "size": 500, "overlap": 100},
}
STRATEGY_NAMES = ["BM25", "DENSE", "HYBRID_RRF", "HYBRID_RERANK", "ADAPTIVE"]

DEFAULT_TUNABLES: dict[str, Any] = {
    "bm25": {"k1": 1.5, "b": 0.75},
    "rrf_k": 60,
    "rerank_depth": 30,
    "evidence_weights": {"relevance": 0.35, "semantic": 0.20, "coverage": 0.25, "agreement": 0.10, "support": 0.10},
    "semantic_floor": 0.45,
    "semantic_ceiling": 0.85,
    "agreement_depth": 3,
    "support_min_chunk_coverage": 0.4,
    "thresholds": {"answer": 0.65, "limited": 0.45, "oos_boost": 0.10},
    "citation_min_support": 0.6,
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_tunables(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "configs" / "default.json"
    if path.exists():
        return _deep_merge(DEFAULT_TUNABLES, json.loads(path.read_text()))
    return copy.deepcopy(DEFAULT_TUNABLES)


@dataclass
class Settings:
    pg_docs_version: str = "16"
    pg_docs_base_url: str = "https://www.postgresql.org/docs"
    raw_html_dir: Path = ROOT / "data" / "raw_html"
    processed_dir: Path = ROOT / "data" / "processed"
    gold_dir: Path = ROOT / "data" / "gold"
    results_dir: Path = ROOT / "results"
    thresholds_file: Path = ROOT / "configs" / "thresholds.json"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedder_backend: str = "sentence-transformers"  # or "hash" (tests/smoke only)
    reranker_backend: str = "cross-encoder"  # or "lexical" (fallback, not a real cross-encoder)
    dense_backend: str = "auto"  # auto | numpy | pgvector
    database_url: str = "postgresql://docrag:docrag@localhost:5432/docrag"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_enabled: bool = True
    default_chunking: str = "section_based"
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        s = cls()
        s.pg_docs_version = e("PG_DOCS_VERSION", s.pg_docs_version)
        s.embedding_model = e("EMBEDDING_MODEL", s.embedding_model)
        s.reranker_model = e("RERANKER_MODEL", s.reranker_model)
        s.embedder_backend = e("EMBEDDER_BACKEND", s.embedder_backend)
        s.reranker_backend = e("RERANKER_BACKEND", s.reranker_backend)
        s.dense_backend = e("DENSE_BACKEND", s.dense_backend)
        s.database_url = e("DATABASE_URL", s.database_url)
        s.ollama_url = e("OLLAMA_URL", s.ollama_url)
        s.ollama_model = e("OLLAMA_MODEL", s.ollama_model)
        s.llm_enabled = e("LLM_ENABLED", "true").lower() in {"1", "true", "yes"}
        s.default_chunking = e("DEFAULT_CHUNKING", s.default_chunking)
        s.seed = int(e("SEED", s.seed))
        if e("DATA_DIR"):
            base = Path(e("DATA_DIR"))
            s.raw_html_dir, s.processed_dir, s.gold_dir = base / "raw_html", base / "processed", base / "gold"
        if e("RESULTS_DIR"):
            s.results_dir = Path(e("RESULTS_DIR"))
        return s
