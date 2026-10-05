from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Strategy = Literal["BM25", "DENSE", "HYBRID_RRF", "HYBRID_RERANK", "ADAPTIVE"]
Chunking = Literal["section_based", "fixed_300_overlap", "fixed_500_overlap"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    strategy: Strategy = "ADAPTIVE"
    chunking: Optional[Chunking] = None
    k: Optional[int] = Field(None, ge=1, le=20)


class AskRequest(SearchRequest):
    generate: bool = True


class IngestRequest(BaseModel):
    strategies: Optional[list[Chunking]] = None
    max_pages: Optional[int] = Field(None, ge=1)


class EvaluateRequest(BaseModel):
    split: Literal["dev", "test"] = "dev"
    mode: Literal["smoke", "full"] = "smoke"
    strategies: Optional[list[Strategy]] = None
    chunkings: Optional[list[Chunking]] = None
    limit: Optional[int] = Field(None, ge=1)
    confirm_frozen_thresholds: bool = Field(False, description="Must be true to touch the TEST split.")


class EvidenceItem(BaseModel):
    label: str
    chunk_id: str
    section_id: str
    section_title: str
    breadcrumb: str
    url: str
    snippet: str
    score: float
    rerank_prob: Optional[float] = None
    rerank_logit: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None


class SearchResponse(BaseModel):
    request_id: str
    query: str
    decision: str
    confidence: float
    query_type: str
    retrieval_strategy: str
    evidence: list[EvidenceItem]
    supporting_sections: list[str]
    why: dict
    trace: dict


class AskResponse(SearchResponse):
    answer: Optional[str] = None
    answer_mode: str
    citation_report: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
