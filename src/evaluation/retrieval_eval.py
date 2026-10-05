"""Per-query retrieval evaluation at SECTION level.

Chunks are mapped to sections; a chunk counts as a hit for gold section G if chunk.section_id == G OR G is an ancestor
of the chunk's section (so gold 'sql-createindex' matches its refsections). Ranked chunks are collapsed to unique
section keys (first occurrence) BEFORE cutting at k, so fixed-size chunking cannot gain by returning many chunks of
one section. Only answerable questions contribute to retrieval metrics.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.evaluation.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from src.evidence.gate import ABSTAIN

KS = (3, 5, 10)
GOLD_TO_ROUTER = {"LOOKUP": "LOOKUP", "HOW_TO": "HOW_TO", "CONCEPTUAL": "CONCEPTUAL",
                  "MULTI_SECTION": "MULTI_SECTION", "OUT_OF_SCOPE": "OUT_OF_SCOPE"}


def load_gold(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def resolve_keys(chunks, gold: set[str]) -> list[str]:
    keys: list[str] = []
    for c in chunks:
        key = next((g for g in [c.section_id, *c.ancestors] if g in gold), c.section_id)
        if key not in keys:
            keys.append(key)
    return keys


def evaluate_query(pipeline, item: dict, strategy: str, chunking: str, ks=KS) -> dict:
    res = pipeline.run(item["question"], strategy, chunking, generate=False)
    gold = set(item.get("gold_sections") or [])
    answerable = item["type"] != "OUT_OF_SCOPE"
    keys = resolve_keys([h.chunk for h in res["_hits"]], gold)
    rec = {
        "id": item["id"], "question": item["question"], "type": item["type"], "difficulty": item.get("difficulty"),
        "answerable": answerable, "gold_sections": sorted(gold), "strategy": strategy, "chunking": chunking,
        "retrieved_sections": keys[:10], "bm25_top_sections": res["_bm25_top"], "dense_top_sections": res["_dense_top"],
        "pred_type": res["query_type"], "route": res["trace"]["retrieval_strategy"], "confidence": res["confidence"],
        "decision": res["decision"],
        "oos_boost": res["_gate"].required_answer - pipeline.thresholds(strategy.upper(), chunking).answer,
        "thresholds_source": res["_gate"].thresholds_source, "timings": res["trace"]["timings"],
        "evidence_components": res["trace"]["evidence_score"],
    }
    if answerable:
        rec["recall"] = {k: recall_at_k(keys, gold, k) for k in ks}
        rec["rr"] = reciprocal_rank(keys, gold, 10)
        rec["ndcg"] = {k: ndcg_at_k(keys, gold, k) for k in ks}
    return rec


def _agg_rows(rows: list[dict], ks) -> dict:
    out = {"n": len(rows)}
    if not rows:
        return out
    for k in ks:
        out[f"recall@{k}"] = mean(r["recall"][k] for r in rows)
        out[f"ndcg@{k}"] = mean(r["ndcg"][k] for r in rows)
    out["mrr"] = mean(r["rr"] for r in rows)
    return out


def aggregate(records: list[dict], ks=KS) -> dict:
    ans = [r for r in records if r["answerable"]]
    agg = {"n_questions": len(records), "n_answerable": len(ans), "overall": _agg_rows(ans, ks), "by_type": {},
           "by_difficulty": {}}
    for t in sorted({r["type"] for r in ans}):
        agg["by_type"][t] = _agg_rows([r for r in ans if r["type"] == t], ks)
    for d in sorted({r["difficulty"] for r in ans if r["difficulty"]}):
        agg["by_difficulty"][d] = _agg_rows([r for r in ans if r["difficulty"] == d], ks)
    stages = defaultdict(list)
    for r in records:
        for k, v in r["timings"].items():
            stages[k].append(v)
    agg["latency_ms_mean"] = {k: round(mean(v), 3) for k, v in stages.items()}
    tot = stages.get("total_ms", [])
    agg["latency_total_ms"] = {"mean": round(mean(tot), 3) if tot else None, "p50": round(percentile(tot, 50), 3) if tot else None,
                               "p95": round(percentile(tot, 95), 3) if tot else None}
    agg["router_accuracy"] = mean(r["pred_type"] == GOLD_TO_ROUTER.get(r["type"], r["type"]) for r in records) if records else None
    return agg
