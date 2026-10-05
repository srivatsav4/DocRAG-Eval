# DocRAG-Eval

### Evidence-Adaptive Retrieval for Technical Documentation

> An evaluation-first, evidence-adaptive RAG system that dynamically selects retrieval strategies and abstains when documentation evidence is insufficient.

## Why this project exists
Most RAG demos are `question → embed → top-k → LLM`. They never say whether retrieval was any good, and they answer even when the documentation contains nothing relevant. DocRAG-Eval treats retrieval quality, evidence sufficiency and abstention as **measurable** properties, over one fixed version of the official PostgreSQL documentation.

## Research question
**Does query-adaptive retrieval improve retrieval quality compared with a fixed hybrid pipeline?** (`ADAPTIVE` vs `HYBRID_RRF` and `HYBRID_RERANK`, paired bootstrap CIs, per chunking strategy.)

## Key idea
```
Question → Query analysis → Query type → Strategy selection → Candidate retrieval → RRF fusion → Reranking
        → Evidence scoring → Evidence gate → ANSWER | ANSWER_WITH_LIMITATION | ABSTAIN → Citation check → Answer + Evidence + Trace
```
Every response carries a retrieval trace and a "Why did I answer / refuse?" block.

## Architecture
Clean modular monolith (no LangChain/LlamaIndex; BM25, RRF, scorer, gate, checker are written from scratch):
`ingest/` (BeautifulSoup parser, chunkers, stable IDs) · `retrieval/` (BM25, dense, RRF, reranker, router) · `evidence/` (scorer, gate, citation checker) · `generation/` (Ollama, optional) · `evaluation/` (metrics, runners, plots) · `api/` (FastAPI) · `ui/` (Streamlit). The LLM is **optional**: retrieval, evidence scoring, gating and all evaluation run without it.

## Adaptive retrieval
Deterministic rule-based classifier (`src/retrieval/router.py`); every decision logs the rules that fired. Routes are **hypotheses to be tested on DEV**, not proven optima.

| Query type | Strategy (RRF weights bm25:dense, candidates, rerank) |
|---|---|
| LOOKUP | lexical-weighted hybrid 2:1, 30, no rerank |
| HOW_TO | balanced hybrid 1:1, 50, cross-encoder |
| CONCEPTUAL | dense-weighted hybrid 1:2, 50, cross-encoder |
| MULTI_SECTION | wide hybrid 1:1, 100, cross-encoder, final_k=8 |
| OUT_OF_SCOPE | cautious hybrid 1:1 + rerank; gate threshold raised by `oos_boost` |

Fixed baselines: `BM25`, `DENSE`, `HYBRID_RRF` (1:1, 50), `HYBRID_RERANK`. The classifier is only a hint: PostgreSQL-flavoured but undocumented questions (e.g. the "unrelated technology" demo) are caught by the evidence gate, not the classifier.

## Evidence scoring
`confidence = Σ wᵢcᵢ / Σ wᵢ` over available components (`src/evidence/scorer.py`, weights in `configs/default.json`):
`relevance` = sigmoid(best reranker logit) · `semantic` = rescaled best dense cosine · `coverage` = IDF-weighted share of informative query terms found in the evidence (unseen terms get max IDF) · `agreement` = overlap of BM25 and dense top-3 sections · `support` = enough distinct sections (2 for MULTI_SECTION).
**Limitations:** hand-designed heuristic, *not* a calibrated probability; the semantic rescaling is embedding-model specific; weights are uncalibrated defaults.

## Abstention
Decisions: `ANSWER`, `ANSWER_WITH_LIMITATION`, `ABSTAIN`. Defaults (0.65 / 0.45, +0.10 for OUT_OF_SCOPE) are **uncalibrated**. `run_eval.py --split dev --tune` picks the abstain threshold maximising balanced accuracy on DEV and freezes it to `configs/thresholds.json` (with the DEV file hash). TEST runs refuse to start unless thresholds were frozen on the current DEV file.

## Citation verification
The LLM must cite `[S#]`. `check_citations` reports citation coverage, heuristic citation correctness, unsupported claims and invalid citations using lexical-overlap support (numbers/identifiers must match). It is **not** an entailment/factuality verifier: a pass does not prove a claim is true.

## Dataset
Official PostgreSQL docs, one fixed major version (default 16, `PG_DOCS_VERSION`), crawled politely by `scripts/download_docs.py` (not redistributed). Each chunk stores version, source URL, section ID, title, breadcrumb, ingestion timestamp, content hash and a deterministic ID (`pg16:indexes-partial:section_based:000:ab12cd34`). Chunking: `section_based`, `fixed_300_overlap`, `fixed_500_overlap` (fixed windows span the page and are assigned to the majority-owner section).

## Evaluation methodology
* Gold: 80 questions (25 LOOKUP, 25 HOW_TO, 15 MULTI_SECTION, 15 OUT_OF_SCOPE), easy/medium/hard; split 50 DEV / 30 TEST (`gold_dev.jsonl`, `gold_test.jsonl`, stratified, seed 42).
* **Gold labels are an unverified seed**: section IDs were written from knowledge of the PG16 docs structure and must be checked with `python scripts/validate_gold.py` and reviewed manually before any number is trusted (see `data/gold/README.md`).
* Metrics (section level; ranked chunks collapsed to unique sections, ancestors count as hits): Recall@3/5/10, MRR@10, nDCG@3/5/10; per-stage latency; abstention recall/precision, wrongly refused, coverage, selective accuracy; automatic citation heuristics; manual grading sheet for a sample.
* Paired bootstrap (2000 resamples, seeded) for ADAPTIVE vs fixed hybrids. With ≤30 TEST questions expect wide intervals — report them.
* The router rules are hand-written; if you edit them after looking at DEV errors, say so. Never touch TEST labels or rules after seeing TEST results.

## Experimental setup
Embedding `BAAI/bge-small-en-v1.5`; reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`; LLM `qwen2.5:3b` via Ollama (optional); exact cosine search (numpy or pgvector — same results by construction, pgvector path is the production backend). Each run writes `reproducibility.json` (Python, package versions, models, PG/pgvector versions if reachable, docs version, chunking configs, thresholds hash, seed, git commit).

## Results
**NOT YET MEASURED.** Results will be populated after running:
```
make eval-tune        # DEV, freezes thresholds
make eval-retrieval   # DEV full grid
make eval-test        # TEST, once
```
| Strategy | Chunking | Recall@5 | MRR | nDCG@5 | Mean latency |
|---|---|---|---|---|---|
| BM25 / Dense / Hybrid RRF / Hybrid+Rerank / Adaptive | × 3 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

## Accuracy vs latency
NOT YET MEASURED — `results/<run>/plots/accuracy_vs_latency.png`, `latency_breakdown.png`, `recall_comparison.png`, `mrr_comparison.png`, `ndcg_comparison.png`, `adaptive_vs_fixed.png`, `query_type_performance.png`.

## Abstention results
NOT YET MEASURED — `plots/abstention_sweep.png`, `summary.json → rows[].abstention`.

## Failure analysis
See `docs/failure_analysis.md`. Candidates are auto-extracted to `results/<run>/failure_candidates.jsonl`; no failures are pre-labelled. NOT YET MEASURED.

## API
`GET /health` · `POST /search` · `POST /ask` · `POST /ingest` · `POST /evaluate` (TEST needs `confirm_frozen_thresholds`) · `GET /metrics` · `GET /trace/{request_id}`. Pydantic schemas in `src/api/schemas.py`; errors: 422 validation, 400 bad value, 503 corpus not ingested, 404 unknown trace. Swagger at `/docs`.

## UI
Streamlit "Evidence Inspector" (`make ui`): decision, confidence vs required, query type, strategy, why-I-answered checklist, citations/evidence, trace JSON, latency chart.

## Project structure
`src/{ingest,retrieval,evidence,generation,evaluation,api,utils}`, `scripts/`, `ui/app.py`, `tests/`, `configs/` (`default.json`, frozen `thresholds.json` after tuning), `data/{raw_html,processed,gold}`, `results/`, `docs/`.

## Installation
Python 3.11+. `make setup` (venv + `requirements.txt`; CPU torch recommended, see comments there).

## Quick start
```bash
cp .env.example .env            # set POSTGRES_PASSWORD
make db                         # PostgreSQL + pgvector (Docker)
make ingest                     # crawl PG docs + parse + chunk (needs network)
make validate-gold              # check gold IDs against the corpus
make ingest-db                  # embed + load pgvector (optional; numpy fallback otherwise)
make api                        # http://localhost:8000/docs
make ui                         # Evidence Inspector
make demo                       # CLI contrast: answered vs refused
```

## Docker
`docker compose up -d db` runs PostgreSQL+pgvector. Optional API container: `docker compose --profile app up` (reaches host Ollama through `host.docker.internal`). Ollama is **not** containerised. The compose file was not run in the authoring environment (no Docker there).

## Ollama setup
Install Ollama on the host, `ollama pull qwen2.5:3b`, `ollama serve`. Without it, `/ask` returns evidence + decision with `answer_mode: retrieval_only`; set `LLM_ENABLED=false` to skip probing.

## Running evaluation
`make eval-smoke` (≈12 questions, 1 chunking — plumbing check, not a result) · `python scripts/run_eval.py --full` · `python scripts/run_ablation.py` (all chunkings) · `make eval-answers` then grade the CSV and run `python scripts/run_answers.py --summarize-grades <csv>`. `--embedder hash --reranker lexical` runs offline but is **non-semantic**; never report those numbers.

## Testing
`pytest` (71 tests; fake embedder/reranker/LLM, no network, DB, models or Ollama needed).

## Reproducibility
Seeded split and bootstrap, deterministic IDs, cached embeddings keyed by corpus hash, `reproducibility.json` per run, pinned PG docs version, frozen thresholds with DEV hash.

## Limitations
* Gold labels are an unvalidated seed with single annotator; multiple pages may legitimately answer a question.
* Rule-based router and keyword lexicon are crude; evidence score is uncalibrated; citation check is lexical.
* Small TEST set (30) → wide confidence intervals; hash/lexical fallbacks exist only for offline smoke tests.
* pgvector, the real models, Ollama, Docker and the doc crawler were **not exercised in the authoring sandbox** (no network); they are implemented but unverified end-to-end.

## Future work
Learned query classifier, per-type thresholds, query expansion, section-hierarchy-aware retrieval, NLI-based claim verification, multi-annotator gold.

## Interview discussion
* *What did you build?* From-scratch BM25/RRF, router, evidence scorer + gate, citation checker, and an ablation harness with frozen-threshold protocol.
* *How do you know adaptive is better?* `ADAPTIVE_vs_HYBRID_*` paired bootstrap in `summary.json` — and if it isn't, that is a reportable finding.
* *What if RAG is wrong?* The gate abstains; `failure_candidates.jsonl` + `docs/failure_analysis.md` capture failures.
* *How were parameters chosen?* Thresholds tuned on DEV only, frozen, then TEST; weights/routes are stated hypotheses.
