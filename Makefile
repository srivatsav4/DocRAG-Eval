PY ?= python
.PHONY: setup db download ingest ingest-db gold validate-gold test eval-smoke eval-tune eval-retrieval eval-test eval-ablation eval-answers api ui demo clean

setup:            ## create venv + install deps
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env || true

db:               ## start PostgreSQL + pgvector
	docker compose up -d db

download:         ## crawl official PostgreSQL docs (needs network)
	$(PY) scripts/download_docs.py

ingest: download  ## download + parse + chunk (all strategies)
	$(PY) scripts/ingest.py

ingest-db:        ## also embed and load into pgvector (needs `make db`)
	$(PY) scripts/ingest.py --to-db

gold:             ## regenerate deterministic DEV/TEST split
	$(PY) scripts/build_gold_split.py

validate-gold:    ## check gold section IDs exist in the ingested corpus
	$(PY) scripts/validate_gold.py

test:
	$(PY) -m pytest

eval-smoke:       ## quick sanity run (NOT a result)
	$(PY) scripts/run_eval.py --smoke

eval-tune:        ## DEV: full run + tune and freeze abstention thresholds
	$(PY) scripts/run_eval.py --full --split dev --tune

eval-retrieval:   ## DEV full retrieval/abstention evaluation
	$(PY) scripts/run_eval.py --full --split dev

eval-test:        ## TEST with thresholds frozen on DEV (run once, after tuning)
	$(PY) scripts/run_eval.py --full --split test

eval-ablation:
	$(PY) scripts/run_ablation.py

eval-answers:     ## needs Ollama; writes a manual-grading CSV
	$(PY) scripts/run_answers.py --split dev --n 20

api:
	uvicorn src.api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

demo:
	$(PY) scripts/demo.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache
