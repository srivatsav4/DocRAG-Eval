"""Collect everything needed to reproduce a run."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.config import CHUNKING_CONFIGS, ROOT, Settings

PACKAGES = ["numpy", "sentence-transformers", "torch", "transformers", "fastapi", "pydantic", "beautifulsoup4",
            "matplotlib", "psycopg", "pgvector", "httpx", "streamlit", "pytest"]


def _ver(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _db_versions(url: str) -> dict:
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=2) as c:
            pg = c.execute("SHOW server_version").fetchone()[0]
            v = c.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
            return {"postgres_server": pg, "pgvector": v[0] if v else None}
    except Exception:  # noqa: BLE001
        return {"postgres_server": "unavailable", "pgvector": "unavailable"}


def collect_environment(settings: Settings, tun: dict, extra: dict | None = None) -> dict:
    manifest = settings.processed_dir / "manifest.json"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    th = Path(settings.thresholds_file)
    env = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version, "platform": platform.platform(), "git_commit": commit or None,
        "packages": {p: _ver(p) for p in PACKAGES},
        "models": {"embedding": settings.embedding_model, "embedder_backend": settings.embedder_backend,
                   "reranker": settings.reranker_model, "reranker_backend": settings.reranker_backend,
                   "llm": settings.ollama_model if settings.llm_enabled else None},
        "dense_backend": settings.dense_backend, "database": _db_versions(settings.database_url),
        "pg_docs_version": settings.pg_docs_version,
        "corpus_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        "chunking_configs": CHUNKING_CONFIGS, "seed": settings.seed, "tunables": tun,
        "thresholds_file_sha256": hashlib.sha256(th.read_bytes()).hexdigest() if th.exists() else None,
    }
    env.update(extra or {})
    return env
