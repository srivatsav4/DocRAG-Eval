"""Persist parsed sections/chunks to JSONL (+ manifest). Vectors are handled in retrieval.dense."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import CHUNKING_CONFIGS, Settings
from src.ingest.chunker import Chunk, chunk_sections
from src.ingest.parser import Section, parse_html
from src.utils.logging import get_logger

log = get_logger("ingest")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunks_path(settings: Settings, strategy: str) -> Path:
    return settings.processed_dir / f"chunks_{strategy}.jsonl"


def load_chunks(settings: Settings, strategy: str) -> list[Chunk]:
    p = chunks_path(settings, strategy)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `make ingest` (or POST /ingest) first.")
    return [Chunk(**r) for r in read_jsonl(p)]


def load_sections(settings: Settings) -> list[dict]:
    return read_jsonl(settings.processed_dir / "sections.jsonl")


def corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.content_hash.encode())
    return h.hexdigest()


def ingest_directory(settings: Settings, strategies: list[str] | None = None, max_pages: int | None = None,
                     base_url: str | None = None) -> dict:
    """raw_html/*.html -> sections.jsonl + chunks_<strategy>.jsonl + manifest.json"""
    files = sorted(settings.raw_html_dir.glob("*.html"))
    if max_pages:
        files = files[:max_pages]
    if not files:
        raise FileNotFoundError(f"No .html files in {settings.raw_html_dir}. Run scripts/download_docs.py first.")
    base = base_url or f"{settings.pg_docs_base_url}/{settings.pg_docs_version}"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sections: list[Section] = []
    for f in files:
        sections += parse_html(f.read_text(encoding="utf-8", errors="ignore"), f"{base}/{f.name}",
                               settings.pg_docs_version)
    seen, uniq = set(), []
    for s in sections:  # de-duplicate section ids across pages (keep first)
        if s.section_id not in seen:
            seen.add(s.section_id)
            uniq.append(s)
    sections = uniq
    write_jsonl(settings.processed_dir / "sections.jsonl", (s.__dict__ for s in sections))
    manifest = {"pg_docs_version": settings.pg_docs_version, "source_url": base, "n_pages": len(files),
                "n_sections": len(sections), "ingested_at": ts, "chunking": {}}
    for strat in strategies or list(CHUNKING_CONFIGS):
        chunks = chunk_sections(sections, strat, settings.pg_docs_version, ts)
        write_jsonl(chunks_path(settings, strat), (c.to_dict() for c in chunks))
        manifest["chunking"][strat] = {"n_chunks": len(chunks), "config": CHUNKING_CONFIGS[strat],
                                       "corpus_hash": corpus_hash(chunks)}
        log.info("%s: %d chunks", strat, len(chunks))
    (settings.processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
