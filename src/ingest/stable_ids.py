"""Deterministic identifiers. Same input text + structure => same ID on every machine."""
from __future__ import annotations

import hashlib
import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    """SHA-256 of whitespace-normalised text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def normalize_section_id(raw: str) -> str:
    """PostgreSQL docs use upper-case anchors in places (INDEXES-PARTIAL); we compare lower-case."""
    return raw.strip().lstrip("#").lower()


def make_chunk_id(pg_version: str, section_id: str, strategy: str, ordinal: int, chash: str) -> str:
    """Human-readable + collision-resistant: pg16:indexes-partial:section_based:000:ab12cd34"""
    return f"pg{pg_version}:{section_id}:{strategy}:{ordinal:03d}:{chash[:8]}"
