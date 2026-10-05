"""Chunking strategies: section_based | fixed_300_overlap | fixed_500_overlap (see config.CHUNKING_CONFIGS).

section_based : one chunk per section (long sections are split in overlapping word windows).
fixed_N       : fixed word windows over the whole PAGE (ignores section boundaries). Each chunk is assigned the
                section owning the majority of its words so it can still be scored against gold section IDs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.config import CHUNKING_CONFIGS
from src.ingest.parser import Section
from src.ingest.stable_ids import content_hash, make_chunk_id


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    ancestors: list[str]
    title: str
    breadcrumb: str
    url: str
    text: str
    index_text: str
    content_hash: str
    pg_version: str
    ingested_at: str
    strategy: str
    ordinal: int
    spanned_sections: list[str] = field(default_factory=list)
    n_words: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    out, start = [], 0
    while start < n:
        end = min(n, start + size)
        out.append((start, end))
        if end == n:
            break
        start = end - overlap
    return out


def _mk(sec: Section, words: list[str], strategy: str, ordinal: int, version: str, ts: str,
        spanned: list[str], include_title: bool) -> Chunk:
    text = " ".join(words)
    h = content_hash(text)
    index_text = f"{sec.breadcrumb}\n{text}" if include_title else text
    return Chunk(make_chunk_id(version, sec.section_id, strategy, ordinal, h), sec.section_id, sec.ancestors,
                 sec.title, sec.breadcrumb, sec.url, text, index_text, h, version, ts, strategy, ordinal,
                 spanned, len(words))


def chunk_sections(sections: list[Section], strategy: str, pg_version: str = "",
                   ingested_at: str | None = None, include_title: bool = True) -> list[Chunk]:
    if strategy not in CHUNKING_CONFIGS:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {list(CHUNKING_CONFIGS)}")
    cfg = CHUNKING_CONFIGS[strategy]
    ts = ingested_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    chunks: list[Chunk] = []

    if cfg["kind"] == "section":
        for sec in sections:
            words = sec.text.split()
            if not words:
                continue
            for i, (a, b) in enumerate(_windows(len(words), cfg["max_words"], cfg["overlap_words"])):
                chunks.append(_mk(sec, words[a:b], strategy, i, pg_version, ts, [sec.section_id], include_title))
        return chunks

    by_page: dict[str, list[Section]] = {}
    for sec in sections:
        by_page.setdefault(sec.page, []).append(sec)
    for secs in by_page.values():
        words: list[str] = []
        owner: list[int] = []
        for si, sec in enumerate(secs):
            w = sec.text.split()
            words += w
            owner += [si] * len(w)
        for i, (a, b) in enumerate(_windows(len(words), cfg["size"], cfg["overlap"])):
            cnt = Counter(owner[a:b])
            main = secs[cnt.most_common(1)[0][0]]
            spanned = list(dict.fromkeys(secs[o].section_id for o in owner[a:b]))
            chunks.append(_mk(main, words[a:b], strategy, i, pg_version, ts, spanned, include_title))
    return chunks
