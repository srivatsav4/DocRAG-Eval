"""Dense retrieval: embedders + exact cosine search (numpy) or pgvector (PostgreSQL)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import Settings
from src.ingest.chunker import Chunk
from src.ingest.loader import corpus_hash
from src.retrieval.bm25 import tokenize
from src.utils.logging import get_logger

log = get_logger("dense")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic signed feature-hashing bag-of-words embedder. TEST/SMOKE ONLY: not semantic."""

    def __init__(self, dim: int = 256):
        self.dim, self.name = dim, f"hash-{dim} (test-only, not semantic)"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=len(texts) > 200), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0], dtype=np.float32)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder_backend == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)


class NumpyDenseIndex:
    """Exact cosine search (vectors are L2-normalised so cosine = dot product)."""
    backend = "numpy"

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.matrix) == 0:
            return []
        sims = self.matrix @ qvec
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((top, -sims[top]))]
        return [(int(i), float(sims[i])) for i in top]


class PgVectorIndex:
    """Exact cosine search inside PostgreSQL via pgvector (`<=>` cosine distance)."""
    backend = "pgvector"

    def __init__(self, database_url: str, strategy: str, dim: int):
        if not re.fullmatch(r"[a-z0-9_]+", strategy):
            raise ValueError("invalid table suffix")
        self.url, self.table, self.dim = database_url, f"chunks_{strategy}_{dim}", dim

    def _conn(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.url, connect_timeout=3, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def count(self) -> int:
        with self._conn() as c:
            try:
                return c.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
            except Exception:
                return -1

    def build(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        with self._conn() as c:
            c.execute(f"DROP TABLE IF EXISTS {self.table}")
            c.execute(f"""CREATE TABLE {self.table} (idx int PRIMARY KEY, chunk_id text UNIQUE, section_id text,
                title text, breadcrumb text, url text, content text, content_hash text, pg_version text,
                ingested_at text, embedding vector({self.dim}))""")
            with c.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(i, ch.chunk_id, ch.section_id, ch.title, ch.breadcrumb, ch.url, ch.text, ch.content_hash,
                      ch.pg_version, ch.ingested_at, matrix[i]) for i, ch in enumerate(chunks)])

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._conn() as c:
            rows = c.execute(f"SELECT idx, 1 - (embedding <=> %s) FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
                             (qvec, qvec, k)).fetchall()
        return [(int(i), float(s)) for i, s in rows]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def embed_chunks(chunks: list[Chunk], embedder: Embedder, settings: Settings, chunking: str) -> np.ndarray:
    """Embed (with an on-disk cache keyed by chunking + model + corpus hash)."""
    cache = Path(settings.processed_dir) / f"emb_{chunking}_{_slug(embedder.name)}_{corpus_hash(chunks)[:10]}.npy"
    if cache.exists():
        return np.load(cache)
    log.info("embedding %d chunks with %s", len(chunks), embedder.name)
    mat = embedder.encode_documents([c.index_text for c in chunks])
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, mat)
    except OSError:
        pass
    return mat


def build_dense_index(chunks: list[Chunk], embedder: Embedder, settings: Settings, chunking: str):
    mat = embed_chunks(chunks, embedder, settings, chunking)
    if settings.dense_backend in ("auto", "pgvector"):
        try:
            idx = PgVectorIndex(settings.database_url, chunking, embedder.dim)
            if idx.count() != len(chunks):
                log.info("loading %d vectors into pgvector table %s", len(chunks), idx.table)
                idx.build(chunks, mat)
            return idx
        except Exception as e:  # noqa: BLE001
            if settings.dense_backend == "pgvector":
                raise
            log.warning("pgvector unavailable (%s); falling back to in-memory numpy index", type(e).__name__)
    return NumpyDenseIndex(mat)
