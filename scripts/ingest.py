"""Parse raw HTML -> sections + chunks (all chunking strategies). Optionally embed + load pgvector (--to-db)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.config import CHUNKING_CONFIGS, Settings
from src.ingest.loader import ingest_directory, load_chunks


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=list(CHUNKING_CONFIGS))
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--to-db", action="store_true", help="embed chunks and load them into PostgreSQL/pgvector")
    a = ap.parse_args()
    m = ingest_directory(s, a.strategies, a.max_pages)
    print({k: v for k, v in m.items() if k != "chunking"}, {k: v["n_chunks"] for k, v in m["chunking"].items()})
    if a.to_db:
        from src.retrieval.dense import PgVectorIndex, embed_chunks, get_embedder

        emb = get_embedder(s)
        for strat in a.strategies:
            ch = load_chunks(s, strat)
            PgVectorIndex(s.database_url, strat, emb.dim).build(ch, embed_chunks(ch, emb, s, strat))
            print("loaded", strat, len(ch), "chunks into pgvector")


if __name__ == "__main__":
    main()
