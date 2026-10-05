"""Retrieval/abstention evaluation. --smoke = tiny sanity run; --full = complete experiment.

Workflow: (1) --full --split dev --tune   -> tunes + FREEZES thresholds on DEV only
          (2) --full --split test         -> evaluates TEST with frozen thresholds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from src.config import CHUNKING_CONFIGS, STRATEGY_NAMES, Settings
from src.evaluation.experiment_runner import run_grid


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--strategies", nargs="*", choices=STRATEGY_NAMES, default=None)
    ap.add_argument("--chunking", nargs="*", choices=list(CHUNKING_CONFIGS), default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tune", action="store_true", help="DEV only: tune + freeze abstention thresholds")
    ap.add_argument("--allow-uncalibrated", action="store_true", help="allow TEST with default thresholds (flagged)")
    ap.add_argument("--embedder", choices=["sentence-transformers", "hash"], default=None, help="'hash' = non-semantic, smoke only")
    ap.add_argument("--reranker", choices=["cross-encoder", "lexical"], default=None)
    a = ap.parse_args()
    if a.embedder:
        s.embedder_backend = a.embedder
    if a.reranker:
        s.reranker_backend = a.reranker
    mode = "full" if a.full else "smoke"
    summary = run_grid(s, a.split, mode, a.strategies, a.chunking, a.limit, tune=a.tune, allow_uncalibrated=a.allow_uncalibrated)
    print(f"\nresults -> {summary['out_dir']}")
    for r in summary["rows"]:
        o = r["overall"]
        print(f"{r['strategy']:<14}{r['chunking']:<20}R@5={o.get('recall@5', float('nan')):.3f} MRR={o.get('mrr', float('nan')):.3f} "
              f"nDCG@5={o.get('ndcg@5', float('nan')):.3f}  lat={r['latency_total_ms']['mean']}ms")
    if mode == "smoke":
        print("\n(smoke run: tiny sample - NOT a result)")


if __name__ == "__main__":
    main()
