"""CLI demo of the 'Why did I answer?' contrast. Run: python scripts/demo.py [--no-llm]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.config import Settings, load_tunables
from src.generation.ollama_client import OllamaClient
from src.pipeline import RAGPipeline

QUESTIONS = ["How do I create a partial index in PostgreSQL?",
             "How do I configure PostgreSQL to control an unrelated technology that is not documented here?"]


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--question", default=None)
    a = ap.parse_args()
    llm = None if a.no_llm or not s.llm_enabled else OllamaClient(s.ollama_url, s.ollama_model)
    pipe = RAGPipeline(s, load_tunables(), llm=llm)
    for q in ([a.question] if a.question else QUESTIONS):
        r = pipe.run(q, "ADAPTIVE", generate=llm is not None)
        w = r["why"]
        print("=" * 72, f"\nQuestion: {q}\n")
        print(f"Query type  {w['query_type']}\nStrategy    {w['selected_strategy']}\nConfidence  {r['confidence']:.2f}  (required {w['required_confidence']:.2f})\nDecision    {r['decision']}\n")
        print("Answer:", r["answer"] or f"({r['answer_mode']})", "\n")
        print("WHY I ANSWERED" if r["decision"] != "ABSTAIN" else "WHY I REFUSED")
        for c in w["checks"]:
            print(("  ✓ " if c["passed"] else "  ✗ ") + f"{c['label']}: {c['detail']}")
        print("  ->", w["summary"])
        for e in r["evidence"][:3]:
            print(f"  [{e['label']}] {e['breadcrumb']}  ({e['score']:.2f})  {e['url']}")
        print("timings(ms):", r["trace"]["timings"], "\n")


if __name__ == "__main__":
    main()
