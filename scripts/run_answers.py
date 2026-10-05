"""Generate answers for a sample (needs Ollama), auto-check citations, export a MANUAL grading sheet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from src.config import Settings, load_tunables
from src.evaluation.answer_eval import export_grading_sheet, generate_answers, summarize_auto, summarize_manual_grades
from src.evaluation.retrieval_eval import load_gold
from src.generation.ollama_client import OllamaClient
from src.pipeline import RAGPipeline


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--strategy", default="ADAPTIVE")
    ap.add_argument("--summarize-grades", help="path to a filled grading CSV")
    a = ap.parse_args()
    if a.summarize_grades:
        print(json.dumps(summarize_manual_grades(a.summarize_grades), indent=2)); return
    llm = OllamaClient(s.ollama_url, s.ollama_model)
    if not llm.available():
        print("Ollama not reachable at", s.ollama_url, "- start it (ollama serve) and pull", s.ollama_model); return
    gold = load_gold(s.gold_dir / f"gold_{a.split}.jsonl")
    rows = generate_answers(RAGPipeline(s, load_tunables(), llm=llm), gold, a.n, a.strategy, s.default_chunking, s.seed)
    out = s.results_dir / "answers"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"answers_{a.split}.json").write_text(json.dumps(rows, indent=2))
    export_grading_sheet(rows, out / f"grading_{a.split}.csv")
    print(json.dumps(summarize_auto(rows), indent=2))
    print("fill in", out / f"grading_{a.split}.csv", "then run with --summarize-grades <csv>")


if __name__ == "__main__":
    main()
