"""Answer-level evaluation: automatic citation heuristics + a MANUAL grading workflow.

Nothing here is a ground-truth judgement of factual correctness. Human grades (CSV) are the only correctness signal.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean


def generate_answers(pipeline, gold: list[dict], n: int, strategy: str, chunking: str, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    sample = rng.sample(gold, min(n, len(gold)))
    rows = []
    for it in sample:
        res = pipeline.run(it["question"], strategy, chunking, generate=True)
        rows.append({"id": it["id"], "question": it["question"], "type": it["type"], "gold_sections": it.get("gold_sections", []),
                     "decision": res["decision"], "confidence": res["confidence"], "answer_mode": res["answer_mode"],
                     "answer": res["answer"], "citation_report": res["citation_report"],
                     "evidence": [{k: e[k] for k in ("label", "section_id", "snippet")} for e in res["evidence"]]})
    return rows


def summarize_auto(rows: list[dict]) -> dict:
    gen = [r for r in rows if r["answer_mode"] == "generated" and r["citation_report"]]
    if not gen:
        return {"n_generated": 0, "note": "NOT YET MEASURED (no generated answers; is Ollama running?)"}
    return {"n_generated": len(gen),
            "mean_citation_coverage": mean(r["citation_report"]["citation_coverage"] for r in gen),
            "mean_citation_correctness_heuristic": mean(r["citation_report"]["citation_correctness"] for r in gen),
            "answers_with_unsupported_claims": sum(bool(r["citation_report"]["unsupported_claims"]) for r in gen),
            "caveat": "lexical-overlap heuristic, not entailment"}


def export_grading_sheet(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "decision", "answer", "evidence_sections", "correct(0/1/2)", "faithful(0/1)",
                    "citations_ok(0/1)", "notes"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["decision"], r["answer"],
                        ";".join(e["section_id"] for e in r["evidence"]), "", "", "", ""])


def summarize_manual_grades(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["correct(0/1/2)"].strip() != ""]
    if not rows:
        return {"n_graded": 0, "note": "NOT YET MEASURED (grading sheet empty)"}
    g = lambda k: [float(r[k]) for r in rows if r[k].strip() != ""]  # noqa: E731
    return {"n_graded": len(rows), "mean_correct_0_2": mean(g("correct(0/1/2)")),
            "faithful_rate": mean(g("faithful(0/1)")) if g("faithful(0/1)") else None,
            "citations_ok_rate": mean(g("citations_ok(0/1)")) if g("citations_ok(0/1)") else None}
