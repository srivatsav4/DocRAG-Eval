"""Runs the (strategy x chunking x k) grid, abstention eval, bootstrap comparison, plots, failure candidates."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from src.config import CHUNKING_CONFIGS, STRATEGY_NAMES, Settings, load_tunables
from src.evaluation.abstention_eval import abstention_metrics, sweep_thresholds, tune_threshold
from src.evaluation.failure_analysis import candidate_failures
from src.evaluation.metrics import paired_bootstrap
from src.evaluation.plots import make_all_plots
from src.evaluation.retrieval_eval import KS, aggregate, evaluate_query, load_gold
from src.evidence.gate import file_sha256, threshold_key
from src.ingest.loader import write_jsonl
from src.pipeline import RAGPipeline
from src.utils.logging import get_logger
from src.utils.repro import collect_environment

log = get_logger("experiment")


def smoke_subset(gold: list[dict], n: int) -> list[dict]:
    """Round-robin over question types so a tiny run still includes every type (incl. out-of-scope)."""
    by: dict[str, list[dict]] = {}
    for g in gold:
        by.setdefault(g["type"], []).append(g)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        for t in sorted(by):
            if by[t] and len(out) < n:
                out.append(by[t].pop(0))
        i += 1
    return out


def check_test_allowed(settings: Settings, dev_path: Path, allow_uncalibrated: bool) -> None:
    p = Path(settings.thresholds_file)
    if allow_uncalibrated:
        return
    if not p.exists():
        raise RuntimeError("Refusing to evaluate TEST: no frozen thresholds. Run `python scripts/run_eval.py --full --split dev --tune` "
                           "first (or pass --allow-uncalibrated to use default thresholds).")
    data = json.loads(p.read_text())
    if data.get("frozen_on_split") != "dev" or data.get("gold_dev_sha256") != file_sha256(dev_path):
        raise RuntimeError("Frozen thresholds were not tuned on the current gold_dev.jsonl; re-run tuning on DEV.")


def freeze_thresholds(settings: Settings, dev_path: Path, entries: dict, margin: float) -> Path:
    p = Path(settings.thresholds_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"frozen_on_split": "dev", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "gold_dev_sha256": file_sha256(dev_path), "objective": "balanced accuracy (abstain on OOS vs answer)",
                             "margin_answer_minus_limited": margin, "entries": entries}, indent=2))
    return p


def run_grid(settings: Settings, split: str = "dev", mode: str = "smoke", strategies: list[str] | None = None,
             chunkings: list[str] | None = None, limit: int | None = None, out_dir: Path | None = None, tune: bool = False,
             allow_uncalibrated: bool = False, pipeline: RAGPipeline | None = None, gold_path: Path | None = None,
             margin: float = 0.15) -> dict:
    tun = load_tunables()
    strategies = strategies or STRATEGY_NAMES
    chunkings = chunkings or ([settings.default_chunking] if mode == "smoke" else list(CHUNKING_CONFIGS))
    gold_path = Path(gold_path or settings.gold_dir / f"gold_{split}.jsonl")
    dev_path = Path(settings.gold_dir / "gold_dev.jsonl") if gold_path.name != "gold_dev.jsonl" else gold_path
    if split == "test" and gold_path.parent == settings.gold_dir:
        check_test_allowed(settings, dev_path, allow_uncalibrated)
    if tune and split != "dev":
        raise ValueError("--tune is only allowed on the DEV split")
    gold = load_gold(gold_path)
    n_smoke = limit or (12 if mode == "smoke" else None)
    if mode == "smoke":
        gold = smoke_subset(gold, n_smoke)
    elif limit:
        gold = gold[:limit]
    pipe = pipeline or RAGPipeline(settings, tun)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{split}-{mode}"
    out = Path(out_dir or settings.results_dir / run_id)
    out.mkdir(parents=True, exist_ok=True)

    all_records, rows, by_key = [], [], {}
    for ch in chunkings:
        for st in strategies:
            pipe.run(gold[0]["question"], st, ch)  # warm-up (model load / index build) - not recorded
            recs = [evaluate_query(pipe, it, st, ch) for it in gold]
            by_key[(st, ch)] = recs
            all_records += recs
            agg = aggregate(recs)
            agg.update({"strategy": st, "chunking": ch, "abstention": abstention_metrics(recs)})
            rows.append(agg)
            log.info("%s | %s | recall@5=%s", st, ch, agg["overall"].get("recall@5"))

    comparisons = {}
    for ch in chunkings:
        for fixed in ("HYBRID_RRF", "HYBRID_RERANK"):
            a, b = by_key.get(("ADAPTIVE", ch)), by_key.get((fixed, ch))
            if not a or not b:
                continue
            ans = [(x, y) for x, y in zip(a, b) if x["answerable"]]
            comparisons[f"ADAPTIVE_vs_{fixed}|{ch}"] = {
                "recall@5": paired_bootstrap([x["recall"][5] for x, _ in ans], [y["recall"][5] for _, y in ans], seed=settings.seed),
                "mrr": paired_bootstrap([x["rr"] for x, _ in ans], [y["rr"] for _, y in ans], seed=settings.seed),
                "ndcg@5": paired_bootstrap([x["ndcg"][5] for x, _ in ans], [y["ndcg"][5] for _, y in ans], seed=settings.seed)}

    sweep = None
    adaptive_first = next((k for k in by_key if k[0] == "ADAPTIVE"), None)
    if adaptive_first:
        sweep = sweep_thresholds(by_key[adaptive_first])
    tuned = None
    if tune:
        entries = {}
        for (st, ch), recs in by_key.items():
            try:
                entries[threshold_key(st, ch)] = {**tune_threshold(recs, margin), "oos_boost": tun["thresholds"]["oos_boost"]}
            except ValueError as e:
                log.warning("cannot tune %s|%s: %s", st, ch, e)
        tuned = str(freeze_thresholds(settings, dev_path, entries, margin))
        pipe._thr.clear()

    write_jsonl(out / "per_query.jsonl", ({k: v for k, v in r.items()} for r in all_records))
    write_jsonl(out / "failure_candidates.jsonl", candidate_failures([r for r in all_records if r["strategy"] == "ADAPTIVE"]))
    plots = make_all_plots(rows, sweep, out / "plots")
    summary = {"run_id": run_id, "split": split, "mode": mode, "n_questions": len(gold), "ks": list(KS),
               "rows": rows, "adaptive_vs_fixed_bootstrap": comparisons, "threshold_sweep_adaptive": sweep,
               "thresholds_frozen_to": tuned, "plots": plots,
               "WARNING": "Smoke run: tiny sample, NOT a result." if mode == "smoke" else None,
               "embedder_backend": settings.embedder_backend, "reranker_backend": settings.reranker_backend}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "reproducibility.json").write_text(json.dumps(collect_environment(settings, tun, {"split": split, "mode": mode,
                                                         "gold_sha256": file_sha256(gold_path)}), indent=2, default=str))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "chunking", "n_answerable"] + [f"recall@{k}" for k in KS] + ["mrr"] + [f"ndcg@{k}" for k in KS] +
                   ["latency_mean_ms", "latency_p95_ms", "abstention_recall", "wrongly_refused", "coverage"])
        for r in rows:
            o, a = r["overall"], r["abstention"]
            w.writerow([r["strategy"], r["chunking"], r["n_answerable"]] + [o.get(f"recall@{k}") for k in KS] + [o.get("mrr")] +
                       [o.get(f"ndcg@{k}") for k in KS] + [r["latency_total_ms"]["mean"], r["latency_total_ms"]["p95"],
                        a["abstention_recall"], a["wrongly_refused_answerable"], a["coverage"]])
    summary["out_dir"] = str(out)
    return summary
