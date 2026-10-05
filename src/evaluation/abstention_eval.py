"""Abstention metrics. Positive class = 'should abstain' (out-of-scope question).

  abstention_recall    = correct OOS refusals / all OOS questions
  abstention_precision = correct OOS refusals / all refusals
  wrongly_refused      = answerable questions that were refused
  coverage             = answered (ANSWER or ANSWER_WITH_LIMITATION) / all questions
  selective_accuracy   = answered AND answerable AND gold section in top-5  /  answered
Threshold selection: sweep tau on DEV only (maximise balanced accuracy), freeze, evaluate on TEST.
"""
from __future__ import annotations

from src.evidence.gate import ABSTAIN


def _div(a, b):
    return a / b if b else None


def abstention_metrics(records: list[dict]) -> dict:
    oos = [r for r in records if not r["answerable"]]
    ans = [r for r in records if r["answerable"]]
    refused = [r for r in records if r["decision"] == ABSTAIN]
    correct_refusals = [r for r in oos if r["decision"] == ABSTAIN]
    answered = [r for r in records if r["decision"] != ABSTAIN]
    good = [r for r in answered if r["answerable"] and r["recall"][5] > 0]
    return {
        "n_oos": len(oos), "n_answerable": len(ans), "correct_oos_refusals": len(correct_refusals),
        "incorrect_oos_answers": len(oos) - len(correct_refusals),
        "wrongly_refused_answerable": len([r for r in ans if r["decision"] == ABSTAIN]),
        "abstention_recall": _div(len(correct_refusals), len(oos)),
        "abstention_precision": _div(len(correct_refusals), len(refused)),
        "coverage": _div(len(answered), len(records)),
        "answerable_coverage": _div(len([r for r in ans if r["decision"] != ABSTAIN]), len(ans)),
        "selective_accuracy": _div(len(good), len(answered)),
        "thresholds_source": sorted({r["thresholds_source"] for r in records}),
    }


def sweep_thresholds(records: list[dict], grid=None) -> list[dict]:
    """Re-decide each record with abstain-threshold tau (record's own OOS boost added)."""
    grid = grid or [i / 100 for i in range(0, 101, 2)]
    rows = []
    for tau in grid:
        rr = []
        for r in records:
            d = dict(r)
            d["decision"] = "ANSWER" if r["confidence"] >= tau + r["oos_boost"] else ABSTAIN
            rr.append(d)
        m = abstention_metrics(rr)
        tpr, tnr = m["abstention_recall"], _div(m["n_answerable"] - m["wrongly_refused_answerable"], m["n_answerable"])
        bal = None if tpr is None or tnr is None else (tpr + tnr) / 2
        rows.append({"tau": tau, **{k: m[k] for k in ("coverage", "abstention_recall", "abstention_precision",
                                                       "selective_accuracy", "wrongly_refused_answerable")},
                     "answerable_coverage": m["answerable_coverage"], "balanced_accuracy": bal})
    return rows


def tune_threshold(dev_records: list[dict], margin: float = 0.15) -> dict:
    """Pick tau maximising balanced accuracy (ties -> smaller tau = higher coverage). DEV records only."""
    rows = [r for r in sweep_thresholds(dev_records) if r["balanced_accuracy"] is not None]
    if not rows:
        raise ValueError("need both answerable and out-of-scope DEV questions to tune a threshold")
    best = max(rows, key=lambda r: (round(r["balanced_accuracy"], 6), -r["tau"]))
    return {"limited": best["tau"], "answer": round(best["tau"] + margin, 4), "margin": margin,
            "dev_balanced_accuracy": best["balanced_accuracy"], "dev_coverage": best["coverage"],
            "n_dev": len(dev_records)}
