"""Evidence gate: ANSWER / ANSWER_WITH_LIMITATION / ABSTAIN from a confidence score and thresholds.

Thresholds are configuration, not truth: tune on DEV with scripts/run_eval.py --tune, freeze in
configs/thresholds.json, then evaluate on TEST. Queries the router flags OUT_OF_SCOPE need `oos_boost` more.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings

ANSWER, LIMITED, ABSTAIN = "ANSWER", "ANSWER_WITH_LIMITATION", "ABSTAIN"
REFUSAL_MESSAGE = "I don't have sufficient evidence in the indexed documentation to answer this reliably."


@dataclass(frozen=True)
class Thresholds:
    answer: float = 0.65
    limited: float = 0.45
    oos_boost: float = 0.10
    source: str = "default-uncalibrated"


@dataclass
class GateResult:
    decision: str
    confidence: float
    required_answer: float
    required_limited: float
    thresholds_source: str

    def to_dict(self) -> dict:
        return {"decision": self.decision, "confidence": round(self.confidence, 4),
                "required_confidence": round(self.required_answer, 4),
                "required_for_limited_answer": round(self.required_limited, 4),
                "thresholds_source": self.thresholds_source}


def decide(confidence: float, query_type: str, th: Thresholds) -> GateResult:
    boost = th.oos_boost if query_type == "OUT_OF_SCOPE" else 0.0
    req_a, req_l = th.answer + boost, th.limited + boost
    d = ANSWER if confidence >= req_a else LIMITED if confidence >= req_l else ABSTAIN
    return GateResult(d, confidence, req_a, req_l, th.source)


def threshold_key(strategy: str, chunking: str) -> str:
    return f"{strategy}|{chunking}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_thresholds(settings: Settings, tun: dict, strategy: str, chunking: str) -> Thresholds:
    """Frozen (DEV-tuned) thresholds if present for this (strategy, chunking); otherwise uncalibrated defaults."""
    d = tun["thresholds"]
    p = Path(settings.thresholds_file)
    if p.exists():
        data = json.loads(p.read_text())
        e = data.get("entries", {}).get(threshold_key(strategy.split(":")[0], chunking))
        if e:
            return Thresholds(e["answer"], e["limited"], e.get("oos_boost", d["oos_boost"]),
                              f"frozen-on-{data.get('frozen_on_split', '?')}")
    return Thresholds(d["answer"], d["limited"], d["oos_boost"], "default-uncalibrated")
