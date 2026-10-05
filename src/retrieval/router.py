"""Rule-based, explainable query classifier + routing table (the 'adaptive' part).

Every decision records which rules fired. The routing weights below are HYPOTHESES to be evaluated on DEV,
not established optima. A learned classifier can replace RuleBasedClassifier (same `classify` interface).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    HOW_TO = "HOW_TO"
    CONCEPTUAL = "CONCEPTUAL"
    MULTI_SECTION = "MULTI_SECTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class Strategy:
    name: str
    bm25_weight: float
    dense_weight: float
    candidates: int
    rerank: bool
    final_k: int = 5
    cautious: bool = False

    @property
    def use_bm25(self) -> bool:
        return self.bm25_weight > 0

    @property
    def use_dense(self) -> bool:
        return self.dense_weight > 0


FIXED_STRATEGIES: dict[str, Strategy] = {
    "BM25": Strategy("BM25", 1, 0, 50, False),
    "DENSE": Strategy("DENSE", 0, 1, 50, False),
    "HYBRID_RRF": Strategy("HYBRID_RRF", 1, 1, 50, False),
    "HYBRID_RERANK": Strategy("HYBRID_RERANK", 1, 1, 50, True),
}

ADAPTIVE_ROUTES: dict[QueryType, Strategy] = {
    QueryType.LOOKUP: Strategy("ADAPTIVE:LOOKUP->HYBRID_LEXICAL", 2, 1, 30, False),
    QueryType.HOW_TO: Strategy("ADAPTIVE:HOW_TO->HYBRID_RERANK", 1, 1, 50, True),
    QueryType.CONCEPTUAL: Strategy("ADAPTIVE:CONCEPTUAL->HYBRID_DENSE_RERANK", 1, 2, 50, True),
    QueryType.MULTI_SECTION: Strategy("ADAPTIVE:MULTI_SECTION->WIDE_HYBRID_RERANK", 1, 1, 100, True, final_k=8),
    QueryType.OUT_OF_SCOPE: Strategy("ADAPTIVE:OUT_OF_SCOPE->CAUTIOUS_HYBRID_RERANK", 1, 1, 50, True, cautious=True),
}

ROUTE_RATIONALE = {
    QueryType.LOOKUP: "exact identifiers/parameters -> lexical-weighted RRF (2:1), small pool, no rerank",
    QueryType.HOW_TO: "procedural questions -> balanced hybrid + cross-encoder",
    QueryType.CONCEPTUAL: "paraphrase-heavy -> dense-weighted RRF (1:2) + cross-encoder",
    QueryType.MULTI_SECTION: "needs several sections -> 100 candidates, rerank, final_k=8",
    QueryType.OUT_OF_SCOPE: "likely unanswerable -> balanced hybrid + rerank; evidence gate uses a stricter threshold",
}


def get_strategy(name: str) -> Strategy:
    if name not in FIXED_STRATEGIES:
        raise KeyError(f"unknown fixed strategy {name!r}")
    return FIXED_STRATEGIES[name]


# Small domain lexicon (PostgreSQL / SQL vocabulary). Used ONLY for the out-of-scope hint, not for retrieval.
PG_LEXICON = frozenset("""postgres postgresql psql pg sql query queries table tables index indexes indices column columns row rows
schema schemas database databases vacuum analyze autovacuum transaction transactions isolation mvcc wal replication replica
standby primary backup restore dump pg_dump pg_restore role roles grant revoke privilege privileges trigger triggers view
views materialized partition partitions partitioning constraint constraints foreign key unique primary select insert update
delete copy explain plan planner cluster tablespace extension extensions function functions plpgsql procedure json jsonb
datatype type types timestamp date interval text varchar integer numeric serial sequence lock locks locking checkpoint
shared_buffers work_mem authentication pg_hba ssl tls connection connections listen notify cursor window cte join joins
toast gin gist brin btree hash archive archiving recovery pitr logical subscription publication stats statistics monitoring
collation encoding locale permissions rls policy""".split())
FOREIGN_TECH = frozenset("""mysql mariadb mongodb redis kafka kubernetes k8s terraform docker nginx apache react angular vue django
flask pytorch tensorflow pandas excel salesforce sqlite cassandra elasticsearch oracle aws azure gcp""".split())

_RE_MULTI = re.compile(r"\b(compare|comparison|differences?\s+between|versus|vs\.?|trade-?offs?|both|relationship\s+between|"
                       r"end[- ]to[- ]end|pros\s+and\s+cons|how\s+do\s+\w+\s+(and|relate|interact)|work\s+together)\b", re.I)
_RE_LOOKUP = re.compile(r"\b(default(\s+(value|setting))?|what\s+is\s+the\s+(default|maximum|minimum|syntax|name|port)|"
                        r"which\s+(parameter|setting|function|command|view|catalog|data\s?type|option|flag|operator|"
                        r"system\s+view|authentication)|what\s+does\s+.{1,40}\s+(option|flag|parameter|command)|"
                        r"syntax\s+(of|for)|maximum|minimum|accept)\b", re.I)
_RE_HOWTO = re.compile(r"^\s*(how\s+(do|can|should|would|to)|what('s|\s+is)\s+the\s+(best\s+)?way|steps?\s+to|ways?\s+to)|"
                       r"\b(set\s*up|configure|enable|disable|create|install|restore|back\s*up|migrate|tune|speed\s+up|"
                       r"schedule|rebuild|write)\b", re.I)
_RE_CONCEPT = re.compile(r"\b(what\s+(is|are)|why|explain|how\s+does\s+.{1,50}\s+work|purpose\s+of|concept|overview|"
                         r"when\s+(is|are|does|should))\b", re.I)
_ACTIONS = frozenset("create configure enable set grant restore backup monitor tune index partition replicate drop alter "
                     "vacuum reindex load secure authenticate archive compare combine".split())
_RE_IDENT = re.compile(r"\b[a-z]+_[a-z0-9_]+\b|\b(pg|pgsql)_[a-z_]+\b|\b[A-Z]{3,}(\s+[A-Z]{3,})*\b")


@dataclass
class QueryAnalysis:
    query_type: QueryType
    features: dict
    fired_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"query_type": self.query_type.value, "features": self.features, "fired_rules": self.fired_rules}


class RuleBasedClassifier:
    """Ordered rules: OOS -> MULTI_SECTION -> LOOKUP -> HOW_TO -> CONCEPTUAL -> fallback(CONCEPTUAL)."""

    def classify(self, query: str) -> QueryAnalysis:
        q = query.strip()
        toks = re.findall(r"[a-z0-9_]+", q.lower())
        lex_hits = [t for t in toks if t in PG_LEXICON]
        foreign = [t for t in toks if t in FOREIGN_TECH]
        has_ident = bool(_RE_IDENT.search(q))
        n_actions = len({t for t in toks if t in _ACTIONS})
        f = {
            "n_tokens": len(toks), "domain_lexicon_hits": len(lex_hits), "foreign_tech_hits": len(foreign),
            "has_identifier": has_ident, "n_distinct_actions": n_actions,
            "multi_cue": bool(_RE_MULTI.search(q)), "lookup_cue": bool(_RE_LOOKUP.search(q)),
            "howto_cue": bool(_RE_HOWTO.search(q)), "concept_cue": bool(_RE_CONCEPT.search(q)),
        }
        rules: list[str] = []

        def done(t: QueryType, rule: str) -> QueryAnalysis:
            rules.append(rule)
            return QueryAnalysis(t, f, rules)

        if len(toks) < 2:
            return done(QueryType.OUT_OF_SCOPE, "R1_too_short")
        if not lex_hits and not has_ident:
            return done(QueryType.OUT_OF_SCOPE, "R1_no_domain_vocabulary")
        if foreign and len(lex_hits) <= 1:
            return done(QueryType.OUT_OF_SCOPE, "R1_foreign_technology")
        if f["multi_cue"] or (n_actions >= 2 and re.search(r"\band\b", q, re.I)):
            return done(QueryType.MULTI_SECTION, "R2_multi_cue" if f["multi_cue"] else "R2_multiple_actions_joined_by_and")
        if (has_ident and not f["howto_cue"]) or f["lookup_cue"]:
            if not (f["howto_cue"] and not has_ident and not re.search(r"\bdefault\b", q, re.I)):
                return done(QueryType.LOOKUP, "R3_identifier_or_lookup_cue")
        if f["howto_cue"]:
            return done(QueryType.HOW_TO, "R4_howto_cue")
        if f["concept_cue"]:
            return done(QueryType.CONCEPTUAL, "R5_concept_cue")
        return done(QueryType.CONCEPTUAL, "R6_fallback")


def select_strategy(analysis: QueryAnalysis) -> tuple[Strategy, str]:
    t = analysis.query_type
    return ADAPTIVE_ROUTES[t], f"{t.value}: {ROUTE_RATIONALE[t]}"
