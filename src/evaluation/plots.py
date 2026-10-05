"""matplotlib plots generated ONLY from measured results (summary rows / per-query records)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["BM25", "DENSE", "HYBRID_RRF", "HYBRID_RERANK", "ADAPTIVE"]


def _rows(summary_rows, chunking):
    return sorted([r for r in summary_rows if r["chunking"] == chunking], key=lambda r: ORDER.index(r["strategy"]))


def _grouped(summary_rows, chunkings, metrics, title, fname, out):
    fig, axes = plt.subplots(1, len(chunkings), figsize=(5.2 * len(chunkings), 4), squeeze=False)
    for ax, ch in zip(axes[0], chunkings):
        rows = _rows(summary_rows, ch)
        w = 0.8 / len(metrics)
        for j, m in enumerate(metrics):
            ax.bar([i + j * w for i in range(len(rows))], [r["overall"].get(m, 0) for r in rows], w, label=m)
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(rows))])
        ax.set_xticklabels([r["strategy"] for r in rows], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(ch, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=130)
    plt.close(fig)


def make_all_plots(summary_rows: list[dict], sweep: list[dict] | None, out: Path) -> list[str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chunkings = sorted({r["chunking"] for r in summary_rows})
    made = []
    _grouped(summary_rows, chunkings, ["recall@3", "recall@5", "recall@10"], "Recall@k (answerable questions)", "recall_comparison.png", out)
    _grouped(summary_rows, chunkings, ["mrr"], "MRR@10", "mrr_comparison.png", out)
    _grouped(summary_rows, chunkings, ["ndcg@3", "ndcg@5", "ndcg@10"], "nDCG@k", "ndcg_comparison.png", out)
    made += ["recall_comparison.png", "mrr_comparison.png", "ndcg_comparison.png"]

    # latency breakdown (stacked)
    stages = ["classification_ms", "bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "evidence_ms"]
    fig, axes = plt.subplots(1, len(chunkings), figsize=(5.2 * len(chunkings), 4), squeeze=False)
    for ax, ch in zip(axes[0], chunkings):
        rows = _rows(summary_rows, ch)
        bottom = [0.0] * len(rows)
        for st in stages:
            vals = [r["latency_ms_mean"].get(st, 0.0) for r in rows]
            ax.bar([r["strategy"] for r in rows], vals, bottom=bottom, label=st.replace("_ms", ""))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("mean ms / query (no generation)")
        ax.set_title(ch, fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Latency breakdown")
    fig.tight_layout()
    fig.savefig(out / "latency_breakdown.png", dpi=130)
    plt.close(fig)
    made.append("latency_breakdown.png")

    # adaptive vs fixed hybrid
    pick = ["HYBRID_RRF", "HYBRID_RERANK", "ADAPTIVE"]
    fig, axes = plt.subplots(1, len(chunkings), figsize=(5.2 * len(chunkings), 4), squeeze=False)
    for ax, ch in zip(axes[0], chunkings):
        rows = [r for r in _rows(summary_rows, ch) if r["strategy"] in pick]
        ms = ["recall@5", "mrr", "ndcg@5"]
        w = 0.8 / max(1, len(rows))
        for j, r in enumerate(rows):
            ax.bar([i + j * w for i in range(len(ms))], [r["overall"].get(m, 0) for m in ms], w, label=r["strategy"])
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(ms))])
        ax.set_xticklabels(ms)
        ax.set_ylim(0, 1)
        ax.set_title(ch, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Adaptive vs fixed hybrid")
    fig.tight_layout()
    fig.savefig(out / "adaptive_vs_fixed.png", dpi=130)
    plt.close(fig)
    made.append("adaptive_vs_fixed.png")

    # query-type performance (first chunking)
    ch = chunkings[0]
    rows = _rows(summary_rows, ch)
    types = sorted({t for r in rows for t in r["by_type"]})
    if types:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        w = 0.8 / len(rows)
        for j, r in enumerate(rows):
            ax.bar([i + j * w for i in range(len(types))], [r["by_type"].get(t, {}).get("recall@5", 0) for t in types], w, label=r["strategy"])
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(types))])
        ax.set_xticklabels(types, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Recall@5 by query type ({ch})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "query_type_performance.png", dpi=130)
        plt.close(fig)
        made.append("query_type_performance.png")

    # accuracy vs latency
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in summary_rows:
        x, y = r["latency_total_ms"]["mean"], r["overall"].get("mrr")
        if x is None or y is None:
            continue
        ax.scatter(x, y)
        ax.annotate(f"{r['strategy']}\n{r['chunking']}", (x, y), fontsize=6)
    ax.set_xlabel("mean retrieval latency (ms, no generation)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Retrieval quality vs latency")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_latency.png", dpi=130)
    plt.close(fig)
    made.append("accuracy_vs_latency.png")

    if sweep:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [r["tau"] for r in sweep]
        for key in ("coverage", "abstention_recall", "answerable_coverage", "selective_accuracy"):
            ax.plot(xs, [r[key] if r[key] is not None else float("nan") for r in sweep], label=key)
        ax.set_xlabel("abstain threshold tau")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        ax.set_title("Abstention threshold sweep (ADAPTIVE)")
        fig.tight_layout()
        fig.savefig(out / "abstention_sweep.png", dpi=130)
        plt.close(fig)
        made.append("abstention_sweep.png")
    return made
