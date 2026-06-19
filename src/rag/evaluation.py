"""Retrieval-only evaluation, diagnostics, and tuning.

This module intentionally has no dependency on LLM generation. It lets you
debug, calibrate, and tune the RAG pipeline (chunking, indexing, retrieval
params) against retrieval-quality signals derived from the question + gold
option, without paying the cost (or noise) of generation.

Key metrics
-----------
Intrinsic (no labels):
  * coverage          — fraction of queries with ≥1 chunk below max_distance
  * top1_distance     — distribution of the nearest neighbour's distance
  * topk_distance     — distribution of all top-K distances
  * chunk_len         — distribution of retrieved chunk lengths (chars)
  * source_mix        — fraction of retrieved chunks per KB source
  * latency_seconds   — per-query retrieval wall time

Silver-label (no LLM, uses the gold option text):
  * correct_term_recall    — share of key terms from the correct option that
                             appear anywhere in the retrieved context
  * discrimination         — overlap(correct_option, ctx) minus mean
                             overlap(distractor_i, ctx); positive means the
                             retriever is picking evidence aligned with the
                             gold answer more than with distractors
  * retrieval_only_acc@K   — accuracy of a rule "pick the option with the
                             highest term overlap against the retrieved
                             context". This is a ceiling for what RAG alone
                             could contribute if the LLM were a lookup table.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median, quantiles

from src.dataset.loader import Question
from src.rag.retriever import Retriever, _token_set


SOURCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mitre_attack", re.compile(r"\bT\d{4}(?:\.\d+)?\b")),
    ("cwe", re.compile(r"\bCWE[-_ ]?\d+\b", re.IGNORECASE)),
    ("nist_800_53", re.compile(r"NIST SP 800-53|\b[A-Z]{2}-\d+(?:\(\d+\))?\b")),
    ("owasp", re.compile(r"OWASP Top 10|A\d{2}:2021", re.IGNORECASE)),
]


def infer_source(chunk: str) -> str:
    """Heuristically label a chunk with the KB source that produced it."""
    for name, pat in SOURCE_PATTERNS:
        if pat.search(chunk):
            return name
    return "unknown"


def _term_overlap(target: str, context: str) -> tuple[float, int, int]:
    """Return (recall, matched_terms, total_terms) between target and context."""
    target_tokens = _token_set(target)
    if not target_tokens:
        return 0.0, 0, 0
    context_tokens = _token_set(context)
    matched = target_tokens & context_tokens
    return len(matched) / len(target_tokens), len(matched), len(target_tokens)


@dataclass
class QueryDiagnostics:
    question_id: str
    correct_answer: str
    n_retrieved: int
    distances: list[float]
    chunk_lengths: list[int]
    sources: list[str]
    latency_seconds: float
    covered: bool
    correct_term_recall: float
    option_overlap: dict[str, float]
    discrimination: float
    heuristic_pick: str | None
    heuristic_correct: bool
    chunks_preview: list[str] = field(default_factory=list)


def evaluate_query(
    retriever: Retriever,
    question: Question,
    top_k: int | None = None,
    preview_chars: int = 180,
) -> QueryDiagnostics:
    """Run one query and compute all retrieval diagnostics for it."""
    k = top_k or retriever.top_k
    start = time.perf_counter()
    chunks = retriever.query(question.question_text, top_k=k)
    latency = time.perf_counter() - start

    distances = _raw_distances(retriever, question.question_text, k)
    chunk_lengths = [len(c) for c in chunks]
    sources = [infer_source(c) for c in chunks]
    context_blob = "\n\n".join(chunks)

    option_overlap: dict[str, float] = {}
    for letter, text in question.options.items():
        recall, _, _ = _term_overlap(text, context_blob)
        option_overlap[letter] = recall

    correct_recall = option_overlap.get(question.correct_answer, 0.0)
    distractor_scores = [v for k_, v in option_overlap.items()
                         if k_ != question.correct_answer]
    discrimination = correct_recall - (mean(distractor_scores)
                                       if distractor_scores else 0.0)

    heuristic_pick = None
    if option_overlap and max(option_overlap.values()) > 0:
        heuristic_pick = max(option_overlap, key=option_overlap.get)
    heuristic_correct = heuristic_pick == question.correct_answer

    previews = [c[:preview_chars].replace("\n", " ") + ("…" if len(c) > preview_chars else "")
                for c in chunks]

    return QueryDiagnostics(
        question_id=question.id,
        correct_answer=question.correct_answer,
        n_retrieved=len(chunks),
        distances=distances,
        chunk_lengths=chunk_lengths,
        sources=sources,
        latency_seconds=latency,
        covered=len(chunks) > 0,
        correct_term_recall=correct_recall,
        option_overlap=option_overlap,
        discrimination=discrimination,
        heuristic_pick=heuristic_pick,
        heuristic_correct=heuristic_correct,
        chunks_preview=previews,
    )


def _raw_distances(retriever: Retriever, text: str, k: int) -> list[float]:
    """Return top-K raw cosine distances from Chroma (pre-reranking).

    This bypasses the lexical rerank so the distribution reflects the index
    itself, which is what you want when tuning ``max_distance``.
    """
    emb = retriever.embed_query(text)
    res = retriever.collection.query(
        query_embeddings=[emb],
        n_results=max(k, 1),
        include=["distances"],
    )
    if not res.get("distances") or not res["distances"]:
        return []
    return [float(d) for d in res["distances"][0]]


@dataclass
class AggregateMetrics:
    n_queries: int
    coverage: float
    mean_top1_distance: float
    median_top1_distance: float
    p10_top1_distance: float
    p90_top1_distance: float
    mean_topk_distance: float
    mean_chunks_returned: float
    mean_chunk_length: float
    mean_latency_seconds: float
    mean_correct_term_recall: float
    mean_discrimination: float
    retrieval_only_accuracy: float
    source_mix: dict[str, float]


def aggregate(diagnostics: list[QueryDiagnostics]) -> AggregateMetrics:
    if not diagnostics:
        raise ValueError("No diagnostics to aggregate.")

    top1 = [d.distances[0] for d in diagnostics if d.distances]
    topk = [x for d in diagnostics for x in d.distances]
    chunk_lens = [x for d in diagnostics for x in d.chunk_lengths]
    src_counter: Counter[str] = Counter()
    for d in diagnostics:
        src_counter.update(d.sources)
    total_sources = sum(src_counter.values()) or 1

    return AggregateMetrics(
        n_queries=len(diagnostics),
        coverage=sum(1 for d in diagnostics if d.covered) / len(diagnostics),
        mean_top1_distance=mean(top1) if top1 else float("nan"),
        median_top1_distance=median(top1) if top1 else float("nan"),
        p10_top1_distance=_percentile(top1, 10),
        p90_top1_distance=_percentile(top1, 90),
        mean_topk_distance=mean(topk) if topk else float("nan"),
        mean_chunks_returned=mean(d.n_retrieved for d in diagnostics),
        mean_chunk_length=mean(chunk_lens) if chunk_lens else 0.0,
        mean_latency_seconds=mean(d.latency_seconds for d in diagnostics),
        mean_correct_term_recall=mean(d.correct_term_recall for d in diagnostics),
        mean_discrimination=mean(d.discrimination for d in diagnostics),
        retrieval_only_accuracy=sum(1 for d in diagnostics
                                    if d.heuristic_correct) / len(diagnostics),
        source_mix={k: v / total_sources for k, v in src_counter.items()},
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    # quantiles(n=100) returns 99 cut points
    cuts = quantiles(values, n=100, method="inclusive")
    idx = max(0, min(len(cuts) - 1, int(pct) - 1))
    return cuts[idx]


def plot_distance_histogram(distances: list[float], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not distances:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(distances, bins=40, edgecolor="black", alpha=0.8)
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("chunk count")
    ax.set_title("Retrieved chunk distance distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_source_distribution(source_mix: dict[str, float], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not source_mix:
        return
    labels, values = zip(*sorted(source_mix.items(),
                                 key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, edgecolor="black", alpha=0.85)
    ax.set_ylabel("fraction of retrieved chunks")
    ax.set_title("Retrieval provenance")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
