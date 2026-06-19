"""Persistent retrieval cache.

Runs retrieval once over all questions and stores, per question:

  * ``query``        — the exact query text used (built per ``query_mode``),
  * ``final_chunks`` — the chunks the live retriever returned at the build-time
    config (faithful reproduction for the experiment runner), and
  * ``candidates``   — a generous dense candidate pool of ``(chunk, distance)``
    pairs, so any ``top_k`` / ``max_distance`` can be replayed offline by the
    tuner without re-embedding or re-querying ChromaDB.

This makes experiments reproducible and fast (no repeated retrieval) and powers
the cache-backed retrieval tuner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.retriever import rank_candidates


@dataclass
class CachedQuery:
    question_id: str
    query: str
    final_chunks: list[str]
    candidates: list[tuple[str, float]]  # (chunk, cosine_distance), ascending


@dataclass
class RetrievalCache:
    """In-memory view of a retrieval cache file, with offline re-ranking."""

    by_id: dict[str, CachedQuery] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.by_id)

    def get_final_chunks(self, question_id: str) -> list[str]:
        entry = self.by_id.get(question_id)
        return list(entry.final_chunks) if entry else []

    def get_candidates(self, question_id: str) -> list[tuple[str, float]]:
        entry = self.by_id.get(question_id)
        return list(entry.candidates) if entry else []

    def get_query(self, question_id: str) -> str:
        entry = self.by_id.get(question_id)
        return entry.query if entry else ""

    def rank_chunks(
        self,
        question_id: str,
        top_k: int,
        max_distance: float,
        rerank_alpha: float = 0.6,
        min_lexical_overlap: float = 0.0,
        mode: str = "balanced",
    ) -> list[str]:
        """Replay retrieval selection for arbitrary params from the cached pool."""
        entry = self.by_id.get(question_id)
        if entry is None:
            return []
        return rank_candidates(
            query_text=entry.query,
            candidates=entry.candidates,
            top_k=top_k,
            max_distance=max_distance,
            rerank_alpha=rerank_alpha,
            min_lexical_overlap=min_lexical_overlap,
            mode=mode,
        )

    def final_chunks_map(self) -> dict[str, list[str]]:
        """{question_id: final_chunks} for feeding the experiment runner."""
        return {qid: list(e.final_chunks) for qid, e in self.by_id.items()}

    def ranked_chunks_map(
        self,
        top_k: int,
        max_distance: float,
        rerank_alpha: float = 0.6,
        min_lexical_overlap: float = 0.0,
        mode: str = "balanced",
    ) -> dict[str, list[str]]:
        """{question_id: chunks} re-ranked offline at the given params."""
        return {
            qid: self.rank_chunks(qid, top_k, max_distance, rerank_alpha,
                                  min_lexical_overlap, mode)
            for qid in self.by_id
        }


# -- persistence ------------------------------------------------------------

def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def write_cache(path: str | Path, entries: list[CachedQuery], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps({
                "question_id": e.question_id,
                "query": e.query,
                "final_chunks": e.final_chunks,
                "candidates": [[doc, dist] for doc, dist in e.candidates],
            }) + "\n")
    _meta_path(path).write_text(json.dumps(meta, indent=2))


def load_cache(path: str | Path) -> RetrievalCache:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval cache not found: {path}")
    by_id: dict[str, CachedQuery] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[rec["question_id"]] = CachedQuery(
                question_id=rec["question_id"],
                query=rec.get("query", ""),
                final_chunks=list(rec.get("final_chunks", [])),
                candidates=[(c[0], float(c[1])) for c in rec.get("candidates", [])],
            )
    meta = {}
    mp = _meta_path(path)
    if mp.exists():
        meta = json.loads(mp.read_text())
    return RetrievalCache(by_id=by_id, meta=meta)


def default_cache_path(dataset_source: str, collection_name: str) -> Path:
    return Path(f"data/{dataset_source}/retrieval_cache/{collection_name}.jsonl")
