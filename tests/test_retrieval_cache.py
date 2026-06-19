"""Tests for the retrieval cache and offline candidate ranking.

Run directly: ``python tests/test_retrieval_cache.py`` (needs chromadb installed,
since src.rag.retriever imports it; no network or index required).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.cache import CachedQuery, RetrievalCache, load_cache, write_cache
from src.rag.retriever import rank_candidates


def main() -> int:
    failures: list[str] = []

    cands = [
        ("myocardial infarction chest pain aspirin treatment", 0.10),
        ("warfarin anticoagulation bleeding risk management", 0.22),
        ("acetaminophen hepatic toxicity overdose", 0.28),
        ("gardening tips for tomato plants", 0.45),
    ]
    query = "chest pain myocardial infarction best next step aspirin"

    # 1. Distance cutoff respected.
    if rank_candidates(query, cands, top_k=5, max_distance=0.05) != []:
        failures.append("max_distance cutoff did not drop all far candidates")
    if len(rank_candidates(query, cands, top_k=5, max_distance=0.5)) != 4:
        failures.append("loose cutoff should keep all 4 candidates")

    # 2. top_k truncation.
    if len(rank_candidates(query, cands, top_k=2, max_distance=0.3)) != 2:
        failures.append("top_k truncation failed")

    # 3. Fusion ranks the query-matching MI chunk first.
    top1 = rank_candidates(query, cands, top_k=1, max_distance=0.3)
    if not top1 or "myocardial" not in top1[0]:
        failures.append(f"fusion did not rank the relevant chunk first: {top1}")

    # 4. fast mode returns by ascending distance.
    fast = rank_candidates(query, cands, top_k=2, max_distance=0.5, mode="fast")
    if fast != [cands[0][0], cands[1][0]]:
        failures.append(f"fast mode order wrong: {fast}")
    print(f"[{'PASS' if not failures else 'FAIL'}] rank_candidates cutoff/k/fusion/fast")

    # 5. Cache round-trip + offline re-ranking maps.
    entries = [
        CachedQuery(question_id="medqa_test_0", query=query,
                    final_chunks=[cands[0][0], cands[1][0]], candidates=cands),
        CachedQuery(question_id="medqa_test_1", query="unrelated query",
                    final_chunks=[], candidates=[("gardening tomato plants", 0.6)]),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cache.jsonl"
        write_cache(path, entries, {"query_mode": "question_plus_options", "n_questions": 2})
        cache = load_cache(path)
        if len(cache) != 2:
            failures.append("cache length wrong after round-trip")
        if cache.meta.get("query_mode") != "question_plus_options":
            failures.append("cache meta not persisted")
        if cache.get_final_chunks("medqa_test_0") != [cands[0][0], cands[1][0]]:
            failures.append("final_chunks round-trip mismatch")
        # Offline re-rank at a tight cutoff keeps the closest only.
        ranked = cache.rank_chunks("medqa_test_0", top_k=1, max_distance=0.15)
        if not ranked or "myocardial" not in ranked[0]:
            failures.append(f"cache.rank_chunks wrong: {ranked}")
        fmap = cache.final_chunks_map()
        if set(fmap) != {"medqa_test_0", "medqa_test_1"}:
            failures.append("final_chunks_map keys wrong")
        rmap = cache.ranked_chunks_map(top_k=3, max_distance=0.3)
        if rmap["medqa_test_1"] != []:
            failures.append("ranked_chunks_map should be empty for far-only candidates")
    print(f"[{'PASS' if 'cache' not in ''.join(failures) and 'round-trip' not in ''.join(failures) else 'FAIL'}] "
          "cache write/load + offline re-rank maps")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All retrieval-cache tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
