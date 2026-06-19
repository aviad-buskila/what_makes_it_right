"""Cache-backed retrieval correctness evaluation and (k, max_distance) tuner.

Built on the silver-label retrieval-correctness metrics in src/rag/evaluation.py
and the precomputed retrieval cache (scripts/precompute_retrieval.py), so it runs
instantly and reproducibly with no embedding/DB calls and no LLM.

Modes:
  default            evaluate retrieval correctness at the config's top_k /
                     max_distance on a sample, print aggregate metrics. (#1)
  --sweep            grid-sweep top_k x max_distance, rank configs by
                     retrieval-only accuracy and discrimination. (#3)
  --question-id ID   inspect one question's retrieval (chunks, distances,
                     per-option overlap, gold vs heuristic pick). (#1 debug)

Example:
    python scripts/precompute_retrieval.py --config config.yaml
    python scripts/tune_retrieval.py --config config.yaml --sweep \\
        --top-k-grid 1,3,5,8 --max-distance-grid 0.2,0.25,0.3,0.35,0.4
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import Question, load_dataset_by_source, load_questions, save_questions
from src.experiment.config import load_config
from src.rag.cache import default_cache_path, load_cache
from src.rag.evaluation import QueryDiagnostics, aggregate, infer_source, score_chunks
from src.rag.retriever import rank_candidates


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _load_questions(config) -> dict[str, Question]:
    path = Path(f"data/{config.dataset.source}/questions.jsonl")
    if path.exists() and path.stat().st_size > 0:
        questions = load_questions(path)
    else:
        questions = load_dataset_by_source(
            source=config.dataset.source, random_seed=config.random_seed,
            cache_dir=config.dataset.cache_dir, split=config.dataset.split,
            variant=config.dataset.variant,
        )
        save_questions(questions, path)
    return {q.id: q for q in questions}


def _diagnostics_for(
    question: Question, query: str, candidates: list[tuple[str, float]],
    top_k: int, max_distance: float, rerank_alpha: float,
    min_lexical_overlap: float, mode: str,
) -> QueryDiagnostics:
    """Build a QueryDiagnostics from cached candidates at the given params."""
    chunks = rank_candidates(query, candidates, top_k, max_distance,
                             rerank_alpha, min_lexical_overlap, mode)
    dist_by_doc = {doc: dist for doc, dist in candidates}
    distances = [dist_by_doc.get(c, max_distance) for c in chunks]
    scored = score_chunks(question, chunks)
    return QueryDiagnostics(
        question_id=question.id,
        correct_answer=question.correct_answer,
        n_retrieved=len(chunks),
        distances=distances,
        chunk_lengths=[len(c) for c in chunks],
        sources=[infer_source(c) for c in chunks],
        latency_seconds=0.0,
        covered=len(chunks) > 0,
        correct_term_recall=scored["correct_term_recall"],
        option_overlap=scored["option_overlap"],
        discrimination=scored["discrimination"],
        heuristic_pick=scored["heuristic_pick"],
        heuristic_correct=scored["heuristic_correct"],
        chunks_preview=[c[:180].replace("\n", " ") for c in chunks],
    )


def _print_aggregate(m) -> None:
    print(f"\n=== retrieval correctness (n={m.n_queries}) ===")
    print(f"coverage                 : {m.coverage:.3f}")
    print(f"retrieval-only accuracy  : {m.retrieval_only_accuracy:.3f}  (chance 0.25)")
    print(f"mean correct-term recall : {m.mean_correct_term_recall:.3f}")
    print(f"mean discrimination      : {m.mean_discrimination:+.3f}")
    print(f"mean top-1 distance      : {m.mean_top1_distance:.3f}")
    print(f"mean chunks returned     : {m.mean_chunks_returned:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Cache-backed retrieval eval / tuner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cache", default=None, help="retrieval cache path")
    parser.add_argument("--sample", type=int, default=None,
                        help="evaluate only the first N cached questions")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--top-k-grid", default="1,3,5,8")
    parser.add_argument("--max-distance-grid", default="0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--output", default=None, help="dir for sweep report")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_path = Path(args.cache) if args.cache else default_cache_path(
        config.dataset.source, config.rag.collection_name)
    cache = load_cache(cache_path)
    questions = _load_questions(config)
    print(f"cache: {cache_path}  entries={len(cache)}  meta_query_mode="
          f"{cache.meta.get('query_mode')}")

    alpha = config.rag.rerank_alpha
    min_lex = config.rag.min_lexical_overlap
    mode = config.rag.retrieval_mode

    qids = [qid for qid in cache.by_id if qid in questions]
    if args.sample:
        qids = qids[: args.sample]

    # --- Single-question debug (#1) ---
    if args.question_id:
        qid = args.question_id
        if qid not in cache.by_id or qid not in questions:
            print(f"question {qid!r} not in cache/dataset.")
            return
        q = questions[qid]
        entry = cache.by_id[qid]
        diag = _diagnostics_for(q, entry.query, entry.candidates,
                                config.rag.top_k, config.rag.max_distance,
                                alpha, min_lex, mode)
        print(f"\nquestion: {q.question_text[:300]}")
        for letter, text in sorted(q.options.items()):
            marker = "*" if letter == q.correct_answer else " "
            print(f"  {marker} {letter}) {text[:80]}   [overlap={diag.option_overlap.get(letter,0):.2f}]")
        print(f"gold={q.correct_answer} heuristic_pick={diag.heuristic_pick} "
              f"correct={diag.heuristic_correct} recall={diag.correct_term_recall:.2f} "
              f"discrimination={diag.discrimination:+.2f}")
        for i, (chunk, dist) in enumerate(zip(diag.chunks_preview, diag.distances), 1):
            print(f"\n[{i}] dist={dist:.3f}\n    {chunk}")
        return

    # --- Sweep (#3) ---
    if args.sweep:
        top_k_grid = _parse_ints(args.top_k_grid)
        max_distance_grid = _parse_floats(args.max_distance_grid)
        print(f"sweep: top_k={top_k_grid} x max_distance={max_distance_grid} "
              f"over {len(qids)} questions")
        rows = []
        for maxd in max_distance_grid:
            for k in top_k_grid:
                diags = [
                    _diagnostics_for(questions[qid], cache.by_id[qid].query,
                                     cache.by_id[qid].candidates, k, maxd,
                                     alpha, min_lex, mode)
                    for qid in qids
                ]
                m = aggregate(diags)
                rows.append({
                    "top_k": k, "max_distance": maxd,
                    "coverage": round(m.coverage, 4),
                    "retrieval_only_acc": round(m.retrieval_only_accuracy, 4),
                    "mean_correct_recall": round(m.mean_correct_term_recall, 4),
                    "mean_discrimination": round(m.mean_discrimination, 4),
                    "mean_top1_dist": round(m.mean_top1_distance, 4),
                })
                print(f"  k={k:>2} maxd={maxd:.2f}  cov={m.coverage:.2f} "
                      f"acc={m.retrieval_only_accuracy:.3f} disc={m.mean_discrimination:+.3f}")

        rows.sort(key=lambda r: (r["retrieval_only_acc"], r["mean_discrimination"]),
                  reverse=True)
        out_dir = Path(args.output) if args.output else Path(
            f"results/retrieval/{config.name}_tune")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tune_sweep.json").write_text(json.dumps(rows, indent=2))
        best = rows[0]
        print(f"\nBest config: top_k={best['top_k']} max_distance={best['max_distance']} "
              f"(retrieval-only acc={best['retrieval_only_acc']:.3f})")
        print(f"Full ranked sweep -> {out_dir/'tune_sweep.json'}")
        return

    # --- Default: evaluate at the config's settings (#1) ---
    diags = [
        _diagnostics_for(questions[qid], cache.by_id[qid].query,
                         cache.by_id[qid].candidates, config.rag.top_k,
                         config.rag.max_distance, alpha, min_lex, mode)
        for qid in qids
    ]
    _print_aggregate(aggregate(diags))


if __name__ == "__main__":
    main()
