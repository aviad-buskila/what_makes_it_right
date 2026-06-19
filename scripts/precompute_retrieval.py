"""Precompute and cache retrieval for every question (run once).

For each question this stores the exact chunks the live retriever returns at the
config's settings (for faithful, fast experiment reproduction) plus a generous
dense candidate pool with distances (so the tuner can replay any top_k /
max_distance offline). Re-running experiments can then read the cache instead of
re-embedding and re-querying ChromaDB.

Example:
    python scripts/build_rag_index.py     --config config.yaml
    python scripts/precompute_retrieval.py --config config.yaml --pool-size 50
    python scripts/run_experiment.py       --config config.yaml --retrieval-cache \\
        data/medqa/retrieval_cache/medqa_textbooks.jsonl
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import load_dataset_by_source, load_questions, save_questions
from src.experiment.config import load_config
from src.llm.client import ensure_model
from src.rag.cache import CachedQuery, default_cache_path, write_cache
from src.rag.retriever import Retriever


def _load_questions(config):
    path = Path(f"data/{config.dataset.source}/questions.jsonl")
    if path.exists() and path.stat().st_size > 0:
        questions = load_questions(path)
        if questions:
            return questions
    questions = load_dataset_by_source(
        source=config.dataset.source,
        random_seed=config.random_seed,
        cache_dir=config.dataset.cache_dir,
        split=config.dataset.split,
        variant=config.dataset.variant,
    )
    save_questions(questions, path)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Precompute retrieval cache")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pool-size", type=int, default=50,
                        help="dense candidates to cache per question (for tuning)")
    parser.add_argument("--out", default=None,
                        help="output cache path (default: data/<source>/retrieval_cache/<collection>.jsonl)")
    args = parser.parse_args()

    config = load_config(args.config)
    if not Path(config.rag.persist_dir).exists():
        print(f"No RAG index at {config.rag.persist_dir}. Run build_rag_index.py first.")
        return

    if config.rag.embedding_backend == "ollama":
        ensure_model(config.rag.embedding_model)

    retriever = Retriever(
        persist_dir=config.rag.persist_dir,
        collection_name=config.rag.collection_name,
        embedding_model=config.rag.embedding_model,
        embedding_backend=config.rag.embedding_backend,
        top_k=config.rag.top_k,
        timeout_per_query=config.timeout_per_query,
        max_distance=config.rag.max_distance,
        retrieval_mode=config.rag.retrieval_mode,
        dense_multiplier=config.rag.dense_multiplier,
        lexical_multiplier=config.rag.lexical_multiplier,
        rerank_alpha=config.rag.rerank_alpha,
        min_lexical_overlap=config.rag.min_lexical_overlap,
    )

    questions = _load_questions(config)
    out_path = Path(args.out) if args.out else default_cache_path(
        config.dataset.source, config.rag.collection_name
    )
    print(f"Caching retrieval for {len(questions)} questions "
          f"(pool_size={args.pool_size}) -> {out_path}")

    entries: list[CachedQuery] = []
    empty = 0
    for q in tqdm(questions, desc="retrieving"):
        query_text = retriever.build_query_text(
            question_text=q.question_text, options=q.options,
            mode=config.rag.query_mode,
        )
        final_chunks = retriever.query(query_text)
        candidates = retriever.fetch_dense_candidates(query_text, n_candidates=args.pool_size)
        if not final_chunks:
            empty += 1
        entries.append(CachedQuery(
            question_id=q.id,
            query=query_text,
            final_chunks=final_chunks,
            candidates=candidates,
        ))

    meta = {
        "dataset_source": config.dataset.source,
        "dataset_variant": config.dataset.variant,
        "collection_name": config.rag.collection_name,
        "embedding_model": config.rag.embedding_model,
        "embedding_backend": config.rag.embedding_backend,
        "query_mode": config.rag.query_mode,
        "retrieval_mode": config.rag.retrieval_mode,
        "build_top_k": config.rag.top_k,
        "build_max_distance": config.rag.max_distance,
        "rerank_alpha": config.rag.rerank_alpha,
        "pool_size": args.pool_size,
        "n_questions": len(entries),
    }
    write_cache(out_path, entries, meta)
    print(f"Wrote cache: {len(entries)} entries "
          f"({empty} with empty context at build config). Meta -> {out_path}.meta.json")


if __name__ == "__main__":
    main()
