"""RAG robustness sweep.

Re-runs the 2x2 experiment over a grid of *query-time* retrieval
hyperparameters (top_k, max_distance, rerank_alpha, query_mode). These do not
change the index, so no re-embedding/re-indexing is required between grid
points. Each grid point is stored as its own resumable experiment, and the
RAG effect (with/without RAG, per backbone) is reported for each.

Use a subset via ``experiment.max_questions`` in the config for a cheaper sweep.

Example:
    python scripts/rag_sweep.py --config config.yaml \\
        --top-k 1,3,5 --max-distance 0.2,0.3,0.4 \\
        --rerank-alpha 0.6 --query-mode question_plus_options
"""

import argparse
import copy
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import load_dataset_by_source, load_questions, save_questions
from src.experiment.config import load_config
from src.experiment.runner import build_setups, run_experiment
from src.llm.client import ensure_model
from src.rag.retriever import Retriever
from src.storage.results import ResultsStore
from src.analysis.statistics import rag_effect_tests


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _load_questions(config) -> list:
    path = Path(f"data/{config.dataset.source}/questions.jsonl")
    if path.exists() and path.stat().st_size > 0:
        questions = load_questions(path)
    else:
        questions = load_dataset_by_source(
            source=config.dataset.source,
            random_seed=config.random_seed,
            cache_dir=config.dataset.cache_dir,
            split=config.dataset.split,
            variant=config.dataset.variant,
        )
        save_questions(questions, path)
    if config.max_questions is not None and len(questions) > config.max_questions:
        import random
        questions = random.Random(config.random_seed).sample(questions, config.max_questions)
    return questions


def main():
    parser = argparse.ArgumentParser(description="RAG retrieval hyperparameter sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top-k", default="1,3,5", help="comma-separated top_k values")
    parser.add_argument("--max-distance", default="0.2,0.3,0.4",
                        help="comma-separated cosine distance cutoffs")
    parser.add_argument("--rerank-alpha", default="0.6",
                        help="comma-separated dense-fusion weights")
    parser.add_argument("--query-mode", default="question_plus_options",
                        help="comma-separated query modes")
    args = parser.parse_args()

    base_config = load_config(args.config)
    chroma_path = base_config.rag.persist_dir
    if not Path(chroma_path).exists():
        print(f"No RAG index at {chroma_path}. Run build_rag_index.py first.")
        return

    if base_config.rag.embedding_backend == "ollama":
        ensure_model(base_config.rag.embedding_model)

    questions = _load_questions(base_config)
    store = ResultsStore(base_config.results_dir)

    grid = list(itertools.product(
        _ints(args.top_k),
        _floats(args.max_distance),
        _floats(args.rerank_alpha),
        [m.strip() for m in args.query_mode.split(",") if m.strip()],
    ))
    print(f"Sweeping {len(grid)} grid points over {len(questions)} questions.")

    summary_rows = []
    for k, dist, alpha, qmode in grid:
        config = copy.deepcopy(base_config)
        config.rag.top_k = k
        config.rag.max_distance = dist
        config.rag.rerank_alpha = alpha
        config.rag.query_mode = qmode
        config.rag.oracle_modes = None  # sweep is RAG-only
        tag = f"k{k}_d{dist:g}_a{alpha:g}_{qmode}"
        config.name = f"{base_config.name}__sweep_{tag}"

        print(f"\n=== Grid point: {tag} (experiment={config.name}) ===")
        retriever = Retriever(
            persist_dir=chroma_path,
            collection_name=config.rag.collection_name,
            embedding_model=config.rag.embedding_model,
            embedding_backend=config.rag.embedding_backend,
            top_k=k,
            timeout_per_query=config.timeout_per_query,
            max_distance=dist,
            retrieval_mode=config.rag.retrieval_mode,
            dense_multiplier=config.rag.dense_multiplier,
            lexical_multiplier=config.rag.lexical_multiplier,
            rerank_alpha=alpha,
            min_lexical_overlap=config.rag.min_lexical_overlap,
        )
        setups = build_setups(config, retriever=retriever)
        run_experiment(config, questions, setups, store, retriever=retriever)

        df = store.load(config.name)
        rag_tests = rag_effect_tests(df)
        for _, row in rag_tests.iterrows():
            summary_rows.append({
                "grid": tag,
                "comparison": f"{row['setup_a']} vs {row['setup_b']}",
                "a_only": row["a_only_correct"],
                "b_only": row["b_only_correct"],
                "p_value": row["p_value"],
                "odds_ratio": row.get("odds_ratio"),
            })

    print("\n" + "=" * 70)
    print("RAG effect across the sweep (b_only favours +RAG):")
    import pandas as pd
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    out_path = Path(base_config.results_dir) / f"{base_config.name}__sweep_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSweep summary written to {out_path}")


if __name__ == "__main__":
    main()
