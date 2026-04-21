"""Run the full experiment across all setups."""

import argparse
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


def _default_questions_path(dataset_source: str) -> Path:
    return Path(f"data/{dataset_source}/questions.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run MCQ RAG evaluation experiment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--questions",
        default=None,
        help="Questions JSONL path (defaults to data/<dataset_source>/questions.jsonl)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    questions_path = Path(args.questions) if args.questions else _default_questions_path(
        config.dataset.source
    )

    # Load or download questions. Treat a zero-byte / empty JSONL as missing
    # so a stale file from a failed prior download gets replaced automatically.
    needs_download = (
        not questions_path.exists()
        or questions_path.stat().st_size == 0
    )
    if not needs_download:
        print(f"Loading questions from {questions_path}...")
        questions = load_questions(questions_path)
        if not questions:
            needs_download = True
            print(f"{questions_path} contained no questions — re-downloading.")
    if needs_download:
        print(
            f"Downloading questions (source={config.dataset.source}, "
            f"variant={config.dataset.variant})..."
        )
        questions = load_dataset_by_source(
            source=config.dataset.source,
            max_questions=None,  # always save the full dataset
            random_seed=config.random_seed,
            cache_dir=config.dataset.cache_dir,
            split=config.dataset.split,
            variant=config.dataset.variant,
        )
        save_questions(questions, questions_path)

    if config.max_questions is not None and len(questions) > config.max_questions:
        import random
        rng = random.Random(config.random_seed)
        questions = rng.sample(questions, config.max_questions)

    print(f"Domain:        {config.domain}")
    print(f"Dataset:       {config.dataset.source} (variant={config.dataset.variant})")
    print(f"Questions:     {len(questions)}")
    print(f"Repetitions:   {config.repetitions}")
    print(f"Models:        {[m.name for m in config.models]}")
    print(f"Setups:        {len(config.models) * 2} (each model +/- RAG)")
    total = len(questions) * config.repetitions * len(config.models) * 2
    print(f"Total calls:   {total}")

    # Initialize RAG retriever
    chroma_path = config.rag.persist_dir
    retriever = None
    if Path(chroma_path).exists():
        try:
            ensure_model(config.rag.embedding_model)
            retriever = Retriever(
                persist_dir=chroma_path,
                collection_name=config.rag.collection_name,
                embedding_model=config.rag.embedding_model,
                top_k=config.rag.top_k,
                timeout_per_query=config.timeout_per_query,
                max_distance=config.rag.max_distance,
                retrieval_mode=config.rag.retrieval_mode,
                dense_multiplier=config.rag.dense_multiplier,
                lexical_multiplier=config.rag.lexical_multiplier,
                rerank_alpha=config.rag.rerank_alpha,
                min_lexical_overlap=config.rag.min_lexical_overlap,
            )
            print(f"RAG retriever loaded ({retriever.collection.count()} chunks)")
        except Exception as e:
            print(f"Warning: Could not load RAG index: {e}")
            print("RAG setups will be skipped.")
    else:
        print(f"Warning: No RAG index found at {chroma_path}. Run build_rag_index.py first.")
        print("RAG setups will be skipped.")

    # Build setups
    setups = build_setups(config, retriever=retriever)

    # Filter out RAG setups if no retriever
    if retriever is None:
        setups = [s for s in setups if not s.use_rag]
        print(f"Running {len(setups)} setups (RAG setups skipped).")

    # Run
    store = ResultsStore(config.results_dir)
    run_experiment(config, questions, setups, store, retriever=retriever)

    print(f"\nExperiment complete! Results saved to {config.results_dir}/{config.name}.jsonl")


if __name__ == "__main__":
    main()
