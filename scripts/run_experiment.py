"""Run the full experiment across all setups."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import load_questions, load_medqa, save_questions
from src.experiment.config import load_config
from src.experiment.runner import build_setups, run_experiment
from src.rag.retriever import Retriever
from src.storage.results import ResultsStore


def main():
    parser = argparse.ArgumentParser(description="Run medical QA experiment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--questions", default="data/medqa/questions.jsonl", help="Questions JSONL path")
    args = parser.parse_args()

    config = load_config(args.config)
    questions_path = Path(args.questions)

    # Load or download questions
    if questions_path.exists():
        print(f"Loading questions from {questions_path}...")
        questions = load_questions(questions_path)
    else:
        print("Questions file not found, downloading...")
        questions = load_medqa(
            split="test",
            max_questions=config.max_questions,
            random_seed=config.random_seed,
        )
        save_questions(questions, questions_path)

    if config.max_questions and len(questions) > config.max_questions:
        import random
        rng = random.Random(config.random_seed)
        questions = rng.sample(questions, config.max_questions)

    print(f"Questions: {len(questions)}")
    print(f"Repetitions: {config.repetitions}")
    print(f"Models: {[m.name for m in config.models]}")
    print(f"Setups: {len(config.models) * 2} (each model +/- RAG)")
    total = len(questions) * config.repetitions * len(config.models) * 2
    print(f"Total LLM calls: {total}")

    # Initialize RAG retriever
    chroma_path = "data/chroma_db"
    retriever = None
    if Path(chroma_path).exists():
        try:
            retriever = Retriever(
                persist_dir=chroma_path,
                collection_name=config.rag.collection_name,
                embedding_model=config.rag.embedding_model,
                top_k=config.rag.top_k,
            )
            print(f"RAG retriever loaded ({retriever.collection.count()} chunks)")
        except Exception as e:
            print(f"Warning: Could not load RAG index: {e}")
            print("RAG setups will be skipped.")
    else:
        print("Warning: No RAG index found at data/chroma_db. Run build_rag_index.py first.")
        print("RAG setups will be skipped.")

    # Build setups
    setups = build_setups(config, retriever=retriever)

    # Filter out RAG setups if no retriever
    if retriever is None:
        setups = [s for s in setups if not s.use_rag]
        print(f"Running {len(setups)} setups (RAG setups skipped).")

    # Run
    store = ResultsStore(config.results_dir)
    run_experiment(config, questions, setups, store)

    print(f"\nExperiment complete! Results saved to {config.results_dir}/{config.name}.jsonl")


if __name__ == "__main__":
    main()
