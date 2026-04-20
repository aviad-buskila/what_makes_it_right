from __future__ import annotations

from tqdm import tqdm

from src.dataset.loader import Question
from src.experiment.config import ExperimentConfig
from src.experiment.setup import ExperimentSetup
from src.llm.client import check_health, ensure_model
from src.rag.retriever import Retriever
from src.storage.results import ExperimentResult, ResultsStore


def build_setups(config: ExperimentConfig, retriever: Retriever | None = None) -> list[ExperimentSetup]:
    """Create all experimental setups (each model with and without RAG)."""
    setups: list[ExperimentSetup] = []
    for model_config in config.models:
        setups.append(ExperimentSetup(
            model_config=model_config,
            use_rag=False,
            temperature=config.temperature,
            timeout_per_query=config.timeout_per_query,
            answer_retry_attempts=config.answer_retry_attempts,
            domain=config.domain,
        ))
        setups.append(ExperimentSetup(
            model_config=model_config,
            use_rag=True,
            retriever=retriever,
            rag_config=config.rag,
            temperature=config.temperature,
            timeout_per_query=config.timeout_per_query,
            answer_retry_attempts=config.answer_retry_attempts,
            domain=config.domain,
        ))
    return setups


def run_experiment(
    config: ExperimentConfig,
    questions: list[Question],
    setups: list[ExperimentSetup],
    store: ResultsStore,
    retriever: Retriever | None = None,
) -> None:
    """Run the full experiment.

    Loop order: question → retrieve chunks once → model pairs (base + RAG) →
    repetitions.  Each question's chunks are retrieved exactly once and shared
    across all RAG setups, avoiding redundant embedding calls.
    """
    if not check_health():
        raise RuntimeError("Ollama is not running. Start it with 'ollama serve'.")

    experiment_name = config.name
    completed = store.get_completed_keys(experiment_name)

    # Group setups into {model_name: {"base": setup, "rag": setup}} preserving
    # model order so we can ensure each model once before the question loop.
    model_pairs: dict[str, dict[str, ExperimentSetup]] = {}
    for setup in setups:
        model_name = setup.model_config.name
        if model_name not in model_pairs:
            model_pairs[model_name] = {}
        key = "rag" if setup.use_rag else "base"
        model_pairs[model_name][key] = setup

    # Pull / verify all required models once up front.
    seen_model_ids: set[str] = set()
    for setup in setups:
        mid = setup.model_config.ollama_id
        if mid not in seen_model_ids:
            ensure_model(mid)
            seen_model_ids.add(mid)

    total = len(questions) * config.repetitions * len(setups)
    skipped = len(completed)
    if skipped > 0:
        print(f"Resuming experiment: {skipped}/{total} already completed.")

    with tqdm(total=total - skipped, desc="experiment") as pbar:
        retrieval_calls = 0
        retrieval_failures = 0
        for question in questions:
            # Retrieve chunks once for this question and reuse across all RAG setups.
            chunks: list[str] | None = None
            if retriever is not None:
                try:
                    retrieval_calls += 1
                    chunks = retriever.query(question.question_text)
                except Exception as e:
                    retrieval_failures += 1
                    print(f"\nRetrieval failed for {question.id}: {e}. RAG setups will run without context.")

            for pair in model_pairs.values():
                for variant in ("base", "rag"):
                    setup = pair.get(variant)
                    if setup is None:
                        continue

                    for rep in range(config.repetitions):
                        key = (question.id, setup.name, rep)
                        if key in completed:
                            continue

                        try:
                            pre_chunks = chunks if variant == "rag" else None
                            result = setup.answer(question, pre_retrieved_chunks=pre_chunks, repetition=rep)
                            record = ExperimentResult(
                                question_id=question.id,
                                setup_name=setup.name,
                                model_name=setup.model_config.name,
                                has_rag=setup.use_rag,
                                repetition=rep,
                                model_response=result.response,
                                extracted_answer=result.extracted_answer,
                                correct_answer=question.correct_answer,
                                is_correct=result.is_correct,
                                latency_seconds=result.latency_seconds,
                                retrieved_context=result.retrieved_context,
                            )
                            store.append(record, experiment_name)
                            pbar.update(1)
                        except Exception as e:
                            print(f"\nError on {question.id} / {setup.name} / rep {rep}: {e}")
                            if config.record_failures:
                                failure_record = ExperimentResult(
                                    question_id=question.id,
                                    setup_name=setup.name,
                                    model_name=setup.model_config.name,
                                    has_rag=setup.use_rag,
                                    repetition=rep,
                                    model_response="",
                                    extracted_answer=None,
                                    correct_answer=question.correct_answer,
                                    is_correct=False,
                                    latency_seconds=0.0,
                                    succeeded=False,
                                    error_message=str(e),
                                    retrieved_context=None,
                                )
                                store.append(failure_record, experiment_name)
                            continue
    if retriever is not None:
        print(
            "\nRetrieval instrumentation:"
            f" calls={retrieval_calls}, questions={len(questions)}, failures={retrieval_failures}"
        )
