from __future__ import annotations

from tqdm import tqdm

from src.dataset.loader import Question
from src.experiment.config import ExperimentConfig
from src.experiment.setup import ExperimentSetup
from src.llm.client import check_health, ensure_model
from src.storage.results import ExperimentResult, ResultsStore


def build_setups(config: ExperimentConfig, retriever=None) -> list[ExperimentSetup]:
    """Create all 6 experimental setups (each model with and without RAG)."""
    setups: list[ExperimentSetup] = []
    for model_config in config.models:
        # Without RAG
        setups.append(ExperimentSetup(
            model_config=model_config,
            use_rag=False,
            temperature=config.temperature,
            timeout_per_query=config.timeout_per_query,
        ))
        # With RAG
        setups.append(ExperimentSetup(
            model_config=model_config,
            use_rag=True,
            retriever=retriever,
            rag_config=config.rag,
            temperature=config.temperature,
            timeout_per_query=config.timeout_per_query,
        ))
    return setups


def run_experiment(
    config: ExperimentConfig,
    questions: list[Question],
    setups: list[ExperimentSetup],
    store: ResultsStore,
) -> None:
    """Run the full experiment: setups x questions x repetitions.

    Supports resuming from checkpoint via ResultsStore.
    """
    if not check_health():
        raise RuntimeError("Ollama is not running. Start it with 'ollama serve'.")

    experiment_name = config.name
    completed = store.get_completed_keys(experiment_name)
    total = len(setups) * len(questions) * config.repetitions
    skipped = len(completed)

    if skipped > 0:
        print(f"Resuming experiment: {skipped}/{total} already completed.")

    # Group by setup to minimize model swaps
    for setup in setups:
        print(f"\n{'='*60}")
        print(f"Setup: {setup.name} (model: {setup.model_config.ollama_id})")
        print(f"{'='*60}")

        ensure_model(setup.model_config.ollama_id)

        setup_total = len(questions) * config.repetitions
        setup_skipped = sum(
            1 for q in questions for r in range(config.repetitions)
            if (q.id, setup.name, r) in completed
        )

        with tqdm(total=setup_total - setup_skipped, desc=setup.name) as pbar:
            for question in questions:
                for rep in range(config.repetitions):
                    key = (question.id, setup.name, rep)
                    if key in completed:
                        continue

                    try:
                        result = setup.answer(question)
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
                        print(f"\nError on {question.id}, rep {rep}: {e}")
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
