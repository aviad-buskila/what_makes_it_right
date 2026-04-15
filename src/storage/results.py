from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentResult:
    question_id: str
    setup_name: str
    model_name: str
    has_rag: bool
    repetition: int
    model_response: str
    extracted_answer: str | None
    correct_answer: str
    is_correct: bool
    latency_seconds: float
    succeeded: bool = True
    error_message: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retrieved_context: list[str] | None = None


class ResultsStore:
    """Append-only JSONL storage for experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_results_path(self, experiment_name: str) -> Path:
        return self.results_dir / f"{experiment_name}.jsonl"

    def append(self, result: ExperimentResult, experiment_name: str) -> None:
        path = self.get_results_path(experiment_name)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def load(self, experiment_name: str) -> pd.DataFrame:
        path = self.get_results_path(experiment_name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_json(path, lines=True)

    def get_completed_keys(self, experiment_name: str) -> set[tuple[str, str, int]]:
        """Return set of (question_id, setup_name, repetition) already completed.

        Used for resuming experiments.
        """
        path = self.get_results_path(experiment_name)
        if not path.exists():
            return set()
        completed = set()
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                completed.add((r["question_id"], r["setup_name"], r["repetition"]))
        return completed
