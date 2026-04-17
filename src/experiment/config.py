from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str
    ollama_id: str
    category: str  # "large_general", "small_general", "small_domain"


@dataclass
class RagConfig:
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "nomic-embed-text"
    collection_name: str = "medqa_textbooks"
    max_distance: float = 0.27  # cosine distance cutoff; chunks above this are discarded


@dataclass
class ExperimentConfig:
    name: str = "medical_mcq_comparison"
    repetitions: int = 5
    max_questions: int | None = None
    random_seed: int = 42
    temperature: float = 0.7
    timeout_per_query: int = 60
    record_failures: bool = True
    models: list[ModelConfig] = field(default_factory=list)
    rag: RagConfig = field(default_factory=RagConfig)
    results_dir: str = "results"


def load_config(path: str | Path = "config.yaml") -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    exp = raw.get("experiment", {})
    models = [ModelConfig(**m) for m in raw.get("models", [])]
    rag_raw = raw.get("rag", {})
    storage = raw.get("storage", {})

    return ExperimentConfig(
        name=exp.get("name", "medical_mcq_comparison"),
        repetitions=exp.get("repetitions", 5),
        max_questions=exp.get("max_questions"),
        random_seed=exp.get("random_seed", 42),
        temperature=exp.get("temperature", 0.7),
        timeout_per_query=exp.get("timeout_per_query", 60),
        record_failures=exp.get("record_failures", True),
        models=models,
        rag=RagConfig(**rag_raw) if rag_raw else RagConfig(),
        results_dir=storage.get("results_dir", "results"),
    )
