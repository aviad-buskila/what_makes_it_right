from __future__ import annotations

from dataclasses import dataclass

from src.dataset.loader import Question
from src.experiment.config import ModelConfig, RagConfig
from src.llm.client import generate
from src.llm.parser import extract_answer
from src.llm.prompts import build_base_prompt, build_rag_prompt
from src.rag.retriever import Retriever


@dataclass
class SetupResult:
    response: str
    extracted_answer: str | None
    is_correct: bool
    latency_seconds: float
    retrieved_context: list[str] | None


class ExperimentSetup:
    """A single experimental condition: model + optional RAG."""

    def __init__(
        self,
        model_config: ModelConfig,
        use_rag: bool,
        retriever: Retriever | None = None,
        rag_config: RagConfig | None = None,
        temperature: float = 0.7,
        timeout_per_query: int = 60,
    ):
        self.model_config = model_config
        self.use_rag = use_rag
        self.retriever = retriever
        self.rag_config = rag_config
        self.temperature = temperature
        self.timeout_per_query = timeout_per_query

        rag_suffix = "+RAG" if use_rag else ""
        self.name = f"{model_config.name}{rag_suffix}"

    def answer(self, question: Question) -> SetupResult:
        """Send a question to this setup and return the result."""
        context_chunks: list[str] | None = None

        if self.use_rag and self.retriever:
            top_k = self.rag_config.top_k if self.rag_config else 5
            context_chunks = self.retriever.query(question.question_text, top_k=top_k)
            prompt = build_rag_prompt(question, context_chunks)
        else:
            prompt = build_base_prompt(question)

        response_text, latency = generate(
            model_id=self.model_config.ollama_id,
            prompt=prompt,
            temperature=self.temperature,
            timeout=self.timeout_per_query,
        )

        answer = extract_answer(response_text)
        is_correct = answer is not None and answer == question.correct_answer

        return SetupResult(
            response=response_text,
            extracted_answer=answer,
            is_correct=is_correct,
            latency_seconds=latency,
            retrieved_context=context_chunks,
        )
