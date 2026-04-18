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
        domain: str = "medical",
    ):
        self.model_config = model_config
        self.use_rag = use_rag
        self.retriever = retriever
        self.rag_config = rag_config
        self.temperature = temperature
        self.timeout_per_query = timeout_per_query
        self.domain = domain

        rag_suffix = "+RAG" if use_rag else ""
        self.name = f"{model_config.name}{rag_suffix}"

    def answer(self, question: Question, pre_retrieved_chunks: list[str] | None = None) -> SetupResult:
        """Send a question to this setup and return the result.

        If ``pre_retrieved_chunks`` is provided it is used directly (no retrieval
        call is made), allowing the caller to share one retrieval result across
        multiple setups for the same question.
        """
        context_chunks: list[str] | None = None

        if self.use_rag:
            if pre_retrieved_chunks is not None:
                context_chunks = pre_retrieved_chunks
            elif self.retriever:
                top_k = self.rag_config.top_k if self.rag_config else 5
                context_chunks = self.retriever.query(question.question_text, top_k=top_k)
            # Fall back to base prompt when retrieval finds nothing above the
            # distance threshold — injecting empty / irrelevant context hurts.
            if context_chunks:
                prompt = build_rag_prompt(question, context_chunks, domain=self.domain)
            else:
                prompt = build_base_prompt(question, domain=self.domain)
        else:
            prompt = build_base_prompt(question, domain=self.domain)

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
