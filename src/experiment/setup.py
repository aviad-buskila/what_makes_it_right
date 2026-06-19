from __future__ import annotations

from dataclasses import dataclass

from src.dataset.loader import Question
from src.experiment.config import ModelConfig, RagConfig
from src.llm.client import generate
from src.llm.parser import extract_answer, normalize_answer_letter
from src.llm.prompts import build_base_prompt, build_rag_prompt
from src.rag.oracle import build_oracle_context
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
        answer_retry_attempts: int = 2,
        domain: str = "medical",
        oracle_mode: str | None = None,
    ):
        self.model_config = model_config
        self.use_rag = use_rag
        self.retriever = retriever
        self.rag_config = rag_config
        self.temperature = temperature
        self.timeout_per_query = timeout_per_query
        self.answer_retry_attempts = max(1, answer_retry_attempts)
        self.domain = domain
        self.oracle_mode = oracle_mode

        if oracle_mode:
            suffix = f"+Oracle:{oracle_mode}"
        elif use_rag:
            suffix = "+RAG"
        else:
            suffix = ""
        self.name = f"{model_config.name}{suffix}@t{self.temperature:g}"

    def answer(self, question: Question, pre_retrieved_chunks: list[str] | None = None, repetition: int = 0) -> SetupResult:
        """Send a question to this setup and return the result.

        If ``pre_retrieved_chunks`` is provided it is used directly (no retrieval
        call is made), allowing the caller to share one retrieval result across
        multiple setups for the same question.
        """
        context_chunks: list[str] | None = None

        if self.oracle_mode:
            # Oracle setups ignore shared retrieval and build privileged context.
            if self.oracle_mode == "privileged_query":
                if self.retriever is not None:
                    top_k = self.rag_config.top_k if self.rag_config else 5
                    gold_letter = (question.correct_answer or "").strip().upper()
                    gold_text = question.options.get(gold_letter, "")
                    query_text = f"{question.question_text}\n{gold_text}".strip()
                    context_chunks = self.retriever.query(query_text, top_k=top_k)
                else:
                    context_chunks = None
            else:
                context_chunks = build_oracle_context(question, mode=self.oracle_mode)
            if context_chunks:
                prompt = build_rag_prompt(question, context_chunks, domain=self.domain)
            else:
                prompt = build_base_prompt(question, domain=self.domain)
        elif self.use_rag:
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

        seed = hash((question.id, self.name, repetition)) & 0x7FFFFFFF
        response_text = ""
        latency = 0.0
        answer = None
        current_prompt = prompt
        for attempt in range(1, self.answer_retry_attempts + 1):
            current_seed = seed ^ (attempt * 0x9E37)
            response_text, latency = generate(
                model_id=self.model_config.ollama_id,
                prompt=current_prompt,
                temperature=self.temperature,
                timeout=self.timeout_per_query,
                seed=current_seed,
            )
            answer = extract_answer(response_text)
            if answer is not None:
                break
            if attempt < self.answer_retry_attempts:
                current_prompt = (
                    f"{prompt}\n\nIMPORTANT: Invalid output detected. "
                    "Return only JSON exactly like {\"answer\":\"A\"}."
                )

        gold_answer = normalize_answer_letter(question.correct_answer)
        is_correct = answer is not None and gold_answer is not None and answer == gold_answer

        return SetupResult(
            response=response_text,
            extracted_answer=answer,
            is_correct=is_correct,
            latency_seconds=latency,
            retrieved_context=context_chunks,
        )
