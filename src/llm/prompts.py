from __future__ import annotations

from src.dataset.loader import Question

BASE_PROMPT = """\
You are a medical expert. Answer the following multiple-choice question.
Respond with ONLY the letter of the correct answer (A, B, C, or D) on the first line, followed by a brief explanation.

Question: {question}
{options}

Answer:"""

RAG_PROMPT = """\
The following medical reference passages may or may not be relevant to the question below.
Use them only if they directly support your reasoning. If they are not relevant, ignore them and rely on your own knowledge.

{context}

Answer the following multiple-choice question.
Respond with ONLY the letter of the correct answer (A, B, C, or D) on the first line, followed by a brief explanation.

Question: {question}
{options}

Answer:"""


def _format_options(question: Question) -> str:
    return "\n".join(f"{key}) {val}" for key, val in sorted(question.options.items()))


def build_base_prompt(question: Question) -> str:
    return BASE_PROMPT.format(
        question=question.question_text,
        options=_format_options(question),
    )


def build_rag_prompt(question: Question, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return RAG_PROMPT.format(
        context=context,
        question=question.question_text,
        options=_format_options(question),
    )
