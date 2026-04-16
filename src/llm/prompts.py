from __future__ import annotations

from src.dataset.loader import Question

BASE_PROMPT = """\
You are a medical expert. Answer the following multiple-choice question.
Respond with ONLY the letter of the correct answer (A, B, C, or D) on the first line, followed by a brief explanation.

Question: {question}
{options}

Answer:"""

RAG_PROMPT = """\
Use the following medical reference material to help answer the question:

{context}

Now answer the following multiple-choice question.
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
