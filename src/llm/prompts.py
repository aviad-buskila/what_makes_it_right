from __future__ import annotations

from src.dataset.loader import Question

_DOMAIN_PERSONA = {
    "medical": "medical expert",
    "cybersecurity": (
        "cybersecurity professional with broad expertise in information security "
        "principles, network security, cryptography, and security standards "
        "(CISSP/Security+ level)"
    ),
}

_DOMAIN_REFERENCE_LABEL = {
    "medical": "medical reference passages",
    "cybersecurity": (
        "cybersecurity reference passages "
        "(MITRE ATT&CK, CWE, NIST SP 800-53, NIST SP 800-63B, OWASP, Wikipedia)"
    ),
}

BASE_PROMPT = """\
You are a {persona}. Answer the following multiple-choice question.
Your response MUST start with exactly one letter — A, B, C, or D — on its own line, followed by a brief explanation.

Question: {question}
{options}

Answer:"""

RAG_PROMPT = """\
You are a {persona}.
The following {reference_label} may or may not be relevant to the question below.
Use them only if they directly support your reasoning. If they are not relevant, ignore them and rely on your own knowledge.

{context}

Answer the following multiple-choice question.
Your response MUST start with exactly one letter — A, B, C, or D — on its own line, followed by a brief explanation.

Question: {question}
{options}

Answer:"""


def _format_options(question: Question) -> str:
    return "\n".join(f"{key}) {val}" for key, val in sorted(question.options.items()))


def _persona(domain: str) -> str:
    return _DOMAIN_PERSONA.get(domain, _DOMAIN_PERSONA["medical"])


def _reference_label(domain: str) -> str:
    return _DOMAIN_REFERENCE_LABEL.get(domain, _DOMAIN_REFERENCE_LABEL["medical"])


def build_base_prompt(question: Question, domain: str = "medical") -> str:
    return BASE_PROMPT.format(
        persona=_persona(domain),
        question=question.question_text,
        options=_format_options(question),
    )


def build_rag_prompt(
    question: Question,
    context_chunks: list[str],
    domain: str = "medical",
) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return RAG_PROMPT.format(
        persona=_persona(domain),
        reference_label=_reference_label(domain),
        context=context,
        question=question.question_text,
        options=_format_options(question),
    )
