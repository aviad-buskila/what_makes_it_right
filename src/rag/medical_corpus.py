"""Authoritative, US-aligned medical RAG corpora (MIRAGE / MedRAG).

These are stronger, better-fitting knowledge bases than the MedMCQA-explanation
proxy, used to test whether the RAG null survives an authoritative corpus.

Subsets (HuggingFace ``MedRAG/<subset>``):
  * ``textbooks``  — the 18 English medical textbooks the USMLE/MedQA questions
    were written from. Best topical fit, US-aligned, textbook-grade. NOTE: because
    MedQA items were authored from these books, this corpus is *RAG-favourable*
    (borderline near-leakage) — ideal for showing RAG **can** help when the corpus
    is well matched, less so as a "fair null" test.
  * ``statpearls`` — StatPearls clinical reference articles. Authoritative,
    US-aligned, and NOT the source of MedQA, so it is the cleaner fit test.
    Requires a manual build (see error message) due to StatPearls licensing.
  * ``pubmed`` / ``wikipedia`` — large general corpora (auto-capped).

Each MedRAG record is already snippet-chunked; ``build_index`` re-chunks only if
a snippet exceeds the configured ``chunk_size``.
"""

from __future__ import annotations

import random

MEDRAG_SUBSETS = ("textbooks", "statpearls", "pubmed", "wikipedia")
_HUGE_SUBSETS = {"pubmed", "wikipedia"}
_HUGE_DEFAULT_CAP = 200_000


def _medrag_row_to_text(row: dict) -> str:
    """Extract document text from a MedRAG row, robust to schema variants.

    Prefers the precombined ``contents`` field MedRAG embeds; falls back to
    ``title`` + ``content``, then to any plain text field.
    """
    contents = str(row.get("contents") or "").strip()
    if contents:
        return contents
    title = str(row.get("title") or "").strip()
    content = str(row.get("content") or row.get("text") or "").strip()
    if title and content:
        return f"{title}. {content}"
    return content or title


def load_medrag_corpus(
    subset: str = "textbooks",
    cache_dir: str | None = None,
    max_corpus_size: int | None = None,
    random_seed: int = 42,
) -> list[dict[str, str]]:
    """Load a MedRAG corpus subset as a list of ``{"id", "text"}`` documents."""
    from datasets import load_dataset  # lazy: keeps the parser importable w/o deps

    key = (subset or "textbooks").strip().lower()
    if key not in MEDRAG_SUBSETS:
        raise ValueError(
            f"Unknown MedRAG subset {subset!r}. Use one of {MEDRAG_SUBSETS}."
        )

    repo = f"MedRAG/{key}"
    try:
        ds = load_dataset(repo, split="train", cache_dir=cache_dir)
    except Exception as exc:
        if key == "statpearls":
            raise RuntimeError(
                "MedRAG/statpearls cannot be auto-downloaded due to StatPearls "
                "licensing. Build the snippets with the MIRAGE StatPearls script "
                "(https://github.com/Teddy-XiongGZ/MedRAG), then load them as a "
                "local dataset, or use subset='textbooks' for an auto-available "
                "US-aligned corpus."
            ) from exc
        raise

    corpus: list[dict[str, str]] = []
    for idx, row in enumerate(ds):
        text = _medrag_row_to_text(row)
        if len(text) < 50:
            continue
        rid = str(row.get("id") or f"{key}_{idx}")
        corpus.append({"id": f"medrag_{key}_{rid}", "text": text})

    cap = max_corpus_size
    if cap is None and key in _HUGE_SUBSETS:
        cap = _HUGE_DEFAULT_CAP
        print(f"[medrag] '{key}' is large; capping to {cap} docs "
              f"(set rag.max_corpus_size to override).")
    if cap is not None and len(corpus) > cap:
        corpus = random.Random(random_seed).sample(corpus, cap)

    return corpus
