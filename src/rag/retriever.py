from __future__ import annotations

import re

import chromadb

from src.llm.client import generate_embedding


class Retriever:
    """Retrieve relevant text chunks from ChromaDB given a query."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "medqa_textbooks",
        embedding_model: str = "nomic-embed-text",
        top_k: int = 5,
        timeout_per_query: int = 60,
        max_distance: float = 0.27,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.timeout_per_query = timeout_per_query
        self.max_distance = max_distance

    def query(self, question_text: str, top_k: int | None = None) -> list[str]:
        """Return the top-K most relevant text chunks for a question."""
        k = top_k or self.top_k
        embedding = generate_embedding(
            question_text,
            model=self.embedding_model,
            timeout=self.timeout_per_query,
        )
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(k * 3, k),
            include=["documents", "distances"],
        )
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)
        if not documents:
            return []

        # Drop chunks that are too far away — they are noise, not knowledge.
        # Returning an empty list signals the caller to fall back to the base prompt.
        close_pairs = [
            (doc, dist) for doc, dist in zip(documents, distances)
            if float(dist) <= self.max_distance
        ]
        if not close_pairs:
            return []
        documents, distances = zip(*close_pairs)

        query_tokens = _token_set(question_text)
        ranked: list[tuple[float, str]] = []
        min_overlap = 0.02
        for doc, distance in zip(documents, distances):
            overlap = _jaccard_overlap(query_tokens, _token_set(doc))
            # Prefer semantic neighbors but penalize weak lexical overlap.
            score = (1.0 - float(distance)) + (0.35 * overlap)
            if overlap >= min_overlap:
                ranked.append((score, doc))

        if not ranked:
            # Fallback: no strong lexical support, avoid flooding model with weak context.
            return documents[:1]

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:k]]


def _token_set(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z]{4,}", text.lower()))
    stopwords = {
        "with", "from", "that", "this", "which", "into", "about", "after", "before",
        "would", "could", "should", "there", "their", "they", "them", "then", "than",
        "have", "has", "were", "been", "being", "where", "when", "what", "your",
        "question", "answer", "following", "option", "options",
    }
    return {t for t in tokens if t not in stopwords}


def _jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
