from __future__ import annotations

import re
from collections import Counter

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
        retrieval_mode: str = "balanced",
        dense_multiplier: int = 4,
        lexical_multiplier: int = 4,
        rerank_alpha: float = 0.6,
        min_lexical_overlap: float = 0.01,
        lexical_pool_size: int = 2000,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.timeout_per_query = timeout_per_query
        self.max_distance = max_distance
        self.retrieval_mode = retrieval_mode
        self.dense_multiplier = max(1, dense_multiplier)
        self.lexical_multiplier = max(1, lexical_multiplier)
        self.rerank_alpha = max(0.0, min(1.0, rerank_alpha))
        self.min_lexical_overlap = max(0.0, min(1.0, min_lexical_overlap))
        self.lexical_pool_size = max(100, lexical_pool_size)
        self._lexical_docs: list[str] = []
        if self.retrieval_mode in {"balanced", "best"}:
            self._prime_lexical_pool()

    def query(self, question_text: str, top_k: int | None = None) -> list[str]:
        """Return the top-K most relevant text chunks for a question.

        retrieval_mode:
        - fast: dense retrieval + distance cutoff only
        - balanced: dense retrieval + lexical fusion rerank
        - best: dense + lightweight lexical pre-candidates + fusion rerank
        """
        k = top_k or self.top_k
        embedding = generate_embedding(
            question_text,
            model=self.embedding_model,
            timeout=self.timeout_per_query,
        )
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(k * self.dense_multiplier, k),
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

        if self.retrieval_mode == "fast":
            return list(documents[:k])

        lexical_candidates = self._lexical_candidates(question_text, k)
        dense_docs = list(documents)
        dense_scores = [1.0 - float(d) for d in distances]
        max_dense = max(dense_scores) if dense_scores else 1.0
        max_dense = max(max_dense, 1e-9)

        query_tokens = _token_set(question_text)
        all_docs = list(dict.fromkeys(dense_docs + lexical_candidates))
        ranked: list[tuple[float, str]] = []
        for doc in all_docs:
            overlap = _bm25ish_score(query_tokens, _token_set(doc))
            dense = 0.0
            if doc in dense_docs:
                idx = dense_docs.index(doc)
                dense = dense_scores[idx] / max_dense
            score = (self.rerank_alpha * dense) + ((1.0 - self.rerank_alpha) * overlap)
            if overlap >= self.min_lexical_overlap or dense > 0:
                ranked.append((score, doc))

        if not ranked:
            return []

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:k]]

    def build_query_text(self, question_text: str, options: dict[str, str], mode: str) -> str:
        """Build retrieval query text from question/options according to mode."""
        key = (mode or "question_only").strip().lower()
        if key == "question_only":
            return question_text
        if key == "question_plus_options":
            opts = " ".join(
                f"{letter}) {text}" for letter, text in sorted(options.items())
            )
            return f"{question_text}\nOptions: {opts}"
        if key == "question_plus_top_terms":
            terms = []
            for _, text in sorted(options.items()):
                terms.extend(_token_set(text))
            top_terms = [t for t, _ in Counter(terms).most_common(12)]
            suffix = " ".join(top_terms)
            return f"{question_text}\nOption key terms: {suffix}" if suffix else question_text
        return question_text

    def _prime_lexical_pool(self) -> None:
        """Cache a subset of documents for lexical candidate generation."""
        try:
            rows = self.collection.get(limit=self.lexical_pool_size, include=["documents"])
        except Exception:
            self._lexical_docs = []
            return
        docs = rows.get("documents", []) if isinstance(rows, dict) else []
        self._lexical_docs = [d for d in docs if isinstance(d, str) and d.strip()]

    def _lexical_candidates(self, question_text: str, k: int) -> list[str]:
        if not self._lexical_docs:
            return []
        q_tokens = _token_set(question_text)
        if not q_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for doc in self._lexical_docs:
            overlap = _bm25ish_score(q_tokens, _token_set(doc))
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        limit = max(k * self.lexical_multiplier, k)
        return [doc for _, doc in scored[:limit]]


def _token_set(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z]{4,}", text.lower()))
    stopwords = {
        "with", "from", "that", "this", "which", "into", "about", "after", "before",
        "would", "could", "should", "there", "their", "they", "them", "then", "than",
        "have", "has", "were", "been", "being", "where", "when", "what", "your",
        "question", "answer", "following", "option", "options",
    }
    return {t for t in tokens if t not in stopwords}


def _bm25ish_score(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    common = a & b
    if not common:
        return 0.0
    tf = Counter(b)
    score = 0.0
    for token in common:
        score += 1.0 / (1.0 + tf[token])
    return score / max(len(a), 1)
