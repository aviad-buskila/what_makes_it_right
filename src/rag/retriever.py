from __future__ import annotations

import re
from collections import Counter

import chromadb

from src.llm.embeddings import get_embedder


class Retriever:
    """Retrieve relevant text chunks from ChromaDB given a query."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "medqa_textbooks",
        embedding_model: str = "nomic-embed-text",
        embedding_backend: str = "ollama",
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
        self.embedding_backend = embedding_backend
        self._embedder = get_embedder(
            model=embedding_model, backend=embedding_backend, timeout=timeout_per_query
        )
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

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string with the configured embedder backend."""
        return self._embedder.embed_query(text)

    def fetch_dense_candidates(
        self, query_text: str, n_candidates: int = 50
    ) -> list[tuple[str, float]]:
        """Return up to ``n_candidates`` (chunk, cosine_distance) pairs, ascending.

        Pure dense retrieval with no distance cutoff and no rerank — this is the
        raw material a retrieval cache stores so that any ``top_k`` /
        ``max_distance`` can be replayed offline via :func:`rank_candidates`.
        """
        embedding = self.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, n_candidates),
            include=["documents", "distances"],
        )
        documents = results["documents"][0] if results["documents"] else []
        distances = (
            results["distances"][0] if results.get("distances")
            else [0.0] * len(documents)
        )
        pairs = [(doc, float(dist)) for doc, dist in zip(documents, distances)]
        pairs.sort(key=lambda p: p[1])
        return pairs

    def query(self, question_text: str, top_k: int | None = None) -> list[str]:
        """Return the top-K most relevant text chunks for a question.

        retrieval_mode:
        - fast: dense retrieval + distance cutoff only
        - balanced: dense retrieval + lexical fusion rerank
        - best: dense + lightweight lexical pre-candidates + fusion rerank
        """
        k = top_k or self.top_k
        embedding = self.embed_query(question_text)
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


def rank_candidates(
    query_text: str,
    candidates: list[tuple[str, float]],
    top_k: int,
    max_distance: float,
    rerank_alpha: float = 0.6,
    min_lexical_overlap: float = 0.0,
    mode: str = "balanced",
) -> list[str]:
    """Select the final top-K chunks from a cached dense candidate pool.

    Pure function (no embedding / DB access) that replays the retriever's
    distance cutoff and hybrid dense+lexical fusion over a pre-fetched
    ``candidates`` list of ``(chunk, cosine_distance)`` pairs. This is what makes
    offline tuning of ``top_k`` and ``max_distance`` possible from a cache.

    Note: unlike the live ``Retriever.query`` in ``balanced``/``best`` mode, this
    does not inject lexical-only candidates from the global corpus pool — it
    ranks within the dense candidate pool, which is what a cache contains.
    """
    close = [(doc, dist) for doc, dist in candidates if dist <= max_distance]
    if not close:
        return []
    if mode == "fast":
        return [doc for doc, _ in close[:top_k]]

    dense_scores = [1.0 - dist for _, dist in close]
    max_dense = max(max(dense_scores, default=1.0), 1e-9)
    query_tokens = _token_set(query_text)

    ranked: list[tuple[float, str]] = []
    for (doc, _dist), dense_raw in zip(close, dense_scores):
        overlap = _bm25ish_score(query_tokens, _token_set(doc))
        dense = dense_raw / max_dense
        score = (rerank_alpha * dense) + ((1.0 - rerank_alpha) * overlap)
        if overlap >= min_lexical_overlap or dense > 0:
            ranked.append((score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


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
