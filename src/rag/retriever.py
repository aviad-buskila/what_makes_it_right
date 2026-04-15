from __future__ import annotations

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
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = embedding_model
        self.top_k = top_k

    def query(self, question_text: str, top_k: int | None = None) -> list[str]:
        """Return the top-K most relevant text chunks for a question."""
        k = top_k or self.top_k
        embedding = generate_embedding(question_text, model=self.embedding_model)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
        )
        return results["documents"][0] if results["documents"] else []
