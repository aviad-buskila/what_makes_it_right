"""Pluggable text embedders for the RAG pipeline.

The default backend is Ollama (unchanged behaviour). Two additional backends
let us test whether the RAG result is an artifact of a single, possibly weak,
embedder:

* ``ollama``  — any Ollama-served embedding model (e.g. ``nomic-embed-text``,
  ``mxbai-embed-large``, ``bge-m3``). Default.
* ``hf``      — any sentence-transformers model (e.g. ``BAAI/bge-large-en-v1.5``).
* ``medcpt``  — MedCPT (Jin et al., 2023), a biomedical dual-encoder trained on
  PubMed search logs. Uses the Query encoder for queries and the Article encoder
  for documents.

All embedders expose :meth:`embed_query` and :meth:`embed_documents`, so a
dual-encoder model can encode queries and passages with different towers while
single-encoder models share one tower.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from src.llm.client import generate_embedding


class Embedder(ABC):
    """Abstract embedder exposing query- and document-side encoding."""

    name: str

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class OllamaEmbedder(Embedder):
    """Embeds via an Ollama-served embedding model (single tower)."""

    def __init__(self, model: str = "nomic-embed-text", timeout: int = 60):
        self.name = model
        self.model = model
        self.timeout = timeout

    def embed_query(self, text: str) -> list[float]:
        return generate_embedding(text, model=self.model, timeout=self.timeout)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class HFEmbedder(Embedder):
    """Embeds via a sentence-transformers model (single tower)."""

    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", device: str | None = None):
        self.name = model
        self.model_name = model
        self._device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "The 'hf' embedding backend requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]


class MedCPTEmbedder(Embedder):
    """MedCPT biomedical dual-encoder (separate query / article towers)."""

    QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
    MAX_LEN = 512

    def __init__(self, device: str | None = None):
        self.name = "medcpt"
        self._device = device
        self._q_tok = self._q_model = None
        self._a_tok = self._a_model = None

    def _load(self, model_id: str):
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "The 'medcpt' embedding backend requires transformers + torch. "
                "Install with: pip install transformers torch"
            ) from exc
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        if self._device:
            model = model.to(self._device)
        model.eval()
        return tok, model

    def _encode(self, texts: list[str], is_query: bool) -> list[list[float]]:
        import torch

        if is_query:
            if self._q_model is None:
                self._q_tok, self._q_model = self._load(self.QUERY_MODEL)
            tok, model = self._q_tok, self._q_model
        else:
            if self._a_model is None:
                self._a_tok, self._a_model = self._load(self.ARTICLE_MODEL)
            tok, model = self._a_tok, self._a_model

        with torch.no_grad():
            encoded = tok(
                texts,
                truncation=True,
                padding=True,
                max_length=self.MAX_LEN,
                return_tensors="pt",
            )
            if self._device:
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
            # MedCPT represents a sequence by the first-token ([CLS]) embedding.
            embeds = model(**encoded).last_hidden_state[:, 0, :]
        return [row.tolist() for row in embeds.cpu()]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], is_query=True)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, is_query=False)


@lru_cache(maxsize=8)
def get_embedder(model: str = "nomic-embed-text", backend: str = "ollama",
                 timeout: int = 60) -> Embedder:
    """Factory for embedders. Cached so repeated calls reuse loaded weights.

    ``backend`` is one of ``ollama`` (default), ``hf``, ``medcpt``.
    """
    key = (backend or "ollama").strip().lower()
    if key == "ollama":
        return OllamaEmbedder(model=model, timeout=timeout)
    if key == "hf":
        return HFEmbedder(model=model)
    if key == "medcpt":
        return MedCPTEmbedder()
    raise ValueError(
        f"Unknown embedding backend {backend!r}. Use 'ollama', 'hf', or 'medcpt'."
    )
