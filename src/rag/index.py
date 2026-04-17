from __future__ import annotations

import random
from pathlib import Path

import chromadb
from datasets import load_dataset
from tqdm import tqdm

from src.llm.client import generate_embedding


def load_textbook_corpus(
    cache_dir: str | None = None,
    max_corpus_size: int = 25_000,
    random_seed: int = 42,
) -> list[dict[str, str]]:
    """Load the MedQA textbook corpus from HuggingFace.

    Returns list of dicts with 'id' and 'text' keys.
    """
    try:
        ds = load_dataset(
            "bigbio/med_qa",
            "med_qa_en_source",
            split="train",
            cache_dir=cache_dir,
        )

        corpus = []
        for idx, row in enumerate(ds):
            # The bigbio format has document-level entries
            text = row.get("document", {}).get("text", "") if isinstance(row.get("document"), dict) else ""
            if not text:
                # Fallback: try direct text fields
                text = row.get("text", "") or row.get("passage", "") or str(row)
            if text.strip():
                corpus.append({"id": f"textbook_{idx}", "text": text.strip()})

        return corpus
    except RuntimeError as exc:
        # datasets>=4 no longer supports script-based dataset loaders used by bigbio/med_qa.
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        # Fallback: use medmcqa medical explanations as knowledge corpus.
        # These are concise, exam-oriented medical explanations covering anatomy,
        # pharmacology, pathology, etc. — real knowledge, no answer labels.
        ds = load_dataset(
            "openlifescienceai/medmcqa",
            split="train",
            cache_dir=cache_dir,
        )
        corpus = []
        for idx, row in enumerate(ds):
            exp = str(row.get("exp") or "").strip()
            if not exp or len(exp) < 50:
                continue
            subject = str(row.get("subject_name") or "").strip()
            topic = str(row.get("topic_name") or "").strip()

            header = " — ".join(filter(None, [subject, topic]))
            text = f"{header}\n\n{exp}" if header else exp
            corpus.append({"id": f"medmcqa_{idx}", "text": text})

        if len(corpus) > max_corpus_size:
            rng = random.Random(random_seed)
            corpus = rng.sample(corpus, max_corpus_size)

        return corpus


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks while preserving word boundaries."""
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    words = clean.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 accounts for a joining space when chunk is non-empty.
        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > chunk_size:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

            # Rebuild overlap context from tail words of previous chunk.
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                next_len = overlap_len + len(w) + (1 if overlap_words else 0)
                if next_len > overlap:
                    break
                overlap_words.append(w)
                overlap_len = next_len
            current = list(reversed(overlap_words))
            current_len = len(" ".join(current)) if current else 0

        if current:
            current_len += 1 + len(word)
        else:
            current_len = len(word)
        current.append(word)

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def build_index(
    persist_dir: str = "data/chroma_db",
    collection_name: str = "medqa_textbooks",
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    cache_dir: str | None = None,
    max_corpus_size: int = 25_000,
) -> chromadb.Collection:
    """Build a ChromaDB index from the textbook corpus.

    Embeds chunks using Ollama and persists to disk.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists, to rebuild
    try:
        client.delete_collection(collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print("Loading textbook corpus...")
    corpus = load_textbook_corpus(cache_dir=cache_dir, max_corpus_size=max_corpus_size)
    print(f"Loaded {len(corpus)} documents from corpus.")

    print("Chunking and embedding...")
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_embeddings: list[list[float]] = []
    chunk_idx = 0
    batch_size = 100

    for doc in tqdm(corpus, desc="Processing documents"):
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for chunk in chunks:
            embedding = generate_embedding(chunk, model=embedding_model)
            batch_ids.append(f"chunk_{chunk_idx}")
            batch_docs.append(chunk)
            batch_embeddings.append(embedding)
            chunk_idx += 1

            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                )
                batch_ids, batch_docs, batch_embeddings = [], [], []

    # Flush remaining
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
        )

    print(f"Index built: {chunk_idx} chunks in collection '{collection_name}'")
    return collection
