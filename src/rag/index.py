from __future__ import annotations

from pathlib import Path

import chromadb
from datasets import load_dataset
from tqdm import tqdm

from src.llm.client import generate_embedding


def load_textbook_corpus(cache_dir: str | None = None) -> list[dict[str, str]]:
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

        # Fallback: build a retrieval corpus from MedQA train questions/options.
        ds = load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            split="train",
            cache_dir=cache_dir,
        )
        corpus = []
        for idx, row in enumerate(ds):
            # Avoid injecting A/B/C/D option lists into retrieval context because
            # they can distract answer selection in downstream prompts.
            question_text = str(row.get("question", "")).strip()
            answer_text = str(row.get("answer", "")).strip()
            meta_info = str(row.get("meta_info", "")).strip()

            sections = []
            if question_text:
                sections.append(f"Clinical prompt: {question_text}")
            if answer_text:
                sections.append(f"Reference diagnosis/concept: {answer_text}")
            if meta_info:
                sections.append(f"Reference notes: {meta_info}")

            text = "\n".join(sections)
            if text.strip():
                corpus.append({"id": f"medqa_train_{idx}", "text": text.strip()})
        return corpus


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def build_index(
    persist_dir: str = "data/chroma_db",
    collection_name: str = "medqa_textbooks",
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    cache_dir: str | None = None,
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
    corpus = load_textbook_corpus(cache_dir=cache_dir)
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
