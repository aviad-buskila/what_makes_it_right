from __future__ import annotations

import random
import re
from pathlib import Path

import chromadb
from datasets import load_dataset
from tqdm import tqdm

from src.llm.embeddings import get_embedder
from src.rag.cyber_corpus import load_cybersecurity_corpus


def load_corpus_by_source(
    source: str,
    cache_dir: str | None = None,
    max_corpus_size: int | None = None,
    random_seed: int = 42,
    include: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    """Dispatch to a corpus loader by name.

    Supported sources:
      - ``medqa``         → MedQA textbook / MedMCQA explanations fallback
      - ``cybersecurity`` → MITRE ATT&CK + CWE + NIST SP 800-53 + OWASP
    """
    key = source.strip().lower()
    if key == "medqa":
        corpus = load_textbook_corpus(
            cache_dir=cache_dir,
            max_corpus_size=max_corpus_size or 25_000,
            random_seed=random_seed,
        )
        return _dedupe_corpus(corpus)
    if key == "cybersecurity":
        corpus = load_cybersecurity_corpus(
            cache_dir=cache_dir or "data/cyber_kb",
            include=tuple(include) if include else ("attack", "cwe",
                                                    "nist", "owasp"),
            max_corpus_size=max_corpus_size,
            random_seed=random_seed,
        )
        return _dedupe_corpus(corpus)
    raise ValueError(
        f"Unknown corpus source {source!r}. Use 'medqa' or 'cybersecurity'."
    )


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
            # Strip leading answer labels like "Ans. a.", "Ans: b", "Answer: C."
            exp = re.sub(r"(?i)^(ans(wer)?\.?\s*[:\-]?\s*[a-d]\.?\s*)+", "", exp).strip()
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
    """Split text into sentence-aware overlapping chunks."""
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    sentences = _split_sentences(clean)
    if not sentences:
        sentences = [clean]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > chunk_size:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

            # Rebuild overlap context from tail sentences.
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                next_len = overlap_len + len(part) + (1 if overlap_parts else 0)
                if next_len > overlap:
                    break
                overlap_parts.append(part)
                overlap_len = next_len
            current = list(reversed(overlap_parts))
            current_len = len(" ".join(current)) if current else 0

        if current:
            current_len += 1 + len(sentence)
        else:
            current_len = len(sentence)
        current.append(sentence)

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _dedupe_corpus(corpus: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in corpus:
        text = row.get("text", "").strip()
        if not text:
            continue
        key = text[:500].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_index(
    persist_dir: str = "data/chroma_db",
    collection_name: str = "medqa_textbooks",
    embedding_model: str = "nomic-embed-text",
    embedding_backend: str = "ollama",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    cache_dir: str | None = None,
    max_corpus_size: int | None = 25_000,
    corpus_source: str = "medqa",
    corpus_include: tuple[str, ...] | None = None,
) -> chromadb.Collection:
    """Build a ChromaDB index from the configured knowledge corpus.

    Embeds chunks using the configured backend (``ollama`` by default; ``hf`` or
    ``medcpt`` for a stronger biomedical embedder) and persists to disk.
    ``corpus_source`` selects between the medical (``medqa``) and cybersecurity
    (``cybersecurity``) KBs.
    """
    embedder = get_embedder(model=embedding_model, backend=embedding_backend)
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

    print(f"Loading corpus (source={corpus_source})...")
    corpus = load_corpus_by_source(
        source=corpus_source,
        cache_dir=cache_dir,
        max_corpus_size=max_corpus_size,
        include=corpus_include,
    )
    print(f"Loaded {len(corpus)} documents from corpus.")

    print("Chunking and embedding...")
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_embeddings: list[list[float]] = []
    batch_meta: list[dict[str, str]] = []
    chunk_idx = 0
    batch_size = 100

    for doc in tqdm(corpus, desc="Processing documents"):
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        if not chunks:
            continue
        chunk_embeddings = embedder.embed_documents(chunks)
        for chunk_pos, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            batch_ids.append(f"chunk_{chunk_idx}")
            batch_docs.append(chunk)
            batch_embeddings.append(embedding)
            batch_meta.append(
                {
                    "source_doc_id": str(doc["id"]),
                    "chunk_pos": str(chunk_pos),
                    "corpus_source": corpus_source,
                }
            )
            chunk_idx += 1

            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_meta,
                )
                batch_ids, batch_docs, batch_embeddings, batch_meta = [], [], [], []

    # Flush remaining
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
        )

    print(f"Index built: {chunk_idx} chunks in collection '{collection_name}'")
    return collection
