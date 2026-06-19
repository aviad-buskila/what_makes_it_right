"""Build the RAG vector index from the configured knowledge corpus."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.config import load_config
from src.rag.index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from the configured corpus")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="ChromaDB persist directory (defaults to rag.persist_dir in config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    persist_dir = args.persist_dir or config.rag.persist_dir
    include = (
        tuple(config.rag.corpus_include) if config.rag.corpus_include else None
    )

    print("Building RAG index...")
    print(f"  Corpus source:   {config.rag.corpus_source}")
    if include:
        print(f"  Corpus include:  {include}")
    print(f"  Embedding model: {config.rag.embedding_model}")
    print(f"  Embedding backend: {config.rag.embedding_backend}")
    print(f"  Chunk size:      {config.rag.chunk_size}")
    print(f"  Chunk overlap:   {config.rag.chunk_overlap}")
    print(f"  Persist dir:     {persist_dir}")

    collection = build_index(
        persist_dir=persist_dir,
        collection_name=config.rag.collection_name,
        embedding_model=config.rag.embedding_model,
        embedding_backend=config.rag.embedding_backend,
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
        max_corpus_size=config.rag.max_corpus_size,
        corpus_source=config.rag.corpus_source,
        corpus_include=include,
    )

    print(f"\nIndex ready: {collection.count()} chunks indexed.")


if __name__ == "__main__":
    main()
