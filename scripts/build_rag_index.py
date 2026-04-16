"""Build the RAG vector index from the MedQA textbook corpus."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.config import load_config
from src.rag.index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from textbook corpus")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--persist-dir", default="data/chroma_db", help="ChromaDB persist directory")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Building RAG index...")
    print(f"  Embedding model: {config.rag.embedding_model}")
    print(f"  Chunk size: {config.rag.chunk_size}")
    print(f"  Chunk overlap: {config.rag.chunk_overlap}")
    print(f"  Persist dir: {args.persist_dir}")

    collection = build_index(
        persist_dir=args.persist_dir,
        collection_name=config.rag.collection_name,
        embedding_model=config.rag.embedding_model,
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
    )

    print(f"\nIndex ready: {collection.count()} chunks indexed.")


if __name__ == "__main__":
    main()
