from __future__ import annotations

import time

import ollama


def check_health() -> bool:
    """Check if Ollama is running."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ensure_model(model_id: str) -> None:
    """Pull a model if not already available locally."""
    try:
        ollama.show(model_id)
    except ollama.ResponseError:
        print(f"Pulling model {model_id}...")
        ollama.pull(model_id)
        print(f"Model {model_id} ready.")


def generate(
    model_id: str,
    prompt: str,
    temperature: float = 0.7,
    timeout: int = 120,
) -> tuple[str, float]:
    """Generate a response from an Ollama model.

    Returns (response_text, latency_seconds).
    """
    start = time.perf_counter()
    response = ollama.generate(
        model=model_id,
        prompt=prompt,
        options={"temperature": temperature, "num_ctx": 4096},
    )
    latency = time.perf_counter() - start
    return response["response"], latency


def generate_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    """Generate an embedding vector for text using Ollama."""
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]
