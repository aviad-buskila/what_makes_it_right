from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import ollama

T = TypeVar("T")


def _should_retry(exc: Exception) -> bool:
    """Retry transient failures, not permanent request errors."""
    if isinstance(exc, ollama.ResponseError):
        # Missing model / bad request are not transient.
        return exc.status_code >= 500
    return True


def _with_retry_backoff(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_backoff_seconds: float = 1.0,
) -> T:
    """Run operation with exponential backoff for transient errors."""
    delay = initial_backoff_seconds
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts or not _should_retry(exc):
                break
            print(
                f"{operation_name} failed on attempt {attempt}/{max_attempts}: {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2

    assert last_exc is not None
    raise last_exc


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
    timeout: int = 60,
    seed: int | None = None,
) -> tuple[str, float]:
    """Generate a response from an Ollama model.

    Returns (response_text, latency_seconds).
    ``seed`` is forwarded to Ollama so that repeated calls with different seeds
    produce independent samples rather than hitting the KV cache.
    """
    start = time.perf_counter()
    client = ollama.Client(timeout=timeout)
    options: dict = {"temperature": temperature, "num_ctx": 4096}
    if seed is not None:
        options["seed"] = seed
    response = _with_retry_backoff(
        lambda: client.generate(
            model=model_id,
            prompt=prompt,
            options=options,
        ),
        operation_name=f"generate({model_id})",
    )
    latency = time.perf_counter() - start
    return response["response"], latency


def generate_embedding(
    text: str,
    model: str = "nomic-embed-text",
    timeout: int = 60,
) -> list[float]:
    """Generate an embedding vector for text using Ollama."""
    client = ollama.Client(timeout=timeout)
    response = _with_retry_backoff(
        lambda: client.embed(model=model, input=text),
        operation_name=f"embed({model})",
    )
    return response["embeddings"][0]
