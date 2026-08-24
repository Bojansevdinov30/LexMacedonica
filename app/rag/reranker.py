"""Small, local cross-encoder used only to reorder retrieved chunks.

The expensive model is loaded lazily and once per Python process.  Retrieval
still works without it: the caller catches loading/inference failures and keeps
the original RRF order.
"""
from __future__ import annotations

import math
import threading
import time
from functools import lru_cache

from app.config import RERANK_MAX_LENGTH, RERANK_MODEL

_INFERENCE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_reranker():
    """Download/load the model on first use, then reuse that exact instance."""
    from sentence_transformers import CrossEncoder
    import torch

    started = time.perf_counter()
    # Ask explicitly for raw logits.  We apply sigmoid ourselves exactly once,
    # avoiding version-dependent CrossEncoder activation defaults.
    model = CrossEncoder(
        RERANK_MODEL,
        max_length=RERANK_MAX_LENGTH,
        activation_fn=torch.nn.Identity(),
    )
    print(f"[reranker] model loaded in {time.perf_counter() - started:.2f}s")
    return model


def _sigmoid(value: float) -> float:
    """Numerically stable conversion from an unrestricted logit to 0..1."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def rerank_chunks(question: str, chunks: list) -> list[tuple]:
    """Return ``[(chunk, relevance_weight), ...]`` ordered best first.

    ``predict`` tokenizes all question/passage pairs and scores them in one
    batch.  The resulting 0..1 values are relevance weights, not probabilities
    that a lawsuit will have a particular outcome.
    """
    if not chunks:
        return []

    pairs = [(question, chunk.text) for chunk in chunks]
    started = time.perf_counter()
    # Serializing inference avoids two simultaneous requests exhausting the limited RAM here.
    with _INFERENCE_LOCK:
        logits = get_reranker().predict(
            pairs, batch_size=len(pairs), show_progress_bar=False)
    ranked = [
        (chunk, _sigmoid(float(logit)))
        for chunk, logit in zip(chunks, logits)
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    print(f"[reranker] scored {len(pairs)} chunks in "
          f"{time.perf_counter() - started:.2f}s")
    return ranked
