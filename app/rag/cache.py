"""Semantic cache: same *meaning* = same answer, zero LLM cost.

An exact-match cache would miss "ме отпуштија без причина" vs "добив отказ без
образложение". Instead we store the query EMBEDDING with the answer; a new
query whose vector is ≥ threshold cosine-similar reuses the stored answer.

At cache sizes we'll ever reach (hundreds of questions) a brute-force numpy
dot product is instant — no index needed.
"""
from __future__ import annotations

import pickle
import threading

import numpy as np

from app.config import CACHE_PATH, SEMANTIC_CACHE_THRESHOLD

_lock = threading.Lock()


def _load() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    return {"vectors": np.zeros((0, 1536), dtype="float32"), "responses": []}


def lookup(query_vector: list[float]) -> dict | None:
    with _lock:
        data = _load()
    if len(data["responses"]) == 0:
        return None
    q = np.array(query_vector, dtype="float32")
    q /= np.linalg.norm(q)
    mat = data["vectors"]  # stored normalized
    sims = mat @ q
    best = int(np.argmax(sims))
    if sims[best] >= SEMANTIC_CACHE_THRESHOLD:
        return dict(data["responses"][best])
    return None


def store(query_vector: list[float], response: dict) -> None:
    q = np.array(query_vector, dtype="float32")
    q /= np.linalg.norm(q)
    with _lock:
        data = _load()
        data["vectors"] = np.vstack([data["vectors"], q[None, :]])
        data["responses"].append(response)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("wb") as f:
            pickle.dump(data, f)
