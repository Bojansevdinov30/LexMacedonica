"""Semantic cache: same *meaning* = same answer, zero LLM cost."""
from __future__ import annotations

import json
import uuid

from app.config import SEMANTIC_CACHE_THRESHOLD
from app.vectorstore import collection, sim


def lookup(query_vector: list[float]) -> dict | None:
    col = collection("semantic_cache")
    if col.count() == 0:
        return None
    r = col.query(query_embeddings=[query_vector], n_results=1,
                  include=["documents", "distances"])
    if r["distances"][0] and sim(r["distances"][0][0]) >= SEMANTIC_CACHE_THRESHOLD:
        return json.loads(r["documents"][0][0])
    return None


def store(query_vector: list[float], response: dict) -> None:
    collection("semantic_cache").add(
        ids=[uuid.uuid4().hex],
        embeddings=[query_vector],
        documents=[json.dumps(response, ensure_ascii=False)],
    )
