"""Semantic cache: same *meaning* = same answer, zero LLM cost.

An exact-match cache would miss "ме отпуштија без причина" vs "добив отказ без
образложение". Instead we store the query EMBEDDING with the answer in a
dedicated Chroma collection; a new query whose vector is ≥ threshold similar
reuses the stored answer.
"""
from __future__ import annotations

import json
import uuid
from functools import lru_cache

import chromadb

from app.config import CHROMA_DIR, SEMANTIC_CACHE_THRESHOLD


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        "semantic_cache", metadata={"hnsw:space": "cosine"}
    )


def lookup(query_vector: list[float]) -> dict | None:
    col = _collection()
    if col.count() == 0:
        return None
    res = col.query(query_embeddings=[query_vector], n_results=1,
                    include=["documents", "distances"])
    if not res["ids"][0]:
        return None
    similarity = 1.0 - res["distances"][0][0]
    if similarity >= SEMANTIC_CACHE_THRESHOLD:
        return json.loads(res["documents"][0][0])
    return None


def store(query_vector: list[float], response: dict) -> None:
    _collection().add(
        ids=[str(uuid.uuid4())],
        embeddings=[query_vector],
        documents=[json.dumps(response, ensure_ascii=False)],
    )
