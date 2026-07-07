"""Hybrid retrieval: BM25 (keywords) + Chroma vectors (meaning), fused with RRF.

Why hybrid? Vector search understands paraphrases ("ме отпуштија" ≈ "престанок
на работен однос") but is weak on exact terms like "член 101" or "РО-343/22".
BM25 is the opposite. Reciprocal Rank Fusion (RRF) merges both rankings without
needing their scores to be comparable: score = Σ 1/(60 + rank).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, TOP_CHUNKS_FOR_LLM
from app.costs import log_cost
from app.ingest.build_index import BM25_PATH, tokenize

RRF_K = 60           # standard constant; damps the impact of exact rank positions
CANDIDATES = 20      # how many candidates each retriever contributes


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    rrf_score: float = 0.0
    vector_similarity: float = 0.0  # cosine similarity (0..1), 0 if BM25-only


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)  # fused, best first
    max_similarity: float = 0.0    # confidence signal for the "Не знам" gate

    def top_chunks(self, n: int = TOP_CHUNKS_FOR_LLM) -> list[RetrievedChunk]:
        return self.chunks[:n]

    def top_cases(self, n: int) -> list[RetrievedChunk]:
        """Best chunk per distinct case — for the probability statistics."""
        seen, out = set(), []
        for c in self.chunks:
            cid = c.metadata.get("case_id")
            if cid not in seen:
                seen.add(cid)
                out.append(c)
            if len(out) >= n:
                break
        return out


@lru_cache(maxsize=1)
def _bm25_index():
    with BM25_PATH.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("cases")


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def embed_query(question: str) -> list[float]:
    vec = _embedder().embed_query(question)
    # ~4 tokens per 3 words is a rough floor; exact usage isn't exposed here,
    # so estimate from length (embeddings are so cheap the error is pennies/year)
    log_cost(EMBEDDING_MODEL, max(1, len(question) // 2), 0, label="embed_query")
    return vec


def retrieve(question: str) -> RetrievalResult:
    candidates: dict[str, RetrievedChunk] = {}

    # --- vector leg ---
    qvec = embed_query(question)
    res = _chroma_collection().query(
        query_embeddings=[qvec], n_results=CANDIDATES,
        include=["documents", "metadatas", "distances"],
    )
    vec_rank = {}
    for rank, (cid, doc, meta, dist) in enumerate(zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])):
        similarity = 1.0 - dist  # cosine distance -> cosine similarity
        candidates[cid] = RetrievedChunk(cid, doc, meta, vector_similarity=similarity)
        vec_rank[cid] = rank

    # --- keyword leg ---
    idx = _bm25_index()
    scores = idx["bm25"].get_scores(tokenize(question))
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    bm25_rank = {}
    for rank, i in enumerate(bm25_order):
        if scores[i] <= 0:
            continue
        cid = idx["ids"][i]
        if cid not in candidates:
            candidates[cid] = RetrievedChunk(cid, idx["texts"][i], idx["metadatas"][i])
        bm25_rank[cid] = rank

    # --- RRF fusion ---
    for cid, chunk in candidates.items():
        score = 0.0
        if cid in vec_rank:
            score += 1.0 / (RRF_K + vec_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[cid])
        chunk.rrf_score = score

    fused = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
    max_sim = max((c.vector_similarity for c in fused), default=0.0)
    return RetrievalResult(chunks=fused, max_similarity=max_sim)
