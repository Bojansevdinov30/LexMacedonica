"""Corpus-wide hybrid retrieval followed by a small local reranking pass.

Dense search understands paraphrases; BM25 finds exact legal terms. Reciprocal
Rank Fusion (RRF) creates a cheap candidate list over the whole corpus. Only
the first few candidates go through the slower cross-encoder.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from app.config import (
    EMBEDDING_MODEL,
    MIN_SIMILARITY_FOR_ANSWER,
    RERANKER_ENABLED,
    RERANK_POOL,
    RETRIEVED_CHUNKS,
    TOP_CHUNKS_FOR_LLM,
)
from app.ingest.build_index import tokenize
from app.rag.reranker import rerank_chunks
from app.vectorstore import collection, sim

RRF_K = 60
# Each leg contributes enough candidates for a final fused pool of 30.
CANDIDATES = RETRIEVED_CHUNKS
WIDE_SEARCH = 100


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    rrf_score: float = 0.0
    vector_similarity: float = 0.0
    reranker_score: float = 0.0


@dataclass
class CaseCandidate:
    metadata: dict
    # Normally the best sigmoid reranker score for this case. On the documented
    # emergency fallback path it is a normalized RRF weight instead.
    similarity: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    cases: list[CaseCandidate] = field(default_factory=list)
    # This deliberately remains DENSE chunk similarity. chains.py uses it for
    # the inexpensive gate before allowing an answer.
    max_similarity: float = 0.0

    def top_chunks(self, n: int = TOP_CHUNKS_FOR_LLM) -> list[RetrievedChunk]:
        return self.chunks[:n]

    def top_cases(self, n: int) -> list[CaseCandidate]:
        return self.cases[:n]


@lru_cache(maxsize=1)
def _bm25_index():
    """Build BM25 once per process from the same chunks stored in Chroma."""
    data = collection("chunks").get(include=["documents", "metadatas"])
    return {
        "bm25": BM25Okapi([tokenize(text) for text in data["documents"]]),
        "texts": data["documents"],
        "metadatas": data["metadatas"],
        "ids": data["ids"],
    }


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def embed_query(question: str) -> list[float]:
    return _embedder().embed_query(question)


def _case_candidates(ranked: list[tuple[RetrievedChunk, float]]) -> list[CaseCandidate]:
    """Deduplicate only the reranked head, keeping each case's best chunk."""
    best: dict[str, tuple[dict, float]] = {}
    for chunk, score in ranked:
        case_id = chunk.metadata.get("case_id")
        if case_id and (case_id not in best or score > best[case_id][1]):
            best[case_id] = (dict(chunk.metadata), score)
    if not best:
        return []

    # Summary text is fetched by ID only for citation cards. It is not searched
    # and does not restrict which chunks may participate in retrieval.
    summaries: dict[str, str] = {}
    try:
        got = collection("summaries").get(ids=list(best), include=["documents"])
        summaries = dict(zip(got["ids"], got["documents"]))
    except Exception as exc:
        print(f"[retrieval] summaries unavailable for cards: {type(exc).__name__}")

    cases = []
    for case_id, (metadata, score) in best.items():
        metadata["summary"] = summaries.get(case_id, "")
        cases.append(CaseCandidate(metadata=metadata, similarity=score))
    cases.sort(key=lambda case: case.similarity, reverse=True)
    return cases


def _fallback_rank(pool: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
    """Keep the app useful if the optional local model cannot run.

    RRF is not on the reranker's scale, so it is normalized only within this
    small fallback pool. This preserves the existing weighted probability
    behavior without pretending that the two score types are comparable.
    """
    maximum = max((chunk.rrf_score for chunk in pool), default=0.0)
    return [(chunk, chunk.rrf_score / maximum if maximum else 0.001)
            for chunk in pool]


def retrieve(question: str, query_vector: list[float] | None = None) -> RetrievalResult:
    started = time.perf_counter()
    if query_vector is None:
        query_vector = embed_query(question)

    candidates: dict[str, RetrievedChunk] = {}

    dense_started = time.perf_counter()
    dense = collection("chunks").query(
        query_embeddings=[query_vector],
        n_results=CANDIDATES,
        include=["documents", "metadatas", "distances"],
    )
    vector_ranks: dict[str, int] = {}
    for rank, (chunk_id, text, metadata, distance) in enumerate(zip(
            dense["ids"][0], dense["documents"][0], dense["metadatas"][0],
            dense["distances"][0])):
        candidates[chunk_id] = RetrievedChunk(
            chunk_id, text, dict(metadata), vector_similarity=sim(distance))
        vector_ranks[chunk_id] = rank
    dense_time = time.perf_counter() - dense_started

    bm25_started = time.perf_counter()
    index = _bm25_index()
    scores = index["bm25"].get_scores(tokenize(question))
    order = sorted(range(len(scores)), key=lambda position: scores[position], reverse=True)
    bm25_ranks: dict[str, int] = {}
    rank = 0
    for position in order[:WIDE_SEARCH]:
        if scores[position] <= 0:
            continue
        chunk_id = index["ids"][position]
        if chunk_id not in candidates:
            candidates[chunk_id] = RetrievedChunk(
                chunk_id, index["texts"][position], dict(index["metadatas"][position]))
        bm25_ranks[chunk_id] = rank
        rank += 1
        if rank >= CANDIDATES:
            break
    bm25_time = time.perf_counter() - bm25_started

    fusion_started = time.perf_counter()
    for chunk_id, chunk in candidates.items():
        chunk.rrf_score = (
            (1.0 / (RRF_K + vector_ranks[chunk_id]) if chunk_id in vector_ranks else 0.0)
            + (1.0 / (RRF_K + bm25_ranks[chunk_id]) if chunk_id in bm25_ranks else 0.0)
        )
    fused = sorted(candidates.values(), key=lambda chunk: chunk.rrf_score, reverse=True)
    fused = fused[:RETRIEVED_CHUNKS]
    fusion_time = time.perf_counter() - fusion_started

    max_similarity = max(
        (chunk.vector_similarity for chunk in candidates.values()), default=0.0)
    print(f"[retrieval] dense={dense_time:.3f}s bm25={bm25_time:.3f}s "
          f"fusion={fusion_time:.3f}s max_dense={max_similarity:.3f}")

    # Reject a weak query before loading or running the expensive reranker.
    if max_similarity < MIN_SIMILARITY_FOR_ANSWER or not fused:
        print(f"[retrieval] stopped before reranking; total={time.perf_counter() - started:.3f}s")
        return RetrievalResult(chunks=fused, max_similarity=max_similarity)

    pool = fused[:RERANK_POOL]
    tail = fused[RERANK_POOL:]
    if RERANKER_ENABLED:
        try:
            ranked = rerank_chunks(question, pool)
        except Exception as exc:
            print(f"[reranker] {type(exc).__name__}: {exc}; using RRF fallback")
            ranked = _fallback_rank(pool)
    else:
        print("[reranker] disabled; using RRF fallback")
        ranked = _fallback_rank(pool)

    for chunk, score in ranked:
        chunk.reranker_score = score
    final_chunks = [chunk for chunk, _ in ranked] + tail
    cases = _case_candidates(ranked)

    print(f"[retrieval] candidates={len(fused)} reranked={len(ranked)} "
          f"unique_cases={len(cases)} total={time.perf_counter() - started:.3f}s")
    return RetrievalResult(
        chunks=final_chunks,
        cases=cases,
        max_similarity=max_similarity,
    )
