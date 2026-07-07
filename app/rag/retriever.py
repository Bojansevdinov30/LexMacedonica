"""Two-stage retrieval: first WHICH cases fit the situation, then WHERE inside.

Stage 1 — case level: search the per-case SUMMARIES ("what is this case
about", see ingest/summarize.py). This matches situations, not stray words —
a case that merely mentions "куче" once won't surface for a dog question,
because its summary won't be about dogs.

Stage 2 — passage level: hybrid search (BM25 keywords + FAISS vectors, fused
with Reciprocal Rank Fusion) restricted to the cases stage 1 approved, to pick
the exact passages the LLM reads.

Why hybrid in stage 2? Vector search understands paraphrases ("ме отпуштија"
≈ "престанок на работен однос") but is weak on exact terms like "член 101".
BM25 is the opposite. RRF merges both rankings: score = Σ 1/(60 + rank).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_CHUNKS_FOR_LLM, VECTOR_META_PATH,
)
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
class CaseCandidate:
    """A whole case matched at the situation (summary) level."""
    metadata: dict          # case_id, case_number, court, date, outcome, summary
    similarity: float       # cosine similarity of the query to the case summary


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)  # fused, best first
    cases: list[CaseCandidate] = field(default_factory=list)    # stage-1 ranking
    max_similarity: float = 0.0    # confidence signal for the "Не знам" gate

    def top_chunks(self, n: int = TOP_CHUNKS_FOR_LLM) -> list[RetrievedChunk]:
        return self.chunks[:n]

    def top_cases(self, n: int) -> list[CaseCandidate]:
        """Most situation-similar cases — for probability and citations."""
        return self.cases[:n]


@lru_cache(maxsize=1)
def _bm25_index():
    with BM25_PATH.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _vector_index():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with VECTOR_META_PATH.open("rb") as f:
        meta = pickle.load(f)
    return index, meta


@lru_cache(maxsize=1)
def _summary_index():
    from app.ingest.summarize import SUMMARY_INDEX_PATH, SUMMARY_META_PATH
    if not SUMMARY_INDEX_PATH.exists():
        return None, None   # summaries not built yet -> single-stage fallback
    index = faiss.read_index(str(SUMMARY_INDEX_PATH))
    with SUMMARY_META_PATH.open("rb") as f:
        metas = pickle.load(f)
    return index, metas


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def embed_query(question: str) -> list[float]:
    vec = _embedder().embed_query(question)
    # ~4 tokens per 3 words is a rough floor; exact usage isn't exposed here,
    # so estimate from length (embeddings are so cheap the error is pennies/year)
    log_cost(EMBEDDING_MODEL, max(1, len(question) // 2), 0, label="embed_query")
    return vec


CASE_CANDIDATES = 12     # how many cases stage 1 lets through
WIDE_SEARCH = 80         # stage 2 searches wide, then filters to those cases


def retrieve(question: str) -> RetrievalResult:
    qvec = np.array([embed_query(question)], dtype="float32")
    faiss.normalize_L2(qvec)  # normalized => inner product IS cosine similarity

    # ---------- stage 1: which CASES fit the situation ----------
    case_hits: list[CaseCandidate] = []
    allowed_case_ids: set[str] | None = None
    s_index, s_metas = _summary_index()
    if s_index is not None:
        sims, positions = s_index.search(qvec, CASE_CANDIDATES)
        for pos, sim in zip(positions[0], sims[0]):
            if pos >= 0:
                case_hits.append(CaseCandidate(s_metas[pos], float(sim)))
        allowed_case_ids = {c.metadata["case_id"] for c in case_hits}

    def case_ok(metadata: dict) -> bool:
        return allowed_case_ids is None or metadata.get("case_id") in allowed_case_ids

    # ---------- stage 2: best PASSAGES inside those cases ----------
    candidates: dict[str, RetrievedChunk] = {}

    # vector leg (search wide, keep only chunks from approved cases)
    index, meta = _vector_index()
    sims, positions = index.search(qvec, WIDE_SEARCH)
    vec_rank = {}
    rank = 0
    for pos, similarity in zip(positions[0], sims[0]):
        if pos < 0 or not case_ok(meta["metadatas"][pos]):
            continue
        cid = meta["ids"][pos]
        candidates[cid] = RetrievedChunk(
            cid, meta["texts"][pos], meta["metadatas"][pos],
            vector_similarity=float(similarity),
        )
        vec_rank[cid] = rank
        rank += 1
        if rank >= CANDIDATES:
            break

    # keyword leg (same filter)
    idx = _bm25_index()
    scores = idx["bm25"].get_scores(tokenize(question))
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_rank = {}
    rank = 0
    for i in bm25_order[:WIDE_SEARCH]:
        if scores[i] <= 0 or not case_ok(idx["metadatas"][i]):
            continue
        cid = idx["ids"][i]
        if cid not in candidates:
            candidates[cid] = RetrievedChunk(cid, idx["texts"][i], idx["metadatas"][i])
        bm25_rank[cid] = rank
        rank += 1
        if rank >= CANDIDATES:
            break

    # RRF fusion
    for cid, chunk in candidates.items():
        score = 0.0
        if cid in vec_rank:
            score += 1.0 / (RRF_K + vec_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[cid])
        chunk.rrf_score = score

    fused = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)

    # confidence gate signal: how well the best CASE matches the situation
    # (falls back to best chunk similarity when summaries aren't built yet)
    if case_hits:
        max_sim = case_hits[0].similarity
    else:
        max_sim = max((c.vector_similarity for c in fused), default=0.0)
    return RetrievalResult(chunks=fused, cases=case_hits, max_similarity=max_sim)
