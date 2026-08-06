"""Two-stage retrieval over ChromaDB: first WHICH cases fit, then WHERE inside.

Stage 1 — case level: search the per-case SUMMARIES. This matches situations, not stray words —
a case that merely mentions "куче" once won't surface for a dog question,
because its summary won't be about dogs.

Stage 2 — passage level: hybrid search (BM25 keywords + Chroma vectors, fused
with Reciprocal Rank Fusion) restricted to the cases stage 1 approved, to pick
the exact passages the LLM reads.

Why hybrid in stage 2? Vector search understands paraphrases ("ме отпуштија"
≈ "престанок на работен однос") but is weak on exact terms like "член 101".
BM25 is the opposite. RRF merges both rankings: score = Σ 1/(60 + rank)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, TOP_CHUNKS_FOR_LLM
from app.ingest.build_index import tokenize
from app.vectorstore import collection, sim

RRF_K = 60           # standard constant; damps the impact of exact rank positions
CANDIDATES = 20      # how many candidates each retriever contributes
CASE_CANDIDATES = 12 # how many cases stage 1 lets through
WIDE_SEARCH = 80     # BM25 looks at this many top keyword hits before filtering


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
    """BM25 rebuilt from the chunks stored in Chroma, once per process (~2-5 s).
    """
    from rank_bm25 import BM25Okapi

    data = collection("chunks").get(include=["documents", "metadatas"])
    return {"bm25": BM25Okapi([tokenize(t) for t in data["documents"]]),
            "texts": data["documents"],
            "metadatas": data["metadatas"],
            "ids": data["ids"]}


@lru_cache(maxsize=1)
def _summaries_collection():
    col = collection("summaries")
    return col if col.count() > 0 else None  # not built yet -> single-stage fallback


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def embed_query(question: str) -> list[float]:
    return _embedder().embed_query(question)


def retrieve(question: str, query_vector: list[float] | None = None) -> RetrievalResult:
    # reuse the caller's embedding when it has one
    if query_vector is None:
        query_vector = embed_query(question)

    # ---------- stage 1: which CASES fit the situation ----------
    case_hits: list[CaseCandidate] = []
    allowed_case_ids: set[str] | None = None
    s_col = _summaries_collection()
    if s_col is not None:
        r = s_col.query(query_embeddings=[query_vector],
                        n_results=min(CASE_CANDIDATES, s_col.count()),
                        include=["documents", "metadatas", "distances"])
        for m, doc, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0]):
            meta = dict(m)
            meta["summary"] = doc   # the summary is the stored document text
            case_hits.append(CaseCandidate(meta, sim(dist)))
        allowed_case_ids = {c.metadata["case_id"] for c in case_hits}

    def case_ok(metadata: dict) -> bool:
        return allowed_case_ids is None or metadata.get("case_id") in allowed_case_ids

    # ---------- stage 2: best PASSAGES inside those cases ----------
    candidates: dict[str, RetrievedChunk] = {}

    # vector leg — Chroma filters by case_id AT the search (no wide-then-filter)
    where = ({"case_id": {"$in": sorted(allowed_case_ids)}}
             if allowed_case_ids else None)
    r = collection("chunks").query(
        query_embeddings=[query_vector], n_results=CANDIDATES, where=where,
        include=["documents", "metadatas", "distances"])
    vec_rank: dict[str, int] = {}
    for rank, (cid, text, m, dist) in enumerate(zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])):
        candidates[cid] = RetrievedChunk(cid, text, dict(m), vector_similarity=sim(dist))
        vec_rank[cid] = rank

    # keyword leg (BM25 can't filter while scoring, so: score all, then filter)
    idx = _bm25_index()
    scores = idx["bm25"].get_scores(tokenize(question))
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_rank: dict[str, int] = {}
    rank = 0
    for i in bm25_order[:WIDE_SEARCH]:
        if scores[i] <= 0 or not case_ok(idx["metadatas"][i]):
            continue
        cid = idx["ids"][i]
        if cid not in candidates:
            candidates[cid] = RetrievedChunk(cid, idx["texts"][i], dict(idx["metadatas"][i]))
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
