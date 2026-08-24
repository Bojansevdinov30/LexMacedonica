from __future__ import annotations
from dataclasses import dataclass, field
from pprint import pprint
from functools import lru_cache

# just in case you need it - python -c "import app.config; from app.vectorstore import _client; _client().delete_collection('semantic_cache')" to clear semantic cache
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from app.config import EMBEDDING_MODEL, TOP_CHUNKS_FOR_LLM
from app.ingest.build_index import tokenize
from app.vectorstore import collection, sim

RRF_K = 60
CANDIDATES = 20
WIDE_SEARCH = 80

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



@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    rrf_score: float= 0.0
    vector_similarity: float=0.0

@dataclass
class CaseCandidate:
    metadata: dict
    similarity: float


@lru_cache(maxsize=1)
def _bm25_index():

    data = collection("chunks").get(include=["documents", "metadatas"])
    return {"bm25": BM25Okapi([tokenize(t) for t in data["documents"]]),
            "texts": data["documents"],
            "metadatas": data["metadatas"],
            "ids": data["ids"]
            }

@lru_cache(maxsize=1)
def _summaries_collection():
    col = collection("summaries")
    return col if col.count() > 0 else None

@lru_cache
def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def embed_query(question: str) -> list[float]:
    return _embedder().embed_query(question)

def retrieve(question: str, CASE_CANDIDATES: int=12, qvec: list[float] | None = None) -> RetrievalResult:
    if qvec is None:
        qvec = embed_query(question)

    # Doesnt print any new information, just 1536 numbers by themself
    # print("#" * 60)
    # print("Vector embedding for the question: ")
    # pprint(qvec)

    # pass through the summaries
    case_hits: list[CaseCandidate] = []
    allowed_case_ids: set[str] | None = None
    s_col = _summaries_collection()
    if s_col is not None:
        r = s_col.query(query_embeddings=[qvec],
                        n_results=min(s_col.count(), CASE_CANDIDATES),
                        include=["documents", "metadatas", "distances"])

        # print("#" * 60)
        # print("Retrieved most similar cases: ")
        # pprint(r["distances"])

        for m, doc, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0]):
            meta = dict(m)
            meta["summary"] = doc
            case_hits.append(CaseCandidate(meta, sim(dist)))
        allowed_case_ids = {c.metadata["case_id"] for c in case_hits}

    def case_ok(metadata: dict) -> bool:
        return allowed_case_ids is None or metadata.get("case_id") in allowed_case_ids

    # find best matches inside the allowed cases
    candidates: dict[str, RetrievedChunk] = {}
    where = ({"case_id": {"$in": sorted(allowed_case_ids)}} if allowed_case_ids else None)
    r = collection("chunks").query(query_embeddings=[qvec],
                                   n_results=CANDIDATES,
                                   where=where,
                                   include=["documents", "metadatas", "distances"])

    print("#" * 60)
    print("Retrieved most similar chunks from those cases: ")
    pprint(r["distances"])

    vec_rank: dict[str, int] = {}
    for rank, (cid, text, m, dist) in enumerate(zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])):
        candidates[cid] = RetrievedChunk(cid, text, dict(m), vector_similarity=sim(dist))
        vec_rank[cid] = rank

    # print("#" * 60)
    # print("Vector ranking: ")
    # pprint(vec_rank)

    idx = _bm25_index()

    # This would print the entire index, massive
    # print("#" * 60)
    # print("bm25 index: ", idx)

    scores = idx["bm25"].get_scores(tokenize(question))
    bm25_order = sorted(range(len(scores)), key= lambda i: scores[i], reverse=True)

    #No meaningful information
    # print("#" * 60)
    # print("bm25 order: ")
    # pprint(bm25_order)

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

    # print("#" * 60)
    # print("bm25 rank: ")
    # pprint(bm25_rank)

    # RRF fusion
    for cid, chunk in candidates.items():
        score = 0.0
        if cid in vec_rank:
            score += 1.0 / (RRF_K + vec_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[cid])
        chunk.rrf_score = score

    fused = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)

    # Already printed with the result
    # print("#" * 60)
    # print("Fused: ")
    # pprint(fused)
    # confidence gate signal: how well the best CASE matches the situation
    # (falls back to best chunk similarity when summaries aren't built yet)
    if case_hits:
        max_sim = case_hits[0].similarity
    else:
        max_sim = max((c.vector_similarity for c in fused), default=0.0)


    # print("#" * 60)
    # print("Retrieval result: ")
    # print("Chunks: ", fused)
    # print("Cases: ", case_hits)
    # print("Max similarity: ", max_sim)
    return RetrievalResult(chunks=fused, cases=case_hits, max_similarity=max_sim)


if __name__ == "__main__":
    QUESTIONS = [
        # typical labor-dispute situations (should answer with cases + %):
        "Работодавачот не ми ги исплати последните три плати. Што можам да направам и какви се шансите да ги добијам преку суд?",
        "Кој е најдобриот фудбалски клуб на светот?"
    ]
    for idx, question in enumerate(QUESTIONS, 1):
        qvec = embed_query(question)
        print("=" * 60)
        print("Question #", idx)
        r1 = retrieve(question, 12, qvec)
        pprint(r1.max_similarity)
        print("+" * 60)
        r2 = retrieve(question, 581, qvec)
        pprint(r2.max_similarity)
        print("=" * 60)
