from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pprint import pprint
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from app.config import EMBEDDING_MODEL, TOP_CHUNKS_FOR_LLM, TOP_CASES_FOR_PROBABILITY
from app.ingest.build_index import tokenize
from app.vectorstore import collection, sim

RRF_K = 60  # standard constant; damps the impact of exact rank positions
CANDIDATES = 30  # how many candidates each retriever contributes
CASE_CANDIDATES = 581  # how many cases stage 1 lets through
WIDE_SEARCH = 80  # BM25 looks at this many top keyword hits before filtering


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
    metadata: dict  # case_id, case_number, court, date, outcome, summary
    similarity: float  # cosine similarity of the query to the case summary


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)  # fused, best first
    cases: list[CaseCandidate] = field(default_factory=list)  # stage-1 ranking
    max_similarity: float = 0.0  # confidence signal for the "Не знам" gate

    def top_chunks(self, n: int = TOP_CHUNKS_FOR_LLM) -> list[RetrievedChunk]:
        return self.chunks[:n]

    def top_cases(self, n: int = TOP_CASES_FOR_PROBABILITY) -> list[CaseCandidate]:
        return self.cases[:n]


@lru_cache(maxsize=1)
def _bm25_index():
    """BM25 rebuilt from the chunks stored in Chroma, once per process (~2-5 s)."""

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


def aggregate_cases(chunks: list[RetrievedChunk], n: int = TOP_CASES_FOR_PROBABILITY) -> list[CaseCandidate]:
    case_scores: dict[str, float] = {}
    case_metadata: dict[str, dict] = {}

    for chunk in chunks:
        case_id = chunk.metadata["case_id"]

        case_scores[case_id] = (
                case_scores.get(case_id, 0.0)
                + chunk.rrf_score
        )

        if case_id not in case_metadata:
            case_metadata[case_id] = {
                "case_id": case_id,
                "case_number": chunk.metadata.get("case_number"),
                "court": chunk.metadata.get("court"),
                "date": chunk.metadata.get("date"),
                "outcome": chunk.metadata.get("outcome"),
            }

    ranked_case_ids = sorted(case_scores, key=case_scores.get, reverse=True)[:n]
    # Get summaries only for these top cases
    summary_col = collection("summaries")

    summary_result = summary_col.get(
        ids=ranked_case_ids,
        include=["documents"],
    )

    summaries = dict(zip(
        summary_result["ids"],
        summary_result["documents"],
    ))

    return [
        CaseCandidate(
            metadata={
                **case_metadata[case_id],
                "summary": summaries.get(case_id, ""),
            },
            similarity=case_scores[case_id],
        )
        for case_id in ranked_case_ids
    ]


def retrieve(question: str, query_vector: list[float] | None = None) -> RetrievalResult:
    # reuse the caller's embedding when it has one
    if query_vector is None:
        query_vector = embed_query(question)

    # ---------- stage 1: which CASES fit the situation ----------
    # case_hits: list[CaseCandidate] = []
    # allowed_case_ids: set[str] | None = None
    # s_col = _summaries_collection()
    # if s_col is not None:
    #     r = s_col.query(query_embeddings=[query_vector],
    #                     n_results=min(CASE_CANDIDATES, s_col.count()),
    #                     include=["documents", "metadatas", "distances"])
    #     for m, doc, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0]):
    #         meta = dict(m)
    #         meta["summary"] = doc  # the summary is the stored document text
    #         case_hits.append(CaseCandidate(meta, sim(dist)))
    #     allowed_case_ids = {c.metadata["case_id"] for c in case_hits}
    #
    # def case_ok(metadata: dict) -> bool:
    #     return allowed_case_ids is None or metadata.get("case_id") in allowed_case_ids

    # ---------- stage 2: best PASSAGES inside those cases ----------
    candidates: dict[str, RetrievedChunk] = {}

    # vector leg — Chroma filters by case_id AT the search (no wide-then-filter)
    # where = ({"case_id": {"$in": sorted(allowed_case_ids)}}
    #          if allowed_case_ids else None)
    r = collection("chunks").query(
        query_embeddings=[query_vector], n_results=CANDIDATES,
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
        if scores[i] <= 0:
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

    max_sim = max((c.vector_similarity for c in fused), default=0.0)

    # ---------- cases from fused chunks ----------
    cases = aggregate_cases(fused)

    return RetrievalResult(chunks=fused, cases=cases, max_similarity=max_sim)

NON_MERITS = {"НЕПОЗНАТО", "ЗАПРЕНА ПОСТАПКА", "СПОГОДБА", "ПОТВРДЕНО", "УКИНАТО"}

READABLE = {
    "УСВОЕНО": "тужбеното барање е усвоено (тужителот добил)",
    "ДЕЛУМНО УСВОЕНО": "тужбеното барање е делумно усвоено",
    "ОДБИЕНО": "тужбеното барање е одбиено (тужениот добил)",
    "ОТФРЛЕНО": "тужбата е отфрлена од процедурални причини",
    "СПОГОДБА": "странките склучиле спогодба/порамнување",
    "ЗАПРЕНА ПОСТАПКА": "постапката е запрена (нпр. повлечена тужба)",
    "ПОТВРДЕНО": "првостепената одлука е потврдена",
    "УКИНАТО": "првостепената одлука е укината",
}


@dataclass
class ProbabilityEstimate:
    outcome: str  # dominant outcome label
    outcome_readable: str
    percent: int  # weighted share of that outcome, 0-100
    counts: dict[str, int]  # raw outcome counts among the sample
    sample_size: int


def estimate(result: RetrievalResult) -> ProbabilityEstimate | None:
    cases = result.top_cases(TOP_CASES_FOR_PROBABILITY)
    pprint(cases)
    weighted: dict[str, float] = {}
    counts: dict[str, int] = {}

    for case in cases:
        outcome = case.metadata.get("outcome", "НЕПОЗНАТО")
        if outcome in NON_MERITS:
            continue
        # similarity of the case SUMMARY to the user's situation — cases that
        # match the situation better influence the estimate more
        weight = max(case.similarity, 0.001)
        weighted[outcome] = weighted.get(outcome, 0.0) + weight
        counts[outcome] = counts.get(outcome, 0) + 1

    if not weighted or sum(counts.values()) < 3:
        return None  # too few comparable cases for an honest percentage

    total = sum(weighted.values())
    dominant = max(weighted, key=weighted.get)
    percent = round(100 * weighted[dominant] / total)

    return ProbabilityEstimate(
        outcome=dominant,
        outcome_readable=READABLE.get(dominant, dominant),
        percent=percent,
        counts=counts,
        sample_size=sum(counts.values()),
    )

if __name__ == "__main__":
    QUESTIONS = [
        # typical labor-dispute situations (should answer with cases + %):
        # "Работодавачот не ми ги исплати последните три плати. Што можам да направам и какви се шансите да ги добијам преку суд?",
        # "Кој е најдобриот фудбалски клуб на светот?"
        "Мојот работодавач постојано врши мобинг врз вработените, вклучувајќи ме и мене. Што можам да направам и какви се шансите да ги добијам преку суд?"
    ]
    for idx, question in enumerate(QUESTIONS):
        qvec = embed_query(question)
        print("=" * 60)
        print("Question #", idx+1, ": ", QUESTIONS[idx])
        r1 = retrieve(question, qvec)
        pprint(r1.top_chunks(3))
        print("+" * 60)
        pprint(estimate(r1))