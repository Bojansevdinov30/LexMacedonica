"""ВАРИЈАНТА А — drop-in замена за app/rag/retriever.py, но БЕЗ фаза 1.

    hybrid BM25+вектори по СИТЕ ~9000 chunks + RRF
    → cross-encoder ги прескорува топ HYBRID_CANDIDATES → chains.py чита топ 3

Summary-индексот повеќе НЕ филтрира кои случаи смеат да учествуваат —
се потпираме на cross-encoder-от да го најде вистинското и во поголем куп.
Поедноставна архитектура (една фаза помалку), но со реални последици —
види ги двете предупредувања подолу.

⚠️ 1. ПРАГОТ „НЕ ЗНАМ" МОРА ДА СЕ ПРЕМЕРИ
   max_similarity сега е rerank-скор (друга скала!), а не summary-сличност.
   Живиот праг 0.43 е измерен на summary-скала и ТУКА НЕ ВАЖИ.
   Измери го:      python -m ideas.reranker_benchmark --gate
   Постави го без менување код (config.py е env-overridable):
                   set LEX_MIN_SIMILARITY_FOR_ANSWER=0.xx

⚠️ 2. ВЕРОЈАТНОСТА СЕ ПРЕСМЕТУВА ОД ПОЛОШ ПРИМЕРОК
   `cases` веќе не е ранг-листа на ситуациски слични случаи, туку се изведува
   од прескорираните ПАСУСИ: по случај се зема најдобриот rerank-скор како
   тежина. Тоа значи:
     • примерокот е помал и пристрасен — само случаи чиј пасус преживеал
       hybrid+rerank (типично 4-6 наместо 10 случаи)
     • тежината мери „колку пасусот одговара на прашањето", а не „колку
       ситуацијата личи на твојата" — а веројатноста е тврдење за ситуации
   Затоа варијанта Б (retriever_reranked.py) е попаметниот прв чекор:
   таму фаза 1 останува и веројатноста е недопрена.

Резимеата за картичките сепак се дочитуваат од Chroma колекцијата
"summaries" (обичен get по id — без embedding, бесплатно), за да не останат
празни картички во интерфејсот.

КАКО СЕ ВКЛУЧУВА — една линија во app/rag/chains.py:
    # from app.rag.retriever import embed_query, retrieve
    from ideas.retriever_rerank_only import embed_query, retrieve
...плус да го поставиш новиот праг (предупредување 1). Враќање назад: врати
ја старата линија, врати го прагот и испразни го семантичкиот кеш.

Предуслов: pip install sentence-transformers
"""
from __future__ import annotations

from app.ingest.build_index import tokenize
from app.rag.retriever import (CANDIDATES, RRF_K, WIDE_SEARCH, CaseCandidate,
                               RetrievalResult, RetrievedChunk, _bm25_index)
# re-export, за drop-in паритет (chains.py/lawyer увезуваат и embed_query)
from app.rag.retriever import embed_query  # noqa: F401
from app.vectorstore import collection, sim
from ideas.reranker_model import rerank_chunks

HYBRID_CANDIDATES = 10   # колку fused пасуси одат на cross-encoder


def _cases_from_chunks(ranked: list[tuple]) -> list[CaseCandidate]:
    """Изведи ранг-листа на СЛУЧАИ од прескорираните пасуси.

    По случај се задржува најдобриот rerank-скор на негов пасус. Резимето се
    дочитува од колекцијата "summaries" за да работат картичките со цитати
    (chunk-метаподатоците немаат summary поле).
    """
    best: dict[str, tuple[dict, float]] = {}
    for chunk, score in ranked:
        cid = chunk.metadata.get("case_id")
        if cid and (cid not in best or score > best[cid][1]):
            best[cid] = (dict(chunk.metadata), score)
    if not best:
        return []

    got = collection("summaries").get(ids=list(best), include=["documents"])
    summaries = dict(zip(got["ids"], got["documents"]))

    cases = []
    for cid, (meta, score) in best.items():
        meta["summary"] = summaries.get(cid, "")
        cases.append(CaseCandidate(meta, score))
    return sorted(cases, key=lambda c: c.similarity, reverse=True)


def retrieve(question: str,
             query_vector: list[float] | None = None) -> RetrievalResult:
    """Ист потпис и ист повратен тип како app.rag.retriever.retrieve."""
    if query_vector is None:
        query_vector = embed_query(question)

    candidates: dict[str, RetrievedChunk] = {}

    # векторски дел — по СИТЕ chunks, без where-филтер (нема фаза 1)
    r = collection("chunks").query(
        query_embeddings=[query_vector], n_results=CANDIDATES,
        include=["documents", "metadatas", "distances"])
    vec_rank: dict[str, int] = {}
    for rank, (cid, text, m, dist) in enumerate(zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])):
        candidates[cid] = RetrievedChunk(cid, text, dict(m), vector_similarity=sim(dist))
        vec_rank[cid] = rank

    # клучен збор дел — истиот BM25 индекс како живиот retriever
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

    # RRF фузија — идентична на живата
    for cid, chunk in candidates.items():
        chunk.rrf_score = ((1.0 / (RRF_K + vec_rank[cid]) if cid in vec_rank else 0.0)
                           + (1.0 / (RRF_K + bm25_rank[cid]) if cid in bm25_rank else 0.0))
    fused = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)

    # cross-encoder-от го одредува финалниот ред
    ranked = rerank_chunks(question, fused[:HYBRID_CANDIDATES])
    chunks = [chunk for chunk, _ in ranked] + fused[HYBRID_CANDIDATES:]

    # max_similarity е сега rerank-скор -> в. предупредување 1 во докстрингот
    return RetrievalResult(chunks=chunks,
                           cases=_cases_from_chunks(ranked),
                           max_similarity=ranked[0][1] if ranked else 0.0)
