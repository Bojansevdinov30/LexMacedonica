"""ВАРИЈАНТА Б (ПРЕПОРАЧАНА) — drop-in замена за app/rag/retriever.py.

    summaries (фаза 1) → hybrid BM25+вектори + RRF (фаза 2)
    → cross-encoder ги прескорува fused пасусите → chains.py чита топ 3

Реранкерот НЕ ја заменува архитектурата — доаѓа НАКРАЈ како слој за
дотерување. RRF паѓа од „финален судија" на „генератор на кандидати":
неговата задача станува само да ги донесе вистинските пасуси во базенот,
а точниот редослед го одредува cross-encoder-от.

ЗОШТО Е ВИСТИНСКИ DROP-IN
Враќа ист RetrievalResult (chunks / cases / max_similarity) и менува САМО
го редоследот на chunks, па сè што чита од резултатот работи непроменето:
  • probability.estimate()  — чита result.cases (недопрено)
  • картичките со цитати    — result.top_cases(3) (недопрено)
  • прагот „Не знам" (0.43) — max_similarity и понатаму е summary-сличност,
                              па калибрацијата останува валидна
  • семантичкиот кеш, chains.py, frontend-от — ништо не знаат за промената

КАКО СЕ ВКЛУЧУВА — една линија во app/rag/chains.py:
    # from app.rag.retriever import embed_query, retrieve
    from ideas.retriever_reranked import embed_query, retrieve
(истото и во app/lawyer/rag.py ако сакаш и адвокатскиот таб да го користи)

Враќање назад: врати ја старата линија. По секое менување испразни го кешот,
инаку старите одговори го маскираат ефектот:
    python -c "import app.config; from app.vectorstore import _client; _client().delete_collection('semantic_cache')"

Предуслов: pip install sentence-transformers   (~2.5 GB torch + 2.3 GB модел)
Латенција: cross-encoder-от прави по едно поминување за секој кандидат, па
времето расте линеарно со RERANK_POOL. Измери со reranker_benchmark.py.
"""
from __future__ import annotations

from app.rag.retriever import RetrievalResult
from app.rag.retriever import retrieve as _two_stage
# re-export: chains.py и lawyer/rag.py увезуваат и embed_query од retriever-от,
# па drop-in модулот мора да го нуди истото име
from app.rag.retriever import embed_query  # noqa: F401
from ideas.reranker_model import rerank_chunks

RERANK_POOL = 20   # колку fused пасуси одат на cross-encoder (10-20 на CPU)

# Ако сакаш и поширок базен: смени CASE_CANDIDATES = 12 -> 25 во
# app/rag/retriever.py. Прецизноста на реранкерот е токму тоа што ти дава
# право да ја олабавиш фаза 1 — поширок базен, а редот го средува тој.
# (Без реранкер, ширењето само дава повеќе влијание на BM25 клучните зборови.)


def retrieve(question: str,
             query_vector: list[float] | None = None) -> RetrievalResult:
    """Ист потпис и ист повратен тип како app.rag.retriever.retrieve."""
    result = _two_stage(question, query_vector)   # непроменет двофазен retriever
    if not result.chunks:
        return result

    pool, tail = result.chunks[:RERANK_POOL], result.chunks[RERANK_POOL:]
    ranked = rerank_chunks(question, pool)
    # единствената промена: редоследот на пасусите. „Опашката" под
    # RERANK_POOL останува на крајот за да не изгубиме кандидати.
    result.chunks = [chunk for chunk, _ in ranked] + tail
    return result
