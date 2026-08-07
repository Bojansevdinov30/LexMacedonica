"""ФИЛТРИРАЊЕ ПО МЕТАПОДАТОЦИ — за кога корпусот ќе порасне.

Денес сите 581 случаи се од ЕДНА област („Работни спорови", граѓанска), па
семантичкото пребарување нема од што да згреши. Кога ќе има кривични, стечајни,
семејни… и десетина илјади документи, ова веќе не важи: прашање за отказ ќе
влече и кривични одлуки што случајно зборуваат за „работно место", затоа што
cosine мери СЛИЧНОСТ НА ТЕКСТ, а не припадност на област.

Решението не е попаметен модел, туку ТВРД филтер пред мекото рангирање:
метаподатоците (legal_area, court, case_subtype, judge, outcome…) веќе се
запишани и во двете Chroma колекции, па филтерот е бесплатен — нула нови
вградувања, нула LLM повици.

    ФИЛТЕР (тврд, точен)  →  ФАЗА 1 резимеа  →  ФАЗА 2 hybrid+RRF  →  топ 3

КЛУЧНА ИДЕЈА: ФИЛТРИРАЈ САМО ВО ФАЗА 1
Фаза 2 и онака смее да гледа само во случаите што фаза 1 ги пуштила
(`allowed_case_ids`, app/rag/retriever.py:106). Значи ако филтерот го
примениш на резимеата, и векторскиот и BM25 делот го наследуваат САМИ —
не се пипа RRF, ниту BM25, ниту веројатноста. Една линија носи сè.

И уште нешто важно: Chroma го применува `where` ЗА ВРЕМЕ на пребарувањето,
не после. Добиваш топ-12 ВНАТРЕ во филтерот, а не топ-12 воопшто од кои
потоа фрлаш — токму затоа филтрирањето не ти го празни резултатот.

════════════════════════════════════════════════════════════════════════
КАДЕ ТОЧНО ОДИ ВО ПАЈПЛАЈНОТ  (линиите се состојба на 07.08.2026)
════════════════════════════════════════════════════════════════════════
Од UI до Chroma, по редослед — 6 мали промени:

 1. templates/index.html:12-17   — <select> кутии над textarea-та
                                   (id="f-area", id="f-court", …)
 2. static/js/chat.js:122        — во телото на барањето:
        const filters = { legal_area: areaSelect.value || undefined };
        postJSON("/api/chat", { question, history, filters })
 3. app/schemas.py:9-12          — ChatRequest: `filters: dict = {}`
 4. app/routers/chat.py:26       — answer_question(req.question, req.history,
                                                   req.filters)
 5. app/rag/chains.py:95         — answer_question(..., filters: dict | None = None)
    app/rag/chains.py:112        — retrieve(question, query_vector=qvec,
                                            filters=filters)
    app/rag/chains.py:105-106+167 — ⚠️ КЕШОТ (види «ЗАМКА 1» подолу)
 6. app/rag/retriever.py:89      — потпис: `..., filters: dict | None = None`
    app/rag/retriever.py:99-101  — во s_col.query(...) додади:
                                       where=to_where(filters)
    app/rag/retriever.py:115-116 — само за случајот кога резимеата ги нема:
                                       where = merge_where(where, to_where(filters))

Најбрз тест без ништо од UI-то (фаза 6 е доволна):
    retrieve("отказ без образложение", filters={"court": "Основен суд Кочани"})

⚠️ ЗАМКА 1 — СЕМАНТИЧКИОТ КЕШ НЕ ГИ ЗНАЕ ФИЛТРИТЕ
`cache.lookup(qvec)` (app/rag/chains.py:106) клучи САМО на векторот на
прашањето. Исто прашање со различен филтер = ист вектор = погоден кеш со
ТУЃ одговор. Најмала исправка — филтрите да патуваат со записот:
    chains.py:167   payload["_filters"] = filters or {}   пред cache.store(...)
    chains.py:106   if cached is not None and cached.get("_filters", {}) == (filters or {}):
(поинаку: посебна колекција по филтер, `semantic_cache_<хеш>` — почисто кога
филтрите ќе станат многу.)

⚠️ ЗАМКА 2 — ПРАГОТ И ВЕРОЈАТНОСТА СЕ СТЕСНУВААТ ЗАЕДНО СО ФИЛТЕРОТ
Тесен филтер = помал базен = послаб најдобар погодок. Затоа:
  • max_similarity паѓа → почесто „Не знам" под 0.43 (config.py). Тоа е
    ТОЧНО однесување, не бубачка: одговор нема од што да се склопи.
  • probability.estimate() враќа None под 3 споредливи случаи
    (app/rag/probability.py:47) → картичките остануваат, процентот исчезнува.
Правило за интерфејсот: кога филтерот врати премалку, кажи го тоа на
корисникот („со овие филтри има само 2 слични случаи"), не глуми сигурност.

⚠️ ЗАМКА 3 — ДЕНЕС ФИЛТЕРОТ БИ ОДМОГНАЛ
66 од 581 случаи ги немаат овие полиња (порталот врати друга одлука под
истиот број при збогатувањето, па метаподатоците останаа празни `""`).
Филтер `legal_area="Граѓанска област"` тивко ги исклучува ТИЕ 66. Со еден
корпус и една област — филтерот е чиста загуба. Има смисла дури кога
областите се навистина повеќе.

ДАТУМИ (не работат вака)
`date` е текст „27.05.2010", а `$gte/$lte` во Chroma се БРОЈЧАНИ. За опсег
по година додади цел број при индексирањето (тоа е ре-индекс, но $0 —
вградувањата се кеширани во embeddings.npz):
    app/ingest/build_index.py:101-104  и  app/ingest/summarize.py:89-92
        "year": int(case["date"][-4:]) if case["date"] else 0
и потоа: {"year": {"$gte": 2015}}

КОГА ФИЛТЕРОТ ВЕЌЕ НЕ Е ДОВОЛЕН (десетици илјади документи)
  • `_bm25_index()` (app/rag/retriever.py:62-72) се гради во меморија при
    секое стартување — линеарно со корпусот. На ~100K пасуси тоа е минути
    по процес: тука BM25 се сели во база (SQLite FTS5) или се снима на диск.
  • CASE_CANDIDATES=12 од 50 000 случаи е многу потесно грло одошто 12 од
    581 — со филтер по област тоа пак станува „12 од оние што се во играта".
"""
from __future__ import annotations

from app.ingest.build_index import tokenize
from app.rag.retriever import (CANDIDATES, CASE_CANDIDATES, RRF_K, WIDE_SEARCH,
                               CaseCandidate, RetrievalResult, RetrievedChunk,
                               _bm25_index, _summaries_collection)
from app.rag.retriever import embed_query  # noqa: F401  (drop-in паритет)
from app.vectorstore import collection, sim

# Полиња што ги има во МЕТАПОДАТОЦИТЕ и на резимеата и на пасусите
# (app/ingest/summarize.py:89-92 и app/ingest/build_index.py:101-104).
# Само по нив смее да се филтрира — сè друго е печатна грешка, не филтер.
FILTERABLE = ("legal_area", "court", "case_type", "case_subtype",
              "judge", "outcome", "foundation_type", "foundation")


def to_where(filters: dict | None) -> dict | None:
    """{"court": "Основен суд Велес"} -> Chroma `where`; None ако нема филтер.

    Вредност може да биде и листа: {"outcome": ["УСВОЕНО", "ДЕЛУМНО УСВОЕНО"]}
    станува `$in`. Празни вредности се игнорираат — така „сите области" во
    паѓачкото мени е едноставно празен string, без посебен случај.
    """
    if not filters:
        return None

    clauses = []
    for field, value in filters.items():
        if field not in FILTERABLE:
            raise ValueError(f"непознато поле за филтрирање: {field}")
        if not value:
            continue                      # „сите" — не додавај услов
        clauses.append({field: {"$in": list(value)} if isinstance(value, (list, tuple))
                        else {"$eq": value}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def merge_where(a: dict | None, b: dict | None) -> dict | None:
    """Спој два `where` услови со И. Chroma нема вгнездување на исто ниво —
    два услови мора експлицитно да се завиткаат во `$and`."""
    if a is None:
        return b
    if b is None:
        return a
    return {"$and": [a, b]}


def values_for(field: str) -> list[str]:
    """Постојните вредности на едно поле — за полнење на паѓачкото мени.

    Чита од SQLite (data/lex.db), не од Chroma: `SELECT DISTINCT` е работа за
    база, додека Chroma би морала да ги повлече сите метаподатоци во меморија.
    """
    if field not in FILTERABLE:
        raise ValueError(f"непознато поле за филтрирање: {field}")

    from sqlalchemy import text as sql_text

    from app.ingest.structure import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(sql_text(
            f"SELECT DISTINCT {field} FROM cases "
            f"WHERE {field} != '' ORDER BY {field}"))
        return [r[0] for r in rows]


def _meta_ok(metadata: dict, filters: dict | None) -> bool:
    """Истиот филтер, но во Python — за BM25, кој не знае за `where`."""
    for field, value in (filters or {}).items():
        if not value:
            continue
        got = metadata.get(field, "")
        if isinstance(value, (list, tuple)):
            if got not in value:
                return False
        elif got != value:
            return False
    return True


def retrieve(question: str,
             query_vector: list[float] | None = None,
             filters: dict | None = None) -> RetrievalResult:
    """Истиот двофазен retriever + тврд филтер. Копија на
    app/rag/retriever.py:89-160 со ТРИ променети места, обележани со ← НОВО.

    `filters=None` значи однесување идентично на живиот retriever, па новиот
    аргумент не расипува ниту еден постоечки повик (chains.py, lawyer/rag.py:71).
    """
    if query_vector is None:
        query_vector = embed_query(question)

    meta_where = to_where(filters)                                   # ← НОВО

    # ---------- фаза 1: кои СЛУЧАИ одговараат на ситуацијата ----------
    case_hits: list[CaseCandidate] = []
    allowed_case_ids: set[str] | None = None
    s_col = _summaries_collection()
    if s_col is not None:
        r = s_col.query(query_embeddings=[query_vector],
                        n_results=min(CASE_CANDIDATES, s_col.count()),
                        where=meta_where,                            # ← НОВО
                        include=["documents", "metadatas", "distances"])
        for m, doc, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0]):
            meta = dict(m)
            meta["summary"] = doc
            case_hits.append(CaseCandidate(meta, sim(dist)))
        allowed_case_ids = {c.metadata["case_id"] for c in case_hits}

    def case_ok(metadata: dict) -> bool:
        return allowed_case_ids is None or metadata.get("case_id") in allowed_case_ids

    # ---------- фаза 2: најдобри ПАСУСИ во тие случаи ----------
    candidates: dict[str, RetrievedChunk] = {}

    # Ако фаза 1 работела, `$in` веќе го носи филтерот — метаподатоците се
    # додаваат само за случајот кога резимеата ги нема (single-stage fallback).
    where = ({"case_id": {"$in": sorted(allowed_case_ids)}} if allowed_case_ids
             else meta_where)                                        # ← НОВО
    r = collection("chunks").query(
        query_embeddings=[query_vector], n_results=CANDIDATES, where=where,
        include=["documents", "metadatas", "distances"])
    vec_rank: dict[str, int] = {}
    for rank, (cid, text, m, dist) in enumerate(zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])):
        candidates[cid] = RetrievedChunk(cid, text, dict(m), vector_similarity=sim(dist))
        vec_rank[cid] = rank

    # BM25 не може да филтрира додека скорира: прво скорирај, па филтрирај
    idx = _bm25_index()
    scores = idx["bm25"].get_scores(tokenize(question))
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_rank: dict[str, int] = {}
    rank = 0
    for i in bm25_order[:WIDE_SEARCH]:
        if scores[i] <= 0 or not case_ok(idx["metadatas"][i]):
            continue
        # без фаза 1 филтерот овде се проверува рачно врз метаподатоците
        if allowed_case_ids is None and not _meta_ok(idx["metadatas"][i], filters):
            continue                                                 # ← НОВО
        cid = idx["ids"][i]
        if cid not in candidates:
            candidates[cid] = RetrievedChunk(cid, idx["texts"][i], dict(idx["metadatas"][i]))
        bm25_rank[cid] = rank
        rank += 1
        if rank >= CANDIDATES:
            break

    # RRF фузија — непроменета
    for cid, chunk in candidates.items():
        chunk.rrf_score = ((1.0 / (RRF_K + vec_rank[cid]) if cid in vec_rank else 0.0)
                           + (1.0 / (RRF_K + bm25_rank[cid]) if cid in bm25_rank else 0.0))
    fused = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)

    max_sim = (case_hits[0].similarity if case_hits
               else max((c.vector_similarity for c in fused), default=0.0))
    return RetrievalResult(chunks=fused, cases=case_hits, max_similarity=max_sim)
