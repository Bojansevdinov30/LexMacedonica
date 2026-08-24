"""АВТОМАТСКО ФИЛТРИРАЊЕ ПО МЕТАПОДАТОЦИ — за кога корпусот ќе порасне.

Денес сите 581 случаи се од ЕДНА област („Работни спорови"), па семантичкото
пребарување нема од што да згреши. Кога ќе има кривични, стечајни, семејни… и
десетина илјади документи, прашање за отказ ќе влече и кривични одлуки што
случајно зборуваат за „работно место": cosine мери СЛИЧНОСТ НА ТЕКСТ, а не
припадност на област.

Метаподатоците (court, judge, legal_area…) веќе се запишани во двете Chroma
колекции, па филтерот е бесплатен. Прашањето е само КОЈ филтер — и тука има
две школи:

  паѓачки менија   корисникот сам бира област/суд пред да прашај
  автоматски       ФИЛТЕРОТ СЕ ЧИТА ОД САМОТО ПРАШАЊЕ ← ова е овој фајл

Втората е вистинската за оваа апликација: граѓанинот пишува „ме отпуштија во
Кочани", не пипа <select>. Плус — не бара НИТУ ЕДНА промена во HTML/JS/схемите,
целата логика седи зад retrieve().

════════════════════════════════════════════════════════════════════════
ТРИ СЛОЈА НА ВАДЕЊЕ ФИЛТРИ — од најевтин кон најскап
════════════════════════════════════════════════════════════════════════
1. ЗАТВОРЕН РЕЧНИК (regex/точно совпаѓање, $0, 0 ms)
   Судовите се 19, судиите 101 — вредностите се ЗНАЕНИ. Значи ова не е
   задача за генерирање („измисли ми филтер"), туку за препознавање: дали
   некоја од познатите вредности се спомнува во прашањето. Затоа LLM тука е
   непотребен — обичен speech-match е и поточен и бесплатен.

2. ЦЕНТРОИД ПО ОБЛАСТ (вектори, $0, 0 ms — го користи ВЕЌЕ пресметаниот
   вектор на прашањето од chains.py:105)
   Никој не пишува „работни спорови" — пишува „не ми ја исплатија платата".
   Значи областа не се препознава, туку се КЛАСИФИЦИРА. Центроид на област =
   просек од векторите на нејзините резимеа (веќе платени, седат во Chroma);
   класификација = cosine до секој центроид. Нула нови повици.

3. LLM КЛАСИФИКАТОР (~$0.00003 + ~0.7 s по прашање)
   Само ако слој 2 не стигне. Не му се дава слобода да измислува — му се дава
   ЗАТВОРЕНА листа и правото да каже „НЕПОЗНАТО". Стои исклучен по default.

(Во LangChain ова се вика SelfQueryRetriever — истата идеја, спакувана така
што не гледаш кога и колку чини. Ова е истото, напишано рачно.)

════════════════════════════════════════════════════════════════════════
ИЗМЕРЕНО ВРЗ ОВОЈ КОРПУС (leave-one-out врз 581 резиме, $0)
════════════════════════════════════════════════════════════════════════
Класификација со центроиди, по поле:

  9 БЛИСКИ ознаки (foundation: „Исплата на плати" vs „Надоместоци од работен
  однос" vs „Други решенија од работен однос" — практично исто нешто)
      top-1 точност 45.7%  ← ПОЛОШО од тупо погодување на најголемата (64%)

  5 ЈАСНО РАЗЛИЧНИ теми (плати / придонеси ПИОМ / штета / поништување
  одлука / трансформација на работен однос)
      top-1 точност           77.0%
      со маргина ≥ 0.05       се пали на 47% од прашањата, точна 91.2%
      со маргина ≥ 0.10       се пали на 19% од прашањата, точна 95.7%

Истото, вживо низ `guess_area()` врз 40 резимеа: се пали 8 пати, точно 4/8 —
затоа USE_AREA_CLASSIFIER стои False додека полето е „foundation".

Заклучок што ја обликува целата политика подолу:
  • Класификацијата вреди САМО кога класите се навистина различни теми — што
    е токму иднината за која се пишува овој фајл (кривично vs работно vs
    семејно се далеку поразлични од овие пет).
  • МАРГИНАТА е механизмот за безбедност: разликата меѓу првиот и вториот
    центроид ја претвора точноста во сигурност. Кога маргината е мала —
    НЕ ФИЛТРИРАЈ. Подобро без филтер отколку со погрешен.
  • Затоа филтерот никогаш не смее да биде последен збор → види «ПРОШИРУВАЊЕ».

ЗОШТО Е ТОА ТОЛКУ ВАЖНО: тврдиот филтер ИСКЛУЧУВА, меките рангирања само
ПОМЕСТУВААТ. Ако RRF го стави вистинскиот случај на 4. место — сè уште е ту.
Ако филтерот погреши со областа — вистинскиот случај го НЕМА и одговорот се
гради врз погрешни одлуки, со полна самодоверба. Затоа сето долу е свесно
плашливо.

ПРОШИРУВАЊЕ (најважниот дел од кодот)
Филтрирај → измери → ако филтерот го расипал резултатот, повтори БЕЗ него:
премалку случаи или паднат max_similarity = врати се на неотфилтрирано
пребарување. Чини уште едно Chroma барање (милисекунди, $0) и го претвора
филтерот од ризик во чиста добивка: кога помага — помага, кога штети —
исчезнува.

Но чесно: проширувањето спасува од ПРАЗЕН резултат, не од ПОГРЕШЕН. Филтер
со погрешна област што сепак враќа 12 случаи изгледа здраво од сите страни —
затоа слој 2 е зад маргина и зад прекинувач, а не пуштен слободно.

════════════════════════════════════════════════════════════════════════
КАДЕ ТОЧНО ОДИ ВО ПАЈПЛАЈНОТ  (линиите се состојба на 07.08.2026)
════════════════════════════════════════════════════════════════════════
Автоматската верзија е ЕДНА линија — нема UI, нема схеми, нема JS:

    app/rag/chains.py:27
        # from app.rag.retriever import embed_query, retrieve
        from ideas.filter_metadata import embed_query, retrieve

Тоа е сè: chains.py:112 веќе повикува retrieve(question, query_vector=qvec),
а филтрите се вадат внатре. (Истото важи и за app/lawyer/rag.py:71.)

Ако сакаш и рачни филтри од интерфејс, тие остануваат можни — retrieve()
прима `filters=` и тогаш прескокнува автоматско вадење. Плумбингот тогаш е:
    templates/index.html:12-17 → static/js/chat.js:122 → app/schemas.py:9-12
    → app/routers/chat.py:26 → app/rag/chains.py:95 → овде

Тест без ништо друго:
    from ideas.filter_metadata import extract_filters, retrieve
    extract_filters("ме отпуштија, судот во Кочани го одби барањето")
    retrieve("работодавачот не ми плати три плати")   # сам си вади филтри

⚠️ ЗАМКА 1 — НИКОГАШ НЕ ФИЛТРИРАЈ ПО `outcome`
Тоа е ОДГОВОРОТ, не прашањето. „Дали ќе добијам?" + филтер outcome=УСВОЕНО
дава 100% веројатност — совршено пресметана глупост. Затоа `outcome` е
намерно исфрлен од AUTO_FIELDS, иако е филтрабилен рачно.

⚠️ ЗАМКА 2 — СЕМАНТИЧКИОТ КЕШ НЕ ГИ ЗНАЕ ФИЛТРИТЕ
`cache.lookup(qvec)` (app/rag/chains.py:106) клучи само на векторот. Две
прашања со блиско значење (праг 0.95) но различен извлечен филтер — на пр.
со и без спомнат суд — ќе делат одговор. Најмала исправка:
    chains.py:167   payload["_filters"] = result.filters_used   пред cache.store(...)
    chains.py:106   if cached is not None and cached.get("_filters", {}) == ...
(Кај автоматските филтри ова е поретко одошто кај рачните — филтерот е
функција од самото прашање — но не е невозможно.)

⚠️ ЗАМКА 3 — ПРАГОТ И ВЕРОЈАТНОСТА СЕ СТЕСНУВААТ ЗАЕДНО СО ФИЛТЕРОТ
Тесен филтер = помал базен: max_similarity паѓа кон прагот 0.43 (почесто
„Не знам"), а probability.estimate() враќа None под 3 споредливи случаи
(app/rag/probability.py:47) — картичките остануваат, процентот исчезнува.
Токму против ова е ПРОШИРУВАЊЕТО подолу. Кога и по проширување останува
малку, кажи му го тоа на корисникот наместо да глумиш сигурност.

⚠️ ЗАМКА 4 — 66 СЛУЧАИ СЕ БЕЗ МЕТАПОДАТОЦИ
Кај 66 од 581 полињата се празни `""` (порталот врати друга одлука под истиот
број при збогатувањето). Секој филтер тивко ги исклучува ТИЕ 66 — уште една
причина зошто проширувањето постои.

ДАТУМИ (не работат вака)
`date` е текст „27.05.2010", а `$gte/$lte` во Chroma се БРОЈЧАНИ. За опсег по
година додади цел број при индексирањето (ре-индекс, но $0 — вградувањата се
кеширани во embeddings.npz):
    app/ingest/build_index.py:101-104  и  app/ingest/summarize.py:89-92
        "year": int(case["date"][-4:]) if case["date"] else 0
Дури тогаш вклучи YEAR_FILTER = True подолу.

КОГА ОВА ВЕЌЕ НЕ Е ДОВОЛНО (десетици илјади документи)
  • `_bm25_index()` (app/rag/retriever.py:62-72) се гради во меморија при
    секое стартување — линеарно со корпусот. На ~100K пасуси тоа е минути по
    процес: тука BM25 се сели во база (SQLite FTS5) или се снима на диск.
  • `_centroids()` подолу ги вчитува СИТЕ вектори на резимеата за да ги
    просече. На 50K резимеа тоа е ~300 MB по процес — центроидите тогаш се
    пресметуваат еднаш при индексирање и се снимаат во мал .npz.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from app.config import MIN_SIMILARITY_FOR_ANSWER
from app.ingest.build_index import tokenize
from app.rag.retriever import (CANDIDATES, CASE_CANDIDATES, RRF_K, WIDE_SEARCH,
                               CaseCandidate, RetrievalResult, RetrievedChunk,
                               _bm25_index, _summaries_collection)
from app.rag.retriever import embed_query  # noqa: F401  (drop-in паритет)
from app.vectorstore import collection, sim

# Полиња што ги има во метаподатоците и на резимеата и на пасусите
# (app/ingest/summarize.py:89-92 и app/ingest/build_index.py:101-104).
FILTERABLE = ("legal_area", "court", "case_type", "case_subtype",
              "judge", "outcome", "foundation_type", "foundation")

# Што се вади АВТОМАТСКИ од прашањето. `outcome` намерно го нема — замка 1.
VOCABULARY_FIELDS = ("court", "judge")   # слој 1: корисникот ги именува
AREA_FIELD = "foundation"                # слој 2: се класифицира
# ^ денес „foundation" е единственото поле со повеќе вредности, па служи како
#   демонстрација. Кога корпусот ќе има повеќе области → "legal_area".

# Слој 2 стои ИСКЛУЧЕН денес — измерено е дека врз „foundation" греши
# половина од своите палења (ознаките се блиски по значење; види ИЗМЕРЕНО
# во докстрингот). Вклучи го кога областите ќе бидат навистина различни.
USE_AREA_CLASSIFIER = False

MIN_CLASS_SIZE = 5      # класа со помалку случаи нема доверлив центроид
CONFIDENCE_MARGIN = 0.05  # 1. минус 2. центроид; помало = не филтрирај (91% точност)
MIN_CASES_AFTER_FILTER = 5  # под ова се проширува (веројатноста бара 3 меритни)
YEAR_FILTER = False     # вклучи дури откако ќе индексираш цел број `year`


# ═══════════════════════ Chroma where-услови ═══════════════════════

def to_where(filters: dict | None) -> dict | None:
    """{"court": "Основен суд Велес"} -> Chroma `where`; None ако нема филтер.

    Вредноста може да е и листа: {"outcome": ["УСВОЕНО", "ДЕЛУМНО УСВОЕНО"]}
    станува `$in`. Празни вредности се игнорираат.
    """
    if not filters:
        return None

    clauses = []
    for field, value in filters.items():
        if field not in FILTERABLE:
            raise ValueError(f"непознато поле за филтрирање: {field}")
        if not value:
            continue
        clauses.append({field: {"$in": list(value)} if isinstance(value, (list, tuple))
                        else {"$eq": value}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def merge_where(a: dict | None, b: dict | None) -> dict | None:
    """Спој два услови со И — Chroma бара експлицитно `$and`."""
    if a is None:
        return b
    if b is None:
        return a
    return {"$and": [a, b]}


@lru_cache(maxsize=8)
def values_for(field: str) -> tuple[str, ...]:
    """Постојните вредности на едно поле — речникот за слој 1.

    Чита од SQLite (data/lex.db): `SELECT DISTINCT` е работа за база, додека
    Chroma би морала да ги повлече сите метаподатоци во меморија.
    """
    if field not in FILTERABLE:
        raise ValueError(f"непознато поле за филтрирање: {field}")

    from sqlalchemy import text as sql_text

    from app.ingest.structure import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(sql_text(
            f"SELECT DISTINCT {field} FROM cases "
            f"WHERE {field} != '' ORDER BY {field}"))
        return tuple(r[0] for r in rows)


# ═══════════════ слој 1: затворен речник (regex, $0) ═══════════════

# зборови што ги има во СЕКОЈ суд — не носат информација кој суд е
_NOISE = {"основен", "граѓански", "суд", "судот", "апелационен", "врховен",
          "со", "одделение", "во", "и"}


def _aliases(value: str) -> list[str]:
    """„Основен суд Штип со одделение во Пробиштип" -> ['штип', 'пробиштип'].

    Корисникот пишува „во Кочани", не целото службено име, па покрај целата
    вредност се памети и секој нејзин информативен збор.
    """
    words = [w for w in re.findall(r"\w+", value.lower())
             if w not in _NOISE and len(w) > 3]
    return [value.lower(), *words]


def match_vocabulary(question: str, field: str) -> str | None:
    """Која позната вредност на `field` се спомнува во прашањето?

    Победува НАЈДОЛГОТО совпаѓање: „Основен суд Битола со одделение Демир
    Хисар" е поспецифично од само „Битола".
    """
    q = " " + " ".join(re.findall(r"\w+", question.lower())) + " "
    best, best_len = None, 0
    for value in values_for(field):
        for alias in _aliases(value):
            # имињата на судиите се лични имиња — бараме го ЦЕЛОТО име, инаку
            # едно „Марија" во прашањето би филтрирало по случаен судија
            if field == "judge" and alias != value.lower():
                continue
            if f" {alias} " in q and len(alias) > best_len:
                best, best_len = value, len(alias)
    return best


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_year(question: str) -> int | None:
    """„одлуки по 2015", „од 2018 наваму" -> 2015 / 2018. Бара индексиран
    цел број `year` (види ДАТУМИ во докстрингот), затоа е зад YEAR_FILTER."""
    m = _YEAR_RE.search(question)
    return int(m.group()) if m else None


# ═══════════ слој 2: класификација со центроиди ($0) ═══════════

@lru_cache(maxsize=4)
def _centroids(field: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Просечен вектор по вредност на `field`, од резимеата во Chroma.

    Векторите се веќе платени и нормализирани, па центроидот е обичен просек
    (пак нормализиран, за cosine да остане обичен скаларен производ).
    Класи со помалку од MIN_CLASS_SIZE случаи се испуштаат — просек од два
    вектори не е центроид, туку шум.
    """
    data = collection("summaries").get(include=["embeddings", "metadatas"])
    vectors = np.array(data["embeddings"], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = [m.get(field, "") for m in data["metadatas"]]

    names, rows = [], []
    for value in sorted(set(labels)):
        if not value:
            continue
        idx = [i for i, l in enumerate(labels) if l == value]
        if len(idx) < MIN_CLASS_SIZE:
            continue
        c = vectors[idx].mean(axis=0)
        names.append(value)
        rows.append(c / np.linalg.norm(c))
    return tuple(names), np.array(rows, dtype="float32")


def guess_area(query_vector: list[float], field: str = AREA_FIELD) -> str | None:
    """Најблискиот центроид — но САМО ако е убедливо поблизок од вториот.

    Маргината (1. минус 2.) е разликата меѓу „ова е јасно работен спор" и
    „вака-така личи на сè". Под CONFIDENCE_MARGIN враќа None = не филтрирај.
    """
    names, cents = _centroids(field)
    if len(names) < 2:
        return None   # една класа = филтерот не носи ништо

    q = np.array(query_vector, dtype="float32")
    q /= np.linalg.norm(q)
    scores = cents @ q
    order = np.argsort(-scores)
    if scores[order[0]] - scores[order[1]] < CONFIDENCE_MARGIN:
        return None
    return names[order[0]]


# ═══════════ слој 3: LLM класификатор (по избор, ~$0.00003) ═══════════

USE_LLM_CLASSIFIER = False   # вклучи само ако слој 2 не стигнува

_CLASSIFY_PROMPT = (
    "Дадена е ситуација од граѓанин и затворена листа на правни области. "
    "Врати ТОЧНО една ставка од листата, збор до збор, или НЕПОЗНАТО ако "
    "ситуацијата не спаѓа јасно во ниту една. Не објаснувај ништо.\n"
    "Листа:\n{options}"
)


def classify_with_llm(question: str, field: str = AREA_FIELD) -> str | None:
    """Затворена класификација, не генерирање: сè што не е од листата се фрла.

    Цена: ~150 влезни + 5 излезни токени на gpt-4o-mini ≈ $0.00003 по
    прашање; латенција ~0.5-1 s. Тоа е ВТОР LLM повик во синџир што денес
    има еден (плус кондензација), па се плаќа и во време, не само во пари.
    """
    from langchain_openai import ChatOpenAI

    options = values_for(field)
    if len(options) < 2:
        return None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=20)
    answer = llm.invoke([
        ("system", _CLASSIFY_PROMPT.format(options="\n".join(f"- {o}" for o in options))),
        ("user", question),
    ]).content.strip()
    return answer if answer in options else None


# ═══════════════════════ вадење на филтрите ═══════════════════════

def extract_filters(question: str,
                    query_vector: list[float] | None = None) -> dict:
    """Прочитај ги филтрите ОД САМОТО ПРАШАЊЕ. Празен dict = без филтер.

    Редоследот е намерен: прво бесплатното и точното (речник), потоа
    бесплатното и веројатното (центроид), и дури на крај платеното (LLM).
    """
    filters: dict = {}

    for field in VOCABULARY_FIELDS:
        value = match_vocabulary(question, field)
        if value:
            filters[field] = value

    if AREA_FIELD not in filters:
        area = None
        if USE_AREA_CLASSIFIER and query_vector is not None:
            area = guess_area(query_vector)
        if area is None and USE_LLM_CLASSIFIER:
            area = classify_with_llm(question)
        if area:
            filters[AREA_FIELD] = area

    if YEAR_FILTER:
        year = extract_year(question)
        if year:
            filters["year"] = {"$gte": year}   # бара индексиран цел број

    return filters


# ═══════════════════════ пребарување ═══════════════════════

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


def _search(question: str, query_vector: list[float],
            filters: dict | None) -> RetrievalResult:
    """Копија на app/rag/retriever.py:89-160 со ТРИ променети места (← НОВО).

    КЛУЧНО: филтерот се применува само во ФАЗА 1. Фаза 2 и онака гледа само
    во случаите што фаза 1 ги пуштила (`allowed_case_ids`), па и векторскиот
    и BM25 делот го наследуваат филтерот сами — RRF, веројатноста и прагот
    остануваат недопрени. Плус, Chroma го применува `where` ЗА ВРЕМЕ на
    пребарувањето: добиваш топ-12 ВНАТРЕ во филтерот, а не топ-12 воопшто од
    кои потоа фрлаш.
    """
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
            continue
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


def retrieve(question: str,
             query_vector: list[float] | None = None,
             filters: dict | None = None) -> RetrievalResult:
    """Ист потпис како app.rag.retriever.retrieve + автоматски филтри.

    `filters=None` (нормалниот повик од chains.py:112) значи „извади ги сам".
    Ако сакаш рачни од интерфејс, подај ги; `filters={}` значи изречно
    БЕЗ филтер — истото однесување како живиот retriever.
    """
    if query_vector is None:
        query_vector = embed_query(question)

    if filters is None:
        filters = extract_filters(question, query_vector)

    result = _search(question, query_vector, filters)

    # ПРОШИРУВАЊЕ: филтерот нема право да го расипе одговорот. Ако по него
    # останало премалку случаи или најдобриот погодок падне под прагот
    # „Не знам", пребарувањето се повторува без филтер (уште едно Chroma
    # барање, милисекунди, $0). Така филтерот е чиста добивка: помага кога
    # помага, а кога штети — исчезнува.
    if filters and (len(result.cases) < MIN_CASES_AFTER_FILTER
                    or result.max_similarity < MIN_SIMILARITY_FOR_ANSWER):
        print(f"[филтер] {filters} даде {len(result.cases)} случаи "
              f"(max_sim={result.max_similarity:.3f}) -> проширувам без филтер")
        result = _search(question, query_vector, None)
        filters = {}

    # кои филтри навистина останаа — за кешот (замка 2) и за да може
    # интерфејсот чесно да каже „пребарував само во Основен суд Кочани".
    # Кога ова ќе стане живо, полето се додава во RetrievalResult
    # (app/rag/retriever.py:48-52) наместо да се лепи вака одвон.
    result.filters_used = filters
    return result
