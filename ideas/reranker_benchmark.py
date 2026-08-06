"""Мери ги двете drop-in варијанти против живиот retriever — пред да смениш
било што во app/.

Два режима:

  python -m ideas.reranker_benchmark
      ЛАТЕНЦИЈА + КОЛКУ СЕ МЕНУВА ИЗБОРОТ. По прашање се пушта живиот
      двофазен retriever, варијанта Б и варијанта А; се печати времето и
      колку од топ-3 пасуси се совпаѓаат со живите. Ниска согласност НЕ
      значи дека едното е подобро — значи дека изборот е РАЗЛИЧЕН и вреди
      рачно да се прочитаат пасусите.

  python -m ideas.reranker_benchmark --gate
      ГО МЕРИ ПРАГОТ „НЕ ЗНАМ" ЗА ВАРИЈАНТА А. max_similarity таму е
      rerank-скор (друга скала од summary-сличноста), па живиот праг 0.43
      не важи. Скриптата пушта прашања на тема и вон тема и предлага праг
      на средината меѓу двете групи. Потоа:
          set LEX_MIN_SIMILARITY_FOR_ANSWER=<предложениот број>

Трошок: по едно вградување (embedding) на прашање, ~$0.0002 вкупно.
LLM НЕ се повикува — ова мери само пребарување.
Предуслов: pip install sentence-transformers
"""
from __future__ import annotations

import sys
import time

ON_TOPIC = [
    "Мојот работодавач не ми ја исплати последната плата. Што можам да направам?",
    "Добив отказ без образложение по 10 години работа.",
    "Дали имам право на регрес за годишен одмор?",
    "Работодавачот ми смени работно место без согласност.",
]
OFF_TOPIC = [
    "рецепт за ајвар со печки",
    "Кој го освои светското првенство во фудбал 2022?",
    "Како да инсталирам Windows на нов компјутер?",
]


def _variants():
    """(име, retrieve-функција) за трите пајплајни."""
    from app.rag import retriever as live
    from ideas import retriever_rerank_only as var_a
    from ideas import retriever_reranked as var_b
    return [("живо (2 фази + RRF)", live.retrieve),
            ("Б (2 фази + реранкер)", var_b.retrieve),
            ("А (RRF + реранкер)", var_a.retrieve)]


def compare() -> None:
    from app.rag.retriever import embed_query
    from ideas.reranker_model import load_reranker

    print("вчитувам reranker (првиот пат симнува ~2.3 GB)…")
    t0 = time.perf_counter()
    load_reranker()
    print(f"  вчитан за {time.perf_counter() - t0:.1f}s\n")

    variants = _variants()
    for q in ON_TOPIC:
        print(f"Прашање: {q[:58]}")
        baseline = None
        for name, fn in variants:
            qvec = embed_query(q)          # исто вградување за сите
            t0 = time.perf_counter()
            res = fn(q, query_vector=qvec)
            dt = time.perf_counter() - t0
            top3 = [c.chunk_id for c in res.top_chunks()]
            if baseline is None:
                baseline = top3
            same = len(set(top3) & set(baseline))
            print(f"  {name:<24} {dt:5.2f}s   исти како живо: {same}/3   "
                  f"max_sim={res.max_similarity:.3f}  случаи={len(res.cases)}")
        print()


def gate() -> None:
    """Предложи праг за варијанта А (нејзиниот max_similarity е друга скала)."""
    from app.config import MIN_SIMILARITY_FOR_ANSWER
    from ideas import retriever_rerank_only as var_a
    from ideas.reranker_model import load_reranker

    print("вчитувам reranker…")
    load_reranker()

    print("\nна тема:")
    on = []
    for q in ON_TOPIC:
        s = var_a.retrieve(q).max_similarity
        on.append(s)
        print(f"  {s:.3f}  {q[:52]}")
    print("вон тема:")
    off = []
    for q in OFF_TOPIC:
        s = var_a.retrieve(q).max_similarity
        off.append(s)
        print(f"  {s:.3f}  {q[:52]}")

    lo, hi = min(on), max(off)
    print(f"\nнајнизок на тема = {lo:.3f}   највисок вон тема = {hi:.3f}")
    if lo > hi:
        print(f"предложен праг: {(lo + hi) / 2:.2f}   (на средината — има јасна граница)")
        print(f"постави го со:  set LEX_MIN_SIMILARITY_FOR_ANSWER={(lo + hi) / 2:.2f}")
    else:
        print("ГРУПИТЕ СЕ ПРЕКЛОПУВААТ — со овој сет прашања нема чист праг.")
        print("Пробај повеќе прашања или задржи ја варијанта Б, каде што")
        print(f"постојниот праг {MIN_SIMILARITY_FOR_ANSWER} останува валиден.")


if __name__ == "__main__":
    gate() if "--gate" in sys.argv else compare()
