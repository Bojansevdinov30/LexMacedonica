"""Заедничко вчитување и скорирање со cross-encoder — го делат двете
drop-in верзии на retriever-от (retriever_reranked.py и retriever_rerank_only.py).

Cross-encoder (реранкер) = модел што го чита прашањето и пасусот ЗАЕДНО и
враќа еден скор за релевантност. Попрецизен е од cosine меѓу два ОДДЕЛНО
вградени вектори (bi-encoder), зашто гледа како зборовите од прашањето се
однесуваат СПРЕМА зборовите од пасусот. Цената: едно поминување низ моделот
за СЕКОЈ кандидат, па не може да се пушти врз цел корпус — само врз
кандидатите што евтиното пребарување веќе ги извади.

При import ништо не се симнува: тешките импорти се внатре во функцијата.
"""
from __future__ import annotations

import math
from functools import lru_cache

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # мултијазичен, работи и на македонски
MAX_LENGTH = 512    # прашање + пасус се сечат на толку токени


@lru_cache(maxsize=1)
def load_reranker():
    """Singleton како _llm() во chains.py — моделот се вчитува еднаш по процес."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise SystemExit(
            "Инсталирај sentence-transformers за да го пробаш ова:\n"
            "    pip install sentence-transformers\n"
            "(влече PyTorch ~2.5 GB; моделот е ~2.3 GB при прво вчитување)")
    return CrossEncoder(RERANK_MODEL, max_length=MAX_LENGTH)


def rerank_chunks(question: str, chunks: list) -> list[tuple]:
    """Скорира (прашање, пасус) парови. Враќа [(chunk, score), ...] опаѓачки.

    chunks се RetrievedChunk објекти (имаат .text). score = sigmoid(логит)
    во (0,1) — читливо, но НЕ е калибрирана веројатност: 0.9 значи само
    „повисоко од 0.5", не „90% сигурно релевантно".
    """
    if not chunks:
        return []
    logits = load_reranker().predict([(question, c.text) for c in chunks])
    scored = [(c, 1.0 / (1.0 + math.exp(-float(lg))))
              for c, lg in zip(chunks, logits)]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
