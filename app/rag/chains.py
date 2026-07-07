"""The main answer chain for Правен асистент (Interface 1).

Pipeline per question:
  1. semantic cache lookup            (hit = free, instant answer)
  2. hybrid retrieval                 (BM25 + vectors + RRF, retriever.py)
  3. confidence gate                  (low similarity -> honest "Не знам")
  4. probability from case outcomes   (our statistic, probability.py)
  5. LLM writes the answer            (Macedonian, cites the top cases)
  6. self-check pass                  (second cheap call verifies grounding)
  7. cache store + return
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, MIN_SIMILARITY_FOR_ANSWER, TOP_CASES_FOR_PROBABILITY
from app.costs import log_llm_response
from app.rag import cache
from app.rag.probability import estimate, READABLE
from app.rag.retriever import retrieve, embed_query

NO_ANSWER = {
    "answer": ("Не знам — во мојата база нема доволно слични случаи за да дадам "
               "поуздана проценка за оваа ситуација. Ве молам обидете се да ја "
               "опишете поинаку, или консултирајте адвокат."),
    "probability": None,
    "cases": [],
}

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ти си правен асистент за македонско право. Одговараш САМО на македонски јазик, "
     "јасно и разбирливо за обични граѓани.\n"
     "Правила:\n"
     "1. Користи ИСКЛУЧИВО информации од дадените извадоци од судски одлуки — ништо друго.\n"
     "2. Кажи што најверојатно ќе се случи во ситуацијата на корисникот, врз основа на "
     "сличните случаи, и зошто.\n"
     "3. Наведената статистика за исходите е пресметана од реални случаи — спомени ја, "
     "но НЕ измислувај свои проценти.\n"
     "4. Не давај правен совет; на крај потсети дека ова е информативна проценка.\n"
     "5. Ако извадоците не се релевантни за прашањето, кажи отворено дека не знаеш."),
    ("user",
     "Прашање на корисникот:\n{question}\n\n"
     "Статистика од најслични случаи: {stats}\n\n"
     "Извадоци од најслични судски одлуки:\n{context}"),
])

SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ти си строг контролор на квалитет. Провери дали СЕКОЕ тврдење во одговорот е "
     "поддржано од извадоците. Ако сè е поддржано, врати го одговорот НЕПРОМЕНЕТ. "
     "Ако нешто не е поддржано, врати поправена верзија без тоа тврдење. "
     "Врати САМО финалниот текст на одговорот, ништо друго."),
    ("user", "Извадоци:\n{context}\n\nОдговор за проверка:\n{answer}"),
])


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2)


def _format_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        parts.append(f"--- Извадок {i}: предмет {m['case_number']}, {m['court']}, "
                     f"{m['date']}, исход: {m['outcome']} ---\n{c.text}")
    return "\n\n".join(parts)


def answer_question(question: str) -> dict:
    question = question.strip()

    # 1. semantic cache
    qvec = embed_query(question)
    cached = cache.lookup(qvec)
    if cached is not None:
        cached["cached"] = True
        return cached

    # 2. hybrid retrieval
    result = retrieve(question)

    # 3. confidence gate — better an honest "не знам" than a hallucination
    if result.max_similarity < MIN_SIMILARITY_FOR_ANSWER or not result.chunks:
        return dict(NO_ANSWER)

    # 4. probability from real outcomes
    prob = estimate(result)
    if prob:
        stats = (f"од {prob.sample_size} најслични случаи, во {prob.percent}% "
                 f"(тежински) {prob.outcome_readable}; распределба: {prob.counts}")
    else:
        stats = "нема доволно споредливи случаи за бројчена проценка"

    # 5. answer
    context = _format_context(result.top_chunks())
    llm = _llm()
    response = llm.invoke(ANSWER_PROMPT.format_messages(
        question=question, stats=stats, context=context))
    log_llm_response(response, label="answer")

    # 6. self-check
    checked = llm.invoke(SELF_CHECK_PROMPT.format_messages(
        context=context, answer=response.content))
    log_llm_response(checked, label="self_check")

    # cited cases for the UI cards: the distinct cases behind the answer
    cases = []
    for chunk in result.top_cases(3):
        m = chunk.metadata
        cases.append({
            "case_number": m["case_number"],
            "court": m["court"],
            "date": m["date"],
            "outcome": READABLE.get(m["outcome"], m["outcome"]),
            "summary": (chunk.text[:220].replace("\n", " ") + "…"),
        })

    payload = {
        "answer": checked.content.strip(),
        "probability": prob.percent if prob else None,
        "cases": cases,
    }

    # 7. cache for next time
    cache.store(qvec, payload)
    return payload
