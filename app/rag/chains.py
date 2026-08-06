"""The main answer chain for Правен асистент (Interface 1).

Pipeline per question:
  0. condensation: if there is chat history, a cheap LLM call rewrites the
     follow-up into a STANDALONE question ("а колку трае постапката?" ->
     "колку трае судска постапка за неисплатени плати?"), so retrieval and
     caching always operate on a complete question
  1. semantic cache lookup            (hit = free, instant answer)
  2. two-stage retrieval              (case summaries -> passages, retriever.py)
     — REUSES the query embedding from step 1, no duplicate OpenAI call
  3. confidence gate                  (low similarity -> honest "Не знам")
  4. probability from case outcomes   (our statistic, probability.py)
  5. ONE LLM call writes the answer (Macedonian, sees the history, cites
     cases) with the self-check folded into the prompt
  6. cache store + return
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, MIN_SIMILARITY_FOR_ANSWER
from app.rag import cache
from app.rag.probability import estimate, READABLE
from app.rag.retriever import retrieve, embed_query


NO_ANSWER = {
    "answer": ("Не знам — во мојата база нема доволно слични случаи за да дадам "
               "сигурна проценка за оваа ситуација. Ве молам обидете се да ја "
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
     "5. Ако извадоците не се релевантни за прашањето, кажи отворено дека не знаеш.\n"
     "6. САМОПРОВЕРКА: пред да го напишеш одговорот, во себе провери го секое тврдење — "
     "дали е директно поддржано од извадоците? Тврдење што не е поддржано, изостави го. "
     "Напиши го САМО конечниот, проверен одговор, без да ја опишуваш проверката."),
    ("user",
     "{history_block}"
     "Прашање на корисникот:\n{question}\n\n"
     "Статистика од најслични случаи: {stats}\n\n"
     "Извадоци од најслични судски одлуки:\n{context}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Дадени се историја на разговор и ново прашање кое може да се потпира на "
     "неа. Преформулирај го новото прашање во САМОСТОЈНО прашање на македонски "
     "што ја содржи целата потребна ситуација од историјата. Ако прашањето е "
     "веќе самостојно, врати го непроменето. Врати САМО прашањето."),
    ("user", "Историја:\n{history}\n\nНово прашање: {question}"),
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


def _condense(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question using the chat history."""
    history_text = "\n".join(
        f"{'Корисник' if h.get('who') == 'user' else 'Асистент'}: {h.get('text', '')[:400]}"
        for h in history[-6:]   # last 3 exchanges are plenty
    )
    response = _llm().invoke(CONDENSE_PROMPT.format_messages(
        history=history_text, question=question))
    return response.content.strip() or question


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    question = question.strip()

    if history and not any(
            h.get("who") == "user" and h.get("text", "").strip() == question
            for h in history):
        question = _condense(question, history)

    # 1. semantic cache (keyed on the standalone question, so follow-ups
    #    that mean the same thing still hit)
    qvec = embed_query(question)
    cached = cache.lookup(qvec)
    if cached is not None:
        cached["cached"] = True
        return cached

    # 2. two-stage retrieval — reusing the vector we just paid for in step 1
    result = retrieve(question, query_vector=qvec)

    # debug: similarity of the top-3 situation-level case matches (stage 1)
    print(f"[слични предмети] прашање: {question!r}")
    for case in result.top_cases(3):
        m = case.metadata
        print(f"  {case.similarity:.3f}  {m['case_number']} ({m['court']}, {m['date']})")

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

    history_block = ""
    if history:
        recent = "\n".join(
            f"{'Корисник' if h.get('who') == 'user' else 'Асистент'}: {h.get('text', '')[:300]}"
            for h in history[-4:]
        )
        history_block = f"Досегашен разговор:\n{recent}\n\n"

    context = _format_context(result.top_chunks())
    response = _llm().invoke(ANSWER_PROMPT.format_messages(
        history_block=history_block, question=question, stats=stats,
        context=context))

    # cited cases for the UI cards — with their human-written-quality
    # summaries ("what the case is about"), far clearer than raw chunk text.
    # case_id lets the card fetch the full decision text («Повеќе»).
    cases = []
    for case in result.top_cases(3):
        m = case.metadata
        cases.append({
            "case_id": m.get("case_id"),
            "case_number": m["case_number"],
            "court": m["court"],
            "date": m["date"],
            "outcome": READABLE.get(m["outcome"], m["outcome"]),
            "summary": m.get("summary", "")[:400],
        })

    payload = {
        "answer": response.content.strip(),
        "probability": prob.percent if prob else None,
        "cases": cases,
    }

    # 6. cache for next time
    cache.store(qvec, payload)
    return payload
