"""Lawyer assistant (Interface 3): law-first retrieval + visible reasoning.

Differences vs the citizen chat:
- retrieval is LAW-FIRST (exact article citations) with cases as support;
  a "cases" mode exists to answer from case law only
- the model's REASONING is returned and shown to the user (lawyers want to
  see how a conclusion was reached, citizens just want the conclusion)
"""
from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL

from app.rag.retriever import embed_query, retrieve
from app.vectorstore import collection, sim

LAWYER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ти си стручен асистент за македонски адвокати. Одговараш на македонски, "
     "прецизно и со правна терминологија.\n"
     "Правила:\n"
     "1. Користи ИСКЛУЧИВО дадени извори (членови од закони и/или судска пракса).\n"
     "2. Секое тврдење поткрепи го со точен цитат: закон + член, или број на предмет.\n"
     "3. Ако изворите не одговараат на прашањето, кажи го тоа отворено.\n"
     "Врати JSON со точно две полиња:\n"
     '  "reasoning": твоето размислување чекор по чекор (кои извори се релевантни, '
     "зошто, како се поврзуваат)\n"
     '  "answer": финалниот одговор за адвокатот'),
    ("user", "Прашање:\n{question}\n\nИзвори:\n{context}"),
])


def laws_available() -> bool:
    return collection("laws").count() > 0


def search_laws(question: str, k: int = 4,
                query_vector: list[float] | None = None) -> list[dict]:
    # reuse the caller's embedding when it has one (explicit None check —
    # an empty list must not silently trigger a re-embed)
    if query_vector is None:
        query_vector = embed_query(question)
    r = collection("laws").query(query_embeddings=[query_vector], n_results=k,
                                 include=["documents", "metadatas", "distances"])
    out = []
    for m, text, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0]):
        out.append({"law": m["law"], "article": m["article"],
                    "text": text, "similarity": sim(dist)})
    return out


def lawyer_answer(question: str, mode: str = "laws") -> dict:
    """mode: "laws" (закони + пракса) or "cases" (само судска пракса)."""
    # ONE embedding, shared by the laws search and the case retrieval below —
    # the question text is identical, so re-embedding it was a wasted call
    qvec = embed_query(question)
    sources_text, sources_list = [], []

    if mode == "laws" and laws_available():
        for a in search_laws(question, k=4, query_vector=qvec):
            sources_text.append(f"--- {a['law']}, {a['article']} ---\n{a['text']}")
            sources_list.append({"type": "закон", "ref": f"{a['law']}, {a['article']}"})
        case_k = 2   # cases as secondary support
    else:
        case_k = 4   # cases-only mode

    case_result = retrieve(question, query_vector=qvec)
    for chunk in case_result.top_chunks(case_k):
        m = chunk.metadata
        sources_text.append(f"--- Судска пракса: {m['case_number']}, {m['court']}, "
                            f"{m['date']}, исход: {m['outcome']} ---\n{chunk.text}")
        sources_list.append({"type": "пракса",
                             "ref": f"{m['case_number']} ({m['court']}, {m['date']})"})

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.1,
                     model_kwargs={"response_format": {"type": "json_object"}})
    response = llm.invoke(LAWYER_PROMPT.format_messages(
        question=question, context="\n\n".join(sources_text)))

    try:
        parsed = json.loads(response.content)
        reasoning = parsed.get("reasoning", "")
        answer = parsed.get("answer", response.content)
    except json.JSONDecodeError:
        reasoning, answer = "", response.content

    return {"reasoning": reasoning, "answer": answer, "sources": sources_list,
            "mode": mode}
