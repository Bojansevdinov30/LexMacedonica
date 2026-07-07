"""Lawyer assistant (Interface 3): law-first retrieval + visible reasoning.

Differences vs the citizen chat:
- retrieval is LAW-FIRST (exact article citations) with cases as support;
  a "cases" mode exists to answer from case law only
- the model's REASONING is returned and shown to the user (lawyers want to
  see how a conclusion was reached, citizens just want the conclusion)
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache

import faiss
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL
from app.costs import log_llm_response
from app.lawyer.build_laws import LAWS_INDEX_PATH, LAWS_META_PATH
from app.rag.retriever import embed_query, retrieve

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


@lru_cache(maxsize=1)
def _laws_index():
    index = faiss.read_index(str(LAWS_INDEX_PATH))
    with LAWS_META_PATH.open("rb") as f:
        meta = pickle.load(f)
    return index, meta


def laws_available() -> bool:
    return LAWS_INDEX_PATH.exists()


def search_laws(question: str, k: int = 4) -> list[dict]:
    index, meta = _laws_index()
    q = np.array([embed_query(question)], dtype="float32")
    faiss.normalize_L2(q)
    sims, positions = index.search(q, k)
    out = []
    for pos, sim in zip(positions[0], sims[0]):
        if pos < 0:
            continue
        m = meta["metadatas"][pos]
        out.append({"law": m["law"], "article": m["article"],
                    "text": meta["texts"][pos], "similarity": float(sim)})
    return out


def lawyer_answer(question: str, mode: str = "laws") -> dict:
    """mode: "laws" (закони + пракса) or "cases" (само судска пракса)."""
    sources_text, sources_list = [], []

    if mode == "laws" and laws_available():
        for a in search_laws(question, k=4):
            sources_text.append(f"--- {a['law']}, {a['article']} ---\n{a['text']}")
            sources_list.append({"type": "закон", "ref": f"{a['law']}, {a['article']}"})
        case_k = 2   # cases as secondary support
    else:
        case_k = 4   # cases-only mode

    case_result = retrieve(question)
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
    log_llm_response(response, label="lawyer")

    try:
        parsed = json.loads(response.content)
        reasoning = parsed.get("reasoning", "")
        answer = parsed.get("answer", response.content)
    except json.JSONDecodeError:
        reasoning, answer = "", response.content

    return {"reasoning": reasoning, "answer": answer, "sources": sources_list,
            "mode": mode}
