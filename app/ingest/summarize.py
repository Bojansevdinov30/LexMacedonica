"""Per-case summaries: "what is this case ABOUT" — the fix for keyword noise.

Problem this solves: chunk search matches words, not situations. A case that
mentions "куче" once in a witness statement would surface for a dog question.
Solution: an LLM writes a 2-3 sentence summary of every case ONCE (what kind
of dispute, what the plaintiff wants, key facts). Retrieval then searches
summaries FIRST to pick candidate cases, and only then finds the best
passages inside those cases (see retriever.py). This is called two-stage /
hierarchical retrieval.

Run:  python -m app.ingest.summarize          (only summarizes what's missing)
"""
from __future__ import annotations

import pickle

import faiss
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import CHAT_MODEL, DATA_DIR, EMBEDDING_MODEL, TXT_DIR
from app.costs import log_llm_response, log_cost
from app.ingest.structure import Case, get_engine

SUMMARY_INDEX_PATH = DATA_DIR / "summaries.index"
SUMMARY_META_PATH = DATA_DIR / "summaries_meta.pkl"

SUMMARY_PROMPT = (
    "Резимирај ја оваа судска одлука во 2-3 реченици на македонски: каков вид "
    "спор е, што бара тужителот, кои се клучните факти и околности. Пиши "
    "содржински (за ситуацијата), не процедурално. Не наведувај имиња, судови "
    "ни датуми — само суштината на ситуацијата."
)


def ensure_summary_column() -> None:
    """SQLite migration: create_all doesn't add columns to existing tables."""
    engine = get_engine()
    with engine.connect() as conn:
        cols = [r[1] for r in conn.execute(sql_text("PRAGMA table_info(cases)"))]
        if "summary" not in cols:
            conn.execute(sql_text("ALTER TABLE cases ADD COLUMN summary TEXT DEFAULT ''"))
            conn.commit()
            print("added cases.summary column")


def generate_missing_summaries() -> None:
    ensure_summary_column()
    engine = get_engine()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.1, max_tokens=200)

    with Session(engine) as session:
        rows = session.execute(sql_text(
            "SELECT case_id FROM cases WHERE summary IS NULL OR summary = ''"
        )).fetchall()
        todo = [r[0] for r in rows]
        print(f"cases missing a summary: {len(todo)}")

        for n, case_id in enumerate(todo, 1):
            txt_path = TXT_DIR / f"{case_id}.txt"
            if not txt_path.exists():
                continue
            # the essence of a decision is at the start (parties, claim,
            # dispositive) — 3500 chars is enough and keeps the call cheap
            excerpt = txt_path.read_text(encoding="utf-8")[:3500]
            response = llm.invoke([("system", SUMMARY_PROMPT), ("user", excerpt)])
            log_llm_response(response, label="case_summary")
            session.execute(
                sql_text("UPDATE cases SET summary = :s WHERE case_id = :cid"),
                {"s": response.content.strip(), "cid": case_id},
            )
            if n % 25 == 0:
                session.commit()
                print(f"  {n}/{len(todo)}")
        session.commit()
    print("summaries done")


def build_summary_index() -> None:
    """Embed every case summary into its own FAISS index (case-level search)."""
    engine = get_engine()
    with Session(engine) as session:
        rows = session.execute(sql_text(
            "SELECT case_id, case_number, court, date, outcome, summary "
            "FROM cases WHERE summary != ''"
        )).fetchall()

    metas = [dict(case_id=r[0], case_number=r[1], court=r[2], date=r[3],
                  outcome=r[4], summary=r[5]) for r in rows]
    print(f"embedding {len(metas)} case summaries")

    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectors = []
    for start in range(0, len(metas), 200):
        batch = [m["summary"] for m in metas[start:start + 200]]
        vectors.extend(embedder.embed_documents(batch))
        log_cost(EMBEDDING_MODEL, sum(len(t) // 2 for t in batch), 0,
                 label="summary_embed")

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(SUMMARY_INDEX_PATH))
    with SUMMARY_META_PATH.open("wb") as f:
        pickle.dump(metas, f)
    print(f"summary index ({index.ntotal} cases) -> {SUMMARY_INDEX_PATH}")


if __name__ == "__main__":
    generate_missing_summaries()
    build_summary_index()
