"""Per-case summaries: "what is this case ABOUT"
Run:  python -m app.ingest.summarize          (only summarizes what's missing)
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from app.vectorstore import clean_meta, collection
from app.config import CHAT_MODEL, DATA_DIR, EMBEDDING_MODEL, TXT_DIR

from app.ingest.structure import Case, get_engine

# SUMMARY_INDEX_PATH = DATA_DIR / "summaries.index"
# SUMMARY_META_PATH = DATA_DIR / "summaries_meta.pkl"

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
            # the essence of a decision is at the start, 3500 chars is enough and keeps the call cheap
            excerpt = txt_path.read_text(encoding="utf-8")[:3500]
            response = llm.invoke([("system", SUMMARY_PROMPT), ("user", excerpt)])
            session.execute(
                sql_text("UPDATE cases SET summary = :s WHERE case_id = :cid"),
                {"s": response.content.strip(), "cid": case_id},
            )
            if n % 25 == 0:
                session.commit()
                print(f"  {n}/{len(todo)}")
        session.commit()
    print("summaries done")


def sync_summaries_to_chroma() -> None:
    """Embed only the case summaries Chroma doesn't have yet, then add them."""

    engine = get_engine()
    with Session(engine) as session:
        rows = session.execute(sql_text(
            "SELECT case_id, case_number, court, date, outcome, summary, "
            "judge, legal_area, case_type, case_subtype, foundation_type, foundation "
            "FROM cases WHERE summary != ''"
        )).fetchall()

    col = collection("summaries")
    known = set(col.get(include=[])["ids"])  # ids only — cheap
    metas = [dict(case_id=r[0], case_number=r[1], court=r[2], date=r[3],
                  outcome=r[4], summary=r[5], judge=r[6], legal_area=r[7],
                  case_type=r[8], case_subtype=r[9], foundation_type=r[10],
                  foundation=r[11]) for r in rows]
    new = [m for m in metas if m["case_id"] not in known]

    if new:
        print(f"embedding {len(new)} new case summaries")
        embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        for start in range(0, len(new), 200):
            batch = new[start:start + 200]
            vectors = embedder.embed_documents([m["summary"] for m in batch])
            col.add(
                ids=[m["case_id"] for m in batch],
                embeddings=vectors,
                documents=[m["summary"] for m in batch],  # the searched text
                metadatas=[clean_meta({k: v for k, v in m.items() if k != "summary"})
                           for m in batch],
            )

    # metadata refresh for the EXISTING rows (без ре-embedding, $0): ако некое
    # поле е поправено во SQLite (преку cases_meta.jsonl + build_index),
    # исправката да стигне и до Chroma
    existing = [m for m in metas if m["case_id"] in known]
    for start in range(0, len(existing), 1000):
        batch = existing[start:start + 1000]
        col.update(
            ids=[m["case_id"] for m in batch],
            metadatas=[clean_meta({k: v for k, v in m.items() if k != "summary"})
                       for m in batch],
        )
    print(f"summaries: Chroma holds {col.count()} "
          f"({len(new)} new, metadata refreshed for {len(existing)})")


if __name__ == "__main__":
    generate_missing_summaries()
    sync_summaries_to_chroma()
