"""Build everything the RAG needs, from the scraped data.

Steps (each idempotent, so re-running is always safe):
  1. PDFs -> data/txt/*.txt            (PyMuPDF, free)
  2. txt + cases_meta.jsonl -> SQLite  (regex outcome extraction, free)
  3. txt -> chunks -> embeddings       (OpenAI, costs ~cents — but cached in
     embeddings.npz so re-runs never re-pay) -> ChromaDB (data/chroma/)

Run:  python -m app.ingest.build_index            # everything
      python -m app.ingest.build_index --no-embed # only the free steps

(Не пуштај го ова додека uvicorn работи — Chroma му треба ексклузивен пристап
до data/chroma/chroma.sqlite3 на Windows.)
"""
import argparse
import json
import re

from app.config import (
    DATA_DIR, EMBEDDING_MODEL, EMBEDDINGS_PATH, RAW_PDF_DIR, TXT_DIR,
)
from app.ingest.chunking import make_splitter
from app.ingest.pdf_extract import extract_all
from app.ingest.structure import Case, extract_outcome, get_engine

# останува само за rollback на старата верзија (other/) — не се пишува веќе
# BM25_PATH = DATA_DIR / "bm25.pkl"


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 — lowercase words, Cyrillic-aware."""
    return re.findall(r"\w+", text.lower())


def load_metadata() -> dict[str, dict]:
    meta = {}
    path = DATA_DIR / "cases_meta.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    meta[row["case_id"]] = row
    return meta


def build_sqlite() -> dict[str, dict]:
    """txt files + scraper metadata -> cases table.

    Returns plain dicts (case_id -> fields), NOT ORM objects: ORM instances
    become unusable ("detached") once their session closes.
    """
    from sqlalchemy.orm import Session

    meta = load_metadata()
    engine = get_engine()
    cases: dict[str, dict] = {}
    counts: dict[str, int] = {}

    with Session(engine) as session:
        for txt in sorted(TXT_DIR.glob("*.txt")):
            case_id = txt.stem
            text = txt.read_text(encoding="utf-8")
            outcome, sentence = extract_outcome(text)
            m = meta.get(case_id, {})
            fields = {
                "case_id": case_id,
                "case_number": m.get("case_number", ""),
                "court": m.get("court", ""),
                "date": m.get("date", ""),
                "preview": m.get("preview", ""),
                "outcome": outcome,
                "outcome_sentence": sentence,
                # секцијата «Повеќе податоци» од порталот
                "judge": m.get("judge", ""),
                "legal_area": m.get("legal_area", ""),
                "case_type": m.get("case_type", ""),
                "case_subtype": m.get("case_subtype", ""),
                "foundation_type": m.get("foundation_type", ""),
                "foundation": m.get("foundation", ""),
            }
            session.merge(Case(**fields))
            cases[case_id] = fields
            counts[outcome] = counts.get(outcome, 0) + 1
        session.commit()

    print(f"SQLite: {len(cases)} cases. Outcomes: {counts}")
    return cases


def build_chunks(cases: dict[str, dict]) -> tuple[list[str], list[dict], list[str]]:
    splitter = make_splitter()
    texts, metadatas, ids = [], [], []
    for txt in sorted(TXT_DIR.glob("*.txt")):
        case = cases.get(txt.stem)
        if case is None:
            continue
        for i, chunk in enumerate(splitter.split_text(txt.read_text(encoding="utf-8"))):
            texts.append(chunk)
            ids.append(f"{case['case_id']}_{i}")
            metadatas.append({k: case[k] for k in
                              ("case_id", "case_number", "court", "date", "outcome",
                               "judge", "legal_area", "case_type", "case_subtype",
                               "foundation_type", "foundation")})
    print(f"Chunks: {len(texts)} from {len(cases)} cases")
    return texts, metadatas, ids


def build_vectors(texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    """Embed NEW chunks (OpenAI) and upsert everything into Chroma.

    The expensive part — the embeddings — is cached in embeddings.npz
    (the npz is the ledger of what we already paid for, and the
    backup that lets data/chroma/ be deleted and rebuilt at any time for $0).
    """
    import numpy as np
    from langchain_openai import OpenAIEmbeddings

    from app.vectorstore import clean_meta, collection

    # load what we already embedded
    old_vectors, old_ids = np.zeros((0, 1536), dtype="float32"), []
    if EMBEDDINGS_PATH.exists():
        stored = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        old_vectors, old_ids = stored["vectors"], list(stored["ids"])

    known = set(old_ids)
    new = [(t, m, i) for t, m, i in zip(texts, metadatas, ids) if i not in known]

    if new:
        embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        BATCH = 200
        fresh = []
        for start in range(0, len(new), BATCH):
            batch = new[start:start + BATCH]
            b_texts = [t for t, _, _ in batch]
            fresh.extend(embedder.embed_documents(b_texts))
            print(f"Embedded {min(start + BATCH, len(new))}/{len(new)} new chunks")
        all_vectors = np.vstack([old_vectors, np.array(fresh, dtype="float32")])
        all_ids = old_ids + [i for _, _, i in new]
        np.savez_compressed(EMBEDDINGS_PATH, vectors=all_vectors, ids=np.array(all_ids))
    else:
        all_vectors, all_ids = old_vectors, old_ids
        print("Embeddings: already up to date")

    # keep only vectors whose chunk still exists, ordered like `ids`
    by_id = {cid: row for cid, row in zip(all_ids, all_vectors)}

    col = collection("chunks")
    BATCH = 1000  # Chroma's max add/upsert batch is ~5.4K
    for s in range(0, len(ids), BATCH):
        col.upsert(
            ids=ids[s:s + BATCH],
            embeddings=[by_id[i].tolist() for i in ids[s:s + BATCH]],
            documents=texts[s:s + BATCH],
            metadatas=[clean_meta(m) for m in metadatas[s:s + BATCH]],
        )
    print(f"Chroma 'chunks': {col.count()} vectors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true",
                    help="skip the OpenAI embedding step (costs money)")
    args = ap.parse_args()

    print(f"Extracting PDFs from {RAW_PDF_DIR}...")
    new = extract_all()
    print(f"  {new} new txt files")

    cases = build_sqlite()
    texts, metadatas, ids = build_chunks(cases)

    if args.no_embed:
        print("Skipping embeddings (--no-embed). Run again without it when ready.")
        return
    build_vectors(texts, metadatas, ids)


if __name__ == "__main__":
    main()
