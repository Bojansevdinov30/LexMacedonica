"""Build every index the RAG needs, from the scraped data.

Steps (each idempotent, so re-running is always safe):
  1. PDFs -> data/txt/*.txt            (PyMuPDF, free)
  2. txt + cases_meta.jsonl -> SQLite  (regex outcome extraction, free)
  3. txt -> chunks -> Chroma vectors   (OpenAI embeddings, costs ~cents)
     and a BM25 keyword index          (free)

Run:  python -m app.ingest.build_index            # everything
      python -m app.ingest.build_index --no-embed # only the free steps
"""
import argparse
import json
import pickle
import re

from app.config import (
    DATA_DIR, EMBEDDING_MODEL, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
    RAW_PDF_DIR, TXT_DIR, VECTOR_META_PATH,
)
from app.ingest.chunking import make_splitter
from app.ingest.pdf_extract import extract_all
from app.ingest.structure import Case, extract_outcome, get_engine

BM25_PATH = DATA_DIR / "bm25.pkl"


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
                              ("case_id", "case_number", "court", "date", "outcome")})
    print(f"Chunks: {len(texts)} from {len(cases)} cases")
    return texts, metadatas, ids


def build_bm25(texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metadatas, "ids": ids}, f)
    print(f"BM25 index -> {BM25_PATH}")


def build_faiss(texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    """Embed new chunks (OpenAI) and (re)build the FAISS HNSW index.

    The expensive part — the embeddings — is cached in embeddings.npz, so
    re-running never re-pays OpenAI for chunks it already embedded. The HNSW
    index itself is cheap to rebuild from the stored vectors.
    """
    import faiss
    import numpy as np
    from langchain_openai import OpenAIEmbeddings

    from app.costs import log_cost

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
            # rough token estimate (~2 chars/token for Cyrillic) for the budget log
            log_cost(EMBEDDING_MODEL, sum(len(t) // 2 for t in b_texts), 0,
                     label="index_embed")
            print(f"Embedded {min(start + BATCH, len(new))}/{len(new)} new chunks")
        all_vectors = np.vstack([old_vectors, np.array(fresh, dtype="float32")])
        all_ids = old_ids + [i for _, _, i in new]
        np.savez_compressed(EMBEDDINGS_PATH, vectors=all_vectors, ids=np.array(all_ids))
    else:
        all_vectors, all_ids = old_vectors, old_ids
        print("Embeddings: already up to date")

    # keep only vectors whose chunk still exists, ordered like `ids`
    by_id = {cid: row for cid, row in zip(all_ids, all_vectors)}
    matrix = np.array([by_id[i] for i in ids], dtype="float32")

    # normalize so inner product == cosine similarity
    faiss.normalize_L2(matrix)
    # Exact (flat) index, not HNSW: FAISS's HNSW construction crashes on this
    # machine (OpenMP access violation, Py3.14/Windows), and at ~6K vectors
    # exact search is instant with perfect recall anyway. HNSW only becomes
    # worth its build cost around millions of vectors.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with VECTOR_META_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "texts": texts, "metadatas": metadatas}, f)
    print(f"FAISS index ({index.ntotal} vectors) -> {FAISS_INDEX_PATH}")


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
    build_bm25(texts, metadatas, ids)

    if args.no_embed:
        print("Skipping embeddings (--no-embed). Run again without it when ready.")
        return
    build_faiss(texts, metadatas, ids)


if __name__ == "__main__":
    main()
