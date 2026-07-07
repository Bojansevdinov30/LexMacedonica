"""Build the laws index: PDF -> articles (Член N) -> embeddings -> FAISS.

Laws are chunked BY ARTICLE, not by size: a lawyer needs the exact "член 76
од ЗРО" citation, and an article is the natural self-contained unit of a law.
Very long articles get sub-split so embeddings stay focused.

Run:  python -m app.lawyer.build_laws
"""
import pickle
import re

import faiss
import numpy as np
import pymupdf
from langchain_openai import OpenAIEmbeddings

from app.config import DATA_DIR, EMBEDDING_MODEL
from app.costs import log_cost
from app.ingest.chunking import make_splitter

LAWS_DIR = DATA_DIR / "laws"
LAWS_INDEX_PATH = DATA_DIR / "laws.index"
LAWS_META_PATH = DATA_DIR / "laws_meta.pkl"
LAWS_EMB_PATH = DATA_DIR / "laws_embeddings.npz"

# "Член 15", "Член 15-а", "Член  4" (double spaces happen in the PDFs)
ARTICLE_RE = re.compile(r"\n\s*Член\s+(\d+(?:\s*-?\s*[а-ѓ])?)\s*\n")


def split_articles(text: str, law_name: str) -> list[tuple[str, str]]:
    """Returns [(article_label, article_text), ...]."""
    parts = ARTICLE_RE.split(text)
    # parts = [preamble, num1, body1, num2, body2, ...] — skip the preamble
    articles = []
    for i in range(1, len(parts) - 1, 2):
        num = re.sub(r"\s+", "", parts[i])
        body = parts[i + 1].strip()
        if body:
            articles.append((f"Член {num}", body))
    return articles


def main() -> None:
    splitter = make_splitter()
    texts, metadatas, ids = [], [], []

    for pdf in sorted(LAWS_DIR.glob("*.pdf")):
        law_name = pdf.stem
        doc = pymupdf.open(pdf)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        articles = split_articles(text, law_name)
        print(f"{law_name}: {len(articles)} articles")

        for label, body in articles:
            # embed "law + article" header WITH the text: retrieval then knows
            # exactly which law an article belongs to
            full = f"{law_name}, {label}:\n{body}"
            pieces = splitter.split_text(full) if len(full) > 4000 else [full]
            for j, piece in enumerate(pieces):
                ids.append(f"{law_name}|{label}|{j}")
                texts.append(piece)
                metadatas.append({"law": law_name, "article": label})

    print(f"Total chunks to embed: {len(texts)}")

    # embed only what's new (same money-saving pattern as build_index)
    old_vectors, old_ids = np.zeros((0, 1536), dtype="float32"), []
    if LAWS_EMB_PATH.exists():
        stored = np.load(LAWS_EMB_PATH, allow_pickle=True)
        old_vectors, old_ids = stored["vectors"], list(stored["ids"])
    known = set(old_ids)
    new = [(t, i) for t, i in zip(texts, ids) if i not in known]

    if new:
        embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        fresh = []
        for start in range(0, len(new), 200):
            batch = [t for t, _ in new[start:start + 200]]
            fresh.extend(embedder.embed_documents(batch))
            log_cost(EMBEDDING_MODEL, sum(len(t) // 2 for t in batch), 0,
                     label="laws_embed")
            print(f"embedded {min(start + 200, len(new))}/{len(new)}")
        all_vectors = np.vstack([old_vectors, np.array(fresh, dtype="float32")])
        all_ids = old_ids + [i for _, i in new]
        np.savez_compressed(LAWS_EMB_PATH, vectors=all_vectors, ids=np.array(all_ids))
    else:
        all_vectors, all_ids = old_vectors, old_ids
        print("laws embeddings already up to date")

    by_id = {cid: row for cid, row in zip(all_ids, all_vectors)}
    matrix = np.array([by_id[i] for i in ids], dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(LAWS_INDEX_PATH))
    with LAWS_META_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "texts": texts, "metadatas": metadatas}, f)
    print(f"laws index ({index.ntotal} vectors) -> {LAWS_INDEX_PATH}")


if __name__ == "__main__":
    main()
