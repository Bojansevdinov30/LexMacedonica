"""Chunking: split long decisions into overlapping pieces for retrieval."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_RATIO


def make_splitter() -> RecursiveCharacterTextSplitter:
    # measured in TOKENS (what the embedding model actually sees), not chars
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=int(CHUNK_SIZE_TOKENS * CHUNK_OVERLAP_RATIO),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
