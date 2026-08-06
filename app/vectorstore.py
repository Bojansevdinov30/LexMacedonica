"""Shared ChromaDB access — ONE persistent client, ONE place for the rules."""
from functools import lru_cache

from app.config import CHROMA_DIR
import chromadb
from chromadb.config import Settings
# Chroma writes to data/chroma/chroma.sqlite3
@lru_cache(maxsize=1)
def _client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # anonymized_telemetry=False: no phoning home, and this machine's TLS
    # interception makes the telemetry POSTs fail noisily anyway
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def collection(name: str):
    """Get-or-create a collection in cosine space (distance = 1 - cosine sim)."""
    return _client().get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"})


def sim(distance: float) -> float:
    """Cosine distance (what Chroma returns) -> cosine similarity (what we tune)."""
    return 1.0 - distance


def clean_meta(d: dict) -> dict:
    """Chroma metadata allows only str/int/float/bool — replace None with ''."""
    return {k: ("" if v is None else v) for k, v in d.items()}
