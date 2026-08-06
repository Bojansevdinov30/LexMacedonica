"""Central configuration for LexMacedonica.
Everything tunable lives here so we never hunt for magic numbers in the code.
Secrets (the OpenAI key) live in .env, loaded via python-dotenv."""
from pathlib import Path

import truststore
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# This machine's HTTPS is TLS-intercepted (antivirus), so Python's bundled
# certificates fail. truststore makes Python use the Windows cert store —
# without this line every OpenAI/tiktoken call dies with CERTIFICATE_VERIFY_FAILED.
truststore.inject_into_ssl()

load_dotenv()


class Settings(BaseSettings):
    """Typed, env-overridable tunables (paths stay plain constants below)."""

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

    # --- RAG tuning ---
    CHUNK_SIZE_TOKENS: int = 800
    CHUNK_OVERLAP_RATIO: float = 0.15
    TOP_CHUNKS_FOR_LLM: int = 3  # keep context small = less noise + cheaper
    TOP_CASES_FOR_PROBABILITY: int = 10
    MIN_SIMILARITY_FOR_ANSWER: float = 0.43
    SEMANTIC_CACHE_THRESHOLD: float = 0.95

    model_config = SettingsConfigDict(env_prefix="LEX_", extra="ignore")


settings = Settings()

# --- Paths (derived from the repo layout, not env-tunable) ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
TXT_DIR = DATA_DIR / "txt"
RAW_HTML_DIR = DATA_DIR / "raw_html"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"  # raw vectors (so re-runs don't re-pay OpenAI)
SQLITE_PATH = DATA_DIR / "lex.db"
CHROMA_DIR = DATA_DIR / "chroma"

# --- Backward-compatible re-exports: the rest of the code keeps importing
#     `from app.config import CHAT_MODEL` exactly as before the Settings class.
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
CHAT_MODEL = settings.CHAT_MODEL
CHUNK_SIZE_TOKENS = settings.CHUNK_SIZE_TOKENS
CHUNK_OVERLAP_RATIO = settings.CHUNK_OVERLAP_RATIO
TOP_CHUNKS_FOR_LLM = settings.TOP_CHUNKS_FOR_LLM
TOP_CASES_FOR_PROBABILITY = settings.TOP_CASES_FOR_PROBABILITY
MIN_SIMILARITY_FOR_ANSWER = settings.MIN_SIMILARITY_FOR_ANSWER
SEMANTIC_CACHE_THRESHOLD = settings.SEMANTIC_CACHE_THRESHOLD
