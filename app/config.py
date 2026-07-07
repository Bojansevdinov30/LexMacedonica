"""Central configuration for LexMacedonica.

Everything tunable lives here so we never hunt for magic numbers in the code.
Secrets (the OpenAI key) live in .env, loaded via python-dotenv.
"""
from pathlib import Path

import truststore
from dotenv import load_dotenv

# This machine's HTTPS is TLS-intercepted (antivirus), so Python's bundled
# certificates fail. truststore makes Python use the Windows cert store —
# without this line every OpenAI/tiktoken call dies with CERTIFICATE_VERIFY_FAILED.
truststore.inject_into_ssl()

load_dotenv()  # makes OPENAI_API_KEY from .env visible to os.environ / OpenAI SDK

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
TXT_DIR = DATA_DIR / "txt"
RAW_HTML_DIR = DATA_DIR / "raw_html"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"     # HNSW index (search structure)
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"   # raw vectors (so re-runs don't re-pay OpenAI)
VECTOR_META_PATH = DATA_DIR / "vector_meta.pkl" # ids/texts/metadatas parallel to vectors
CACHE_PATH = DATA_DIR / "semantic_cache.pkl"
SQLITE_PATH = DATA_DIR / "lex.db"

# --- Models (budget rule: cheapest tier only, see CLAUDE.md) ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# --- RAG tuning ---
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_RATIO = 0.15
TOP_CHUNKS_FOR_LLM = 3        # keep context small = less noise + cheaper
TOP_CASES_FOR_PROBABILITY = 10
MIN_SIMILARITY_FOR_ANSWER = 0.30   # below this → honest "Не знам"
SEMANTIC_CACHE_THRESHOLD = 0.95

# --- Budget guard ---
BUDGET_WARN_USD = 15.0
