import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Embedding Settings
EMBEDDING_PROVIDER = "gemini"  # "gemini" or "bge"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Local BGE model
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"  # Render fallback model

# Chunking Parameters
TARGET_CHUNK_SIZE = 900  # Tokens / characters approx target
TARGET_CHUNK_OVERLAP = 120  # Overlap

# Retrieval & Grounding Settings
INITIAL_TOP_K = 5
# Strict distance threshold for BGE L2 distance: lower distance = higher similarity.
# Chunks with distance > 0.85 are considered irrelevant / out of corpus context.
BGE_SIMILARITY_THRESHOLD = 0.85
GEMINI_SIMILARITY_THRESHOLD = None

SIMILARITY_THRESHOLD = (
    GEMINI_SIMILARITY_THRESHOLD if EMBEDDING_PROVIDER == "gemini" else BGE_SIMILARITY_THRESHOLD
)

# Generator LLM Settings (Gemini Flash Lite)
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Grounding / Strict Corpus Response
NOT_IN_CORPUS_MESSAGE = (
    "Not in corpus — the provided documents do not contain enough information to answer this question."
)
