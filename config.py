import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Embedding Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking Parameters
TARGET_CHUNK_SIZE = 900  # Tokens / characters approx target
TARGET_CHUNK_OVERLAP = 120  # Overlap

# Retrieval & Grounding Settings
INITIAL_TOP_K = 5
# Strict distance threshold for BGE L2 distance: lower distance = higher similarity.
# Chunks with distance > 0.85 are considered irrelevant / out of corpus context.
SIMILARITY_THRESHOLD = 0.85

# Generator LLM Settings (Gemini Flash Lite)
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Grounding / Strict Corpus Response
NOT_IN_CORPUS_MESSAGE = (
    "Not in corpus — the provided documents do not contain enough information to answer this question."
)
