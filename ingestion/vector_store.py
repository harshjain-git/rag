import os
import sys
import warnings
import logging

# Suppress noisy deprecation, HF Hub warnings, and progress bars BEFORE imports
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*HuggingFaceBgeEmbeddings.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load central configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def get_embedding_function():
    """
    Initializes and returns the BAAI/bge-base-en-v1.5 embedding model.
    """
    return HuggingFaceBgeEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 8
    }
)


def get_vector_store():
    """
    Initializes and returns the persistent Chroma vector store instance.
    """
    embeddings = get_embedding_function()
    return Chroma(
        collection_name="cadet_readiness",
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DB_DIR)
    )


def add_chunks_to_vector_store(chunks: List[Dict[str, Any]], vector_store=None, batch_size: int = 100):
    """
    Converts chunk dictionaries into LangChain Document objects and adds them to ChromaDB in batches.
    """
    if vector_store is None:
        vector_store = get_vector_store()

    documents = [
        Document(
            page_content=c["text"],
            metadata=c["metadata"]
        )
        for c in chunks
    ]

    ids = [c["metadata"]["chunk_id"] for c in chunks]
    total_docs = len(documents)

    print(f"      [Progress] Embedding {total_docs} chunks in batches of {batch_size}...")
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        end_idx = min(i + batch_size, total_docs)
        print(f"      [Progress] {end_idx}/{total_docs} chunks embedded & saved.")

    return vector_store




