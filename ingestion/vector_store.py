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

# Load central configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class GeminiEmbeddings:
    """
    Custom LangChain-compatible embeddings class using Google GenAI SDK.
    Implemented to avoid loading heavy PyTorch/Sentence-Transformers libraries on Render.
    """
    def __init__(self):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.model = config.GEMINI_EMBEDDING_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch requests to avoid sending too many texts at once
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch
                )
                for emb in response.embeddings:
                    all_embeddings.append(emb.values)
            except Exception as e:
                print(f"Error calling Gemini Embedding API: {e}")
                raise
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error calling Gemini Embedding API: {e}")
            raise


def get_embedding_function():
    """
    Initializes and returns the embedding model based on config.
    """
    if config.EMBEDDING_PROVIDER == "gemini":
        return GeminiEmbeddings()
    elif config.EMBEDDING_PROVIDER == "bge":
        # Conditionally import so we don't load PyTorch on Render
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        return HuggingFaceBgeEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 8
            }
        )
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {config.EMBEDDING_PROVIDER}")


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




