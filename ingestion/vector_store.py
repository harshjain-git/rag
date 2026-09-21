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


class DailyQuotaExhausted(Exception):
    pass

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
        self.successful_requests = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        import re
        
        # Batch requests to avoid sending too many texts at once
        batch_size = 32
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        batch_num = 0

        for i in range(0, len(texts), batch_size):
            batch_num += 1
            batch = texts[i : i + batch_size]
            print(f"      [Gemini] Embedding batch {batch_num}/{total_batches}...")
            
            max_retries = 5
            fallback_delay = 60
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch
                    )
                    for emb in response.embeddings:
                        all_embeddings.append(emb.values)
                    
                    self.successful_requests += 1
                    print(f"      [Gemini] Successful embedding requests: {self.successful_requests}")
                    break  # Success, break retry loop for this batch
                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if the error is a rate limit or quota error
                    if "429" in error_msg or "exhausted" in error_msg or "quota" in error_msg:
                        # Check for daily limits vs RPM
                        if "per day" in error_msg or "per_day" in error_msg or "rpd" in error_msg or "daily" in error_msg:
                            print(f"      [Gemini] Daily quota exhausted detected: {e}")
                            raise DailyQuotaExhausted("Gemini daily embedding quota appears exhausted.")

                        if attempt < max_retries - 1:
                            wait_time = fallback_delay
                            # Try to extract the suggested wait time
                            match = re.search(r"retry in (?:approximately )?(\d+(?:\.\d+)?)\s*(?:seconds?|s)", error_msg)
                            if match:
                                wait_time = float(match.group(1)) + 5  # Add a 5-second buffer
                            
                            print(f"      [Gemini] Rate limit reached. Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            print(f"      [Gemini] Retrying batch...")
                            continue  # Retry the batch
                    
                    # If it's not a rate limit error or we exhausted retries, raise the error
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


def add_chunks_to_vector_store(chunks: List[Dict[str, Any]], vector_store=None, batch_size: int = 32):
    """
    Converts chunk dictionaries into LangChain Document objects and adds them to ChromaDB in batches.
    Skips chunks that are already persisted to make the process safely resumable.
    """
    if vector_store is None:
        vector_store = get_vector_store()

    ids = [c["metadata"]["chunk_id"] for c in chunks]
    total_docs_requested = len(ids)

    # Make it resumable: Check which chunks are already in ChromaDB
    existing_ids = set()
    try:
        existing_data = vector_store.get(ids=ids)
        existing_ids = set(existing_data.get("ids", []))
    except Exception as e:
        print(f"      [Progress] Fatal error while checking existing Chroma IDs: {e}")
        raise

    chunks_to_process = [c for c, chunk_id in zip(chunks, ids) if chunk_id not in existing_ids]
    docs_to_embed = len(chunks_to_process)

    print(f"      [Progress] Total chunks: {total_docs_requested}")
    print(f"      [Progress] Already persisted: {len(existing_ids)}")
    print(f"      [Progress] Remaining: {docs_to_embed}")

    if docs_to_embed == 0:
        print(f"      [Progress] SUCCESS: All {total_docs_requested} chunks are already persisted.")
        return vector_store

    if config.EMBEDDING_PROVIDER == "gemini":
        estimated_batches = (docs_to_embed + 31) // 32
        print(f"      [Progress] Estimated Gemini embedding batches remaining: {estimated_batches}")

    documents = [
        Document(
            page_content=c["text"],
            metadata=c["metadata"]
        )
        for c in chunks_to_process
    ]
    process_ids = [c["metadata"]["chunk_id"] for c in chunks_to_process]

    print(f"      [Progress] Embedding {docs_to_embed} chunks in batches of {batch_size}...")
    
    successful_chunks = len(existing_ids)
    
    for i in range(0, docs_to_embed, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = process_ids[i : i + batch_size]
        
        try:
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            successful_chunks += len(batch_docs)
            print(f"      [Progress] {successful_chunks}/{total_docs_requested} chunks embedded & saved.")
        except DailyQuotaExhausted:
            print(f"      [Progress] Gemini daily embedding quota appears exhausted.")
            print(f"      [Progress] Successfully persisted {successful_chunks}/{total_docs_requested} chunks.")
            print(f"      [Progress] No further Gemini requests will be attempted.")
            break
        except Exception as e:
            print(f"      [Progress] Fatal error during embedding: {e}")
            raise

    # Print final success and requests used
    if successful_chunks == total_docs_requested:
        print(f"      [Progress] SUCCESS: Embedded and persisted {successful_chunks}/{total_docs_requested} chunks")
        
    try:
        embeddings_obj = getattr(vector_store, "_embedding_function", getattr(vector_store, "embeddings", None))
        if hasattr(embeddings_obj, "successful_requests"):
            print(f"      [Progress] Gemini embedding API requests used: {embeddings_obj.successful_requests}")
    except Exception:
        pass

    return vector_store




