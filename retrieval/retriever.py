"""
Dense Semantic Retrieval Module for Cadet Readiness Advisor.

This module is now a **thin facade** that delegates all vector store queries
to the ``infrastructure.chroma.chroma_retriever.ChromaRetriever`` adapter.
Existing callers continue to work without any import changes.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

# Load central configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def get_cached_vector_store():
    """
    Returns cached persistent Chroma vector store for backward compatibility.
    Delegates to the ChromaRetriever adapter's lazy-initialised store.
    """
    from infrastructure.chroma.chroma_retriever import get_retriever
    return get_retriever().vector_store


def retrieve_evidence(
    query: str,
    top_k: int = config.INITIAL_TOP_K,
    similarity_threshold: float = config.SIMILARITY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Performs fast dense semantic retrieval against ChromaDB for a given query,
    filtering out results that do not satisfy the similarity threshold.

    Now delegates to ``ChromaRetriever.retrieve()``.
    
    Args:
        query: User question string.
        top_k: Number of relevant evidence chunks to retrieve (default=5).
        similarity_threshold: Maximum allowed L2 distance for relevant chunks.
        
    Returns:
        List of dictionaries containing retrieved text, metadata, and similarity score.
    """
    from infrastructure.chroma.chroma_retriever import get_retriever
    return get_retriever().retrieve(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def evaluate_grounding(query: str, top_k: int = config.INITIAL_TOP_K) -> Dict[str, Any]:
    """
    Phase 8 Grounding & Sufficiency Evaluator.
    Retrieves evidence and determines whether retrieved corpus chunks are sufficient to answer.
    Also captures the raw top distance score before threshold filtering for analysis.

    Now delegates to ``ChromaRetriever.retrieve()`` and ``ChromaRetriever.retrieve_raw()``.
    
    Returns:
        Dict with 'is_grounded', 'evidence', 'top_score', and 'fallback_message'.
    """
    from infrastructure.chroma.chroma_retriever import get_retriever
    retriever = get_retriever()

    # Fetch unfiltered top candidate to get the exact raw L2 distance score
    raw_results = retriever.retrieve_raw(query=query, top_k=1)
    top_score = raw_results[0]["score"] if raw_results else None

    # Fetch evidence filtered by similarity threshold
    evidence = retriever.retrieve(
        query=query,
        top_k=top_k,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
    )
    
    if not evidence:
        return {
            "is_grounded": False,
            "evidence": [],
            "top_score": top_score,
            "fallback_message": config.NOT_IN_CORPUS_MESSAGE,
            "status": "NOT_IN_CORPUS"
        }
    
    return {
        "is_grounded": True,
        "evidence": evidence,
        "top_score": top_score,
        "fallback_message": None,
        "status": "GROUNDED"
    }
