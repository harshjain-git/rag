"""
Dense Semantic Retrieval Module for Cadet Readiness Advisor.
Queries persistent ChromaDB vector store and retrieves Top-K evidence chunks with metadata.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

# Load central configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from ingestion.vector_store import get_vector_store

# Shared vector store instance for instant queries
_VECTOR_STORE = None


def get_cached_vector_store():
    """
    Returns cached persistent Chroma vector store to avoid re-initializing on every query.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = get_vector_store()
    return _VECTOR_STORE


def retrieve_evidence(
    query: str,
    top_k: int = config.INITIAL_TOP_K,
    similarity_threshold: float = config.SIMILARITY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Performs fast dense semantic retrieval against ChromaDB for a given query,
    filtering out results that do not satisfy the similarity threshold.
    
    Args:
        query: User question string.
        top_k: Number of relevant evidence chunks to retrieve (default=5).
        similarity_threshold: Maximum allowed L2 distance for relevant chunks.
        
    Returns:
        List of dictionaries containing retrieved text, metadata, and similarity score.
    """
    if not query or not query.strip():
        return []

    vector_store = get_cached_vector_store()

    # Similarity search returning (Document, score) tuples
    results = vector_store.similarity_search_with_score(query.strip(), k=top_k)

    evidence_list = []
    for doc, score in results:
        dist_score = float(score)
        if similarity_threshold is not None and dist_score > similarity_threshold:
            continue
            
        meta = doc.metadata.copy()
        evidence_list.append({
            "text": doc.page_content,
            "metadata": meta,
            "source": meta.get("source", "unknown.pdf"),
            "page": meta.get("page", 1),
            "chunk_id": meta.get("chunk_id", ""),
            "score": dist_score
        })

    return evidence_list


def evaluate_grounding(query: str, top_k: int = config.INITIAL_TOP_K) -> Dict[str, Any]:
    """
    Phase 8 Grounding & Sufficiency Evaluator.
    Retrieves evidence and determines whether retrieved corpus chunks are sufficient to answer.
    Also captures the raw top distance score before threshold filtering for analysis.
    
    Returns:
        Dict with 'is_grounded', 'evidence', 'top_score', and 'fallback_message'.
    """
    # Fetch unfiltered top candidate to get the exact raw L2 distance score
    raw_results = retrieve_evidence(query, top_k=1, similarity_threshold=None)
    top_score = raw_results[0]["score"] if raw_results else None

    # Fetch evidence filtered by similarity threshold
    evidence = retrieve_evidence(query, top_k=top_k, similarity_threshold=config.SIMILARITY_THRESHOLD)
    
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

