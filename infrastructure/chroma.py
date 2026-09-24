# infrastructure/chroma.py

"""Concrete Retriever adapter backed by ChromaDB.

This is the **only** file that interacts with the ChromaDB vector store for
retrieval. All other modules depend on the ``core.ports.Retriever`` Protocol.

Swapping ChromaDB for Qdrant, Pinecone, FAISS, etc. requires only creating a
new adapter — zero changes to agent code, tools, or pipeline.
"""

from typing import Any, Dict, List, Optional

import config
from ingestion.vector_store import get_vector_store


class ChromaRetriever:
    """Implements ``core.ports.Retriever`` using a LangChain-Chroma vector store.

    Features:
    - Lazy singleton vector store (created on first call, reused thereafter).
    - Similarity search with configurable top-K and distance threshold.
    - Raw (unfiltered) retrieval for grounding score capture.
    """

    def __init__(self, vector_store: Any = None) -> None:
        self._vector_store = vector_store

    # ---- lazy vector store -------------------------------------------

    @property
    def vector_store(self):
        """Return (or create) the cached Chroma vector store instance."""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    # ---- Retriever Protocol implementation ---------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = config.INITIAL_TOP_K,
        similarity_threshold: Optional[float] = config.SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence chunks filtered by similarity threshold."""
        if not query or not query.strip():
            return []

        results = self.vector_store.similarity_search_with_score(
            query.strip(), k=top_k
        )

        evidence_list: List[Dict[str, Any]] = []
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
                "score": dist_score,
            })

        return evidence_list

    def retrieve_raw(
        self,
        query: str,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K results without threshold filtering.

        Used to capture the raw top distance score for grounding evaluation.
        """
        return self.retrieve(query=query, top_k=top_k, similarity_threshold=None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_CHROMA_RETRIEVER: ChromaRetriever | None = None


def get_retriever() -> ChromaRetriever:
    """Return a module-level singleton ``ChromaRetriever`` instance.

    Usage::

        from infrastructure.chroma import get_retriever
        retriever = get_retriever()
        evidence = retriever.retrieve(query="...", top_k=5)
    """
    global _CHROMA_RETRIEVER
    if _CHROMA_RETRIEVER is None:
        _CHROMA_RETRIEVER = ChromaRetriever()
    return _CHROMA_RETRIEVER


def retrieve_evidence(
    query: str,
    top_k: int = config.INITIAL_TOP_K,
    similarity_threshold: Optional[float] = config.SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Convenience helper for evidence retrieval."""
    return get_retriever().retrieve(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def get_cached_vector_store():
    """Convenience helper returning the cached vector store instance."""
    return get_retriever().vector_store


def evaluate_grounding(query: str, top_k: int = config.INITIAL_TOP_K) -> Dict[str, Any]:
    """Retrieves evidence and determines if evidence is sufficient."""
    retriever = get_retriever()
    raw_results = retriever.retrieve_raw(query=query, top_k=1)
    top_score = raw_results[0]["score"] if raw_results else None

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
            "status": "NOT_IN_CORPUS",
        }

    return {
        "is_grounded": True,
        "evidence": evidence,
        "top_score": top_score,
        "fallback_message": None,
        "status": "GROUNDED",
    }
