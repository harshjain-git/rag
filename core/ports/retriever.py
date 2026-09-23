# core/ports/retriever.py

"""Retriever port — a provider-agnostic interface for vector store retrieval.

Any concrete retriever adapter (ChromaDB, Qdrant, Pinecone, FAISS, etc.) must
implement this Protocol. The agent pipeline depends *only* on this contract.
"""

from typing import Protocol, List, Dict, Any, Optional


class Retriever(Protocol):
    """Minimal abstraction over any vector store retrieval backend.

    Implementations must provide:
    - ``retrieve``: returns filtered evidence chunks above a similarity threshold.
    - ``retrieve_raw``: returns unfiltered top-K results (used for grounding scoring).
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence chunks that pass the similarity threshold.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            similarity_threshold: Maximum distance score to accept (lower = more
                similar). ``None`` means no filtering.

        Returns:
            List of evidence dicts with keys: ``text``, ``source``, ``page``,
            ``chunk_id``, ``score``, ``metadata``.
        """
        ...

    def retrieve_raw(
        self,
        query: str,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K results without any threshold filtering.

        Useful for capturing the raw top similarity score before filtering.

        Args:
            query: The search query text.
            top_k: Number of results.

        Returns:
            List of evidence dicts (same shape as ``retrieve``).
        """
        ...
