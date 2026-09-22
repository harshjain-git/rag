"""
LangChain Retrieval Tool for Cadet Readiness Advisor (Step 2).
Wraps the existing ChromaDB retrieval pipeline as a standard LangChain tool.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retriever import retrieve_evidence, evaluate_grounding
from langchain_core.tools import tool, BaseTool


@tool("retrieve_corpus_evidence")
def retrieve_corpus_evidence(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieves relevant evidence chunks from the psychometric, ASVAB, and military readiness document corpus.
    
    Args:
        query: The search question or semantic query text.
        top_k: Optional number of top evidence chunks to retrieve. Defaults to configured initial top-k (5).
        
    Returns:
        List of dictionaries, each containing:
            - 'text': Chunk content string.
            - 'metadata': Metadata dictionary (source, page, chunk_id).
            - 'source': Document filename.
            - 'page': Page number.
            - 'chunk_id': Unique chunk identifier.
            - 'score': Similarity / distance score.
    """
    k = top_k if top_k is not None else config.INITIAL_TOP_K
    return retrieve_evidence(query=query, top_k=k)


def get_retrieval_tool() -> BaseTool:
    """
    Returns the standard LangChain retrieval tool instance for the Cadet Readiness Advisor corpus.
    """
    return retrieve_corpus_evidence
