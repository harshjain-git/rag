"""
Citation Generation Engine Module for Cadet Readiness Advisor.
Extracts, deduplicates, and formats source citations from retrieved evidence chunks.
"""

from typing import List, Dict, Any


def extract_citations(evidence_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts unique source document names and page numbers from retrieved evidence chunks.
    
    Args:
        evidence_chunks: List of retrieved evidence chunk dictionaries containing 'source' and 'page'.
        
    Returns:
        Sorted list of unique citation dictionaries: [{'source': 'asvab1.pdf', 'page': 4, 'citation': 'asvab1.pdf — Page 4'}, ...]
    """
    seen = set()
    citations = []

    for chunk in evidence_chunks:
        source = chunk.get("source", "unknown.pdf")
        page = chunk.get("page", 1)
        pair = (source, page)
        
        if pair not in seen:
            seen.add(pair)
            citations.append({
                "source": source,
                "page": page,
                "citation": f"{source} — Page {page}"
            })

    # Sort citations alphabetically by source filename, then numerically by page number
    citations.sort(key=lambda x: (x["source"], x["page"]))
    return citations


def format_citations_block(citations: List[Dict[str, Any]]) -> str:
    """
    Formats list of citation dictionaries into a clean markdown block.
    
    Args:
        citations: List of citation dictionaries from extract_citations().
        
    Returns:
        Markdown formatted string listing all source citations.
    """
    if not citations:
        return ""

    lines = ["\n\n---\n**Sources:**"]
    for c in citations:
        lines.append(f"• {c['citation']}")

    return "\n".join(lines)


def attach_citations(answer: str, evidence_chunks: List[Dict[str, Any]], is_grounded: bool = True) -> str:
    """
    Attaches formatted citations to the generated answer if the query is grounded.
    
    Args:
        answer: Generated LLM answer text.
        evidence_chunks: Retrieved evidence chunks.
        is_grounded: Grounding status boolean.
        
    Returns:
        Complete answer string with citations appended.
    """
    if not is_grounded or not evidence_chunks:
        return answer

    citations = extract_citations(evidence_chunks)
    citations_block = format_citations_block(citations)
    
    return f"{answer.strip()}{citations_block}"
