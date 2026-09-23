"""
Gemini Client Initialization & LLM Generation Module for Cadet Readiness Advisor.

This module is now a **thin facade** that delegates all LLM calls to the
``infrastructure.gemini.gemini_llm.GeminiLLM`` adapter. Existing callers
continue to work without any import changes.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import warnings
import logging
from typing import Any, Dict, List, Optional

# Suppress noisy deprecation & genai warnings
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

import config


def __getattr__(name: str) -> Any:
    if name == "SYSTEM_PROMPT":
        from application.prompts import CADET_ADVISOR_BASE_INSTRUCTIONS
        return CADET_ADVISOR_BASE_INSTRUCTIONS
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_gemini_client():
    """
    Returns the underlying google.genai Client for backward compatibility.
    Delegates to the GeminiLLM adapter's lazy-initialised client.
    """
    from infrastructure.gemini.gemini_llm import get_llm
    return get_llm().client


def generate_structured_json(
    contents: str,
    system_instruction: str = "",
    temperature: float = 0.0
) -> Any:
    """
    Executes a structured JSON generation call to Gemini and parses the response safely.
    Centralized helper reused across orchestrator, query rewriter, verification, and question generation.

    Now delegates to ``GeminiLLM.generate_json()``.
    """
    from infrastructure.gemini.gemini_llm import get_llm
    return get_llm().generate_json(
        contents=contents,
        system_instruction=system_instruction,
        temperature=temperature,
    )

def generate_natural_refusal(query: str) -> str:
    """
    Generates a polite, context-aware refusal using Gemini
    when a question is out of corpus, explaining why it cannot be answered.

    Now delegates to ``GeminiLLM.generate_text()``.
    """
    from application.prompts import build_generation_messages, messages_to_gemini_args

    messages = build_generation_messages(query=query, evidence=[])
    system_instruction, contents = messages_to_gemini_args(messages)

    try:
        from infrastructure.gemini.gemini_llm import get_llm
        return get_llm().generate_text(
            contents=contents,
            system_instruction=system_instruction,
            temperature=0.0,
        )
    except Exception:
        # Fallback to standard message if API call fails
        return config.NOT_IN_CORPUS_MESSAGE


def format_context_prompt(query: str, evidence_chunks: list) -> str:
    """
    Formats retrieved evidence chunks into a clean context block for Gemini.
    """
    context_str = "RETRIEVED DOCUMENT EXCERPTS:\n\n"
    for idx, chunk in enumerate(evidence_chunks, 1):
        source = chunk.get("source", "unknown.pdf")
        page = chunk.get("page", 1)
        text = chunk.get("text", "").strip()
        context_str += f"--- Excerpt #{idx} (Document: {source} | Page: {page}) ---\n{text}\n\n"
    
    context_str += f"USER QUESTION: {query}\n\nANSWER:"
    return context_str


def generate_answer(query: str, grounding_result: dict = None) -> dict:
    """
    Generates a grounded response using Gemini Flash Lite LLM or returns a natural refusal if ungrounded.
    
    Args:
        query: User question string.
        grounding_result: Optional pre-evaluated grounding dictionary from retrieval.evaluate_grounding.
        
    Returns:
        Dict with 'answer', 'is_grounded', 'evidence', and 'status'.
    """
    # Import evaluate_grounding lazily to avoid circular imports
    from retrieval.retriever import evaluate_grounding

    if grounding_result is None:
        grounding_result = evaluate_grounding(query)

    top_score = grounding_result.get("top_score")

    # 1. Phase 8 Grounding Check: If not grounded, generate polite context-aware refusal
    if not grounding_result.get("is_grounded", False):
        refusal_text = generate_natural_refusal(query)
        return {
            "answer": refusal_text,
            "is_grounded": False,
            "top_score": top_score,
            "evidence": [],
            "status": "NOT_IN_CORPUS"
        }

    evidence = grounding_result.get("evidence", [])
    prompt_text = format_context_prompt(query, evidence)

    from application.prompts import CADET_ADVISOR_BASE_INSTRUCTIONS
    from infrastructure.gemini.gemini_llm import get_llm

    try:
        answer_text = get_llm().generate_text(
            contents=prompt_text,
            system_instruction=CADET_ADVISOR_BASE_INSTRUCTIONS,
            temperature=0.0,
        )
    except Exception as e:
        answer_text = f"Error generating LLM response: {str(e)}"

    from citation.citation_engine import extract_citations, format_citations_block

    citations = extract_citations(evidence)
    citations_text = format_citations_block(citations)
    full_answer = f"{answer_text}{citations_text}" if answer_text != config.NOT_IN_CORPUS_MESSAGE else answer_text

    return {
        "answer": full_answer,
        "raw_answer": answer_text,
        "citations": citations,
        "is_grounded": True,
        "top_score": top_score,
        "evidence": evidence,
        "status": "GROUNDED"
    }
