"""
Gemini Client Initialization & LLM Generation Module for Cadet Readiness Advisor.
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
from google import genai
from google.genai import types

# Shared cached client instance
_LLM_CLIENT = None


def __getattr__(name: str) -> Any:
    if name == "SYSTEM_PROMPT":
        from agent.prompts import CADET_ADVISOR_BASE_INSTRUCTIONS
        return CADET_ADVISOR_BASE_INSTRUCTIONS
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_gemini_client() -> genai.Client:
    """
    Initializes and returns a singleton instance of the Google GenAI Client
    using the API key configured in .env / config.py.
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        
        _LLM_CLIENT = genai.Client(api_key=api_key)
    
    return _LLM_CLIENT




def generate_structured_json(
    contents: str,
    system_instruction: str = "",
    temperature: float = 0.0
) -> Any:
    """
    Executes a structured JSON generation call to Gemini and parses the response safely.
    Centralized helper reused across orchestrator, query rewriter, verification, and question generation.
    """
    import json
    client = get_gemini_client()
    response = client.models.generate_content(
        model=config.LLM_MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )
    raw_text = response.text.strip() if hasattr(response, "text") and response.text else "{}"
    return json.loads(raw_text)

def generate_natural_refusal(query: str) -> str:
    """
    Generates a polite, context-aware refusal using Gemini Flash Lite
    when a question is out of corpus, explaining why it cannot be answered.
    """
    from agent.prompts import build_generation_messages, messages_to_gemini_args
    client = get_gemini_client()
    messages = build_generation_messages(query=query, evidence=[])
    system_instruction, contents = messages_to_gemini_args(messages)

    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        return response.text.strip()
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
    client = get_gemini_client()

    from agent.prompts import CADET_ADVISOR_BASE_INSTRUCTIONS

    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL_NAME,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=CADET_ADVISOR_BASE_INSTRUCTIONS,
                temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        answer_text = response.text.strip()
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

