"""
LangChain Query Rewriter Tool for Cadet Readiness Advisor.
Encapsulates single-call structured query resolution and reformulation directly as a standard LangChain BaseTool.
Evaluates whether a user query should be kept unchanged (KEEP) or reformulated (REWRITE) for document retrieval.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool, BaseTool

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from generation.generator import get_gemini_client
from google.genai import types
from agent.prompts import (
    build_query_resolution_messages,
    messages_to_gemini_args,
)


@tool("query_rewriter")
def query_rewriter(query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Analyzes a user query in the context of recent conversation history and determines
    whether to KEEP the query as-is or REWRITE it into a standalone retrieval query.
    
    Args:
        query: The incoming user query text.
        history: Optional list of recent dialogue messages with 'role' and 'content'.
        
    Returns:
        A dictionary with:
            - 'action': 'KEEP' or 'REWRITE'
            - 'query': The final retrieval query text
            - 'reason': Brief explanation of the decision
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return {
            "action": "KEEP",
            "query": "",
            "reason": "Empty query"
        }

    messages = build_query_resolution_messages(cleaned_query, history=history)
    system_instruction, contents = messages_to_gemini_args(messages)

    client = get_gemini_client()

    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        raw_text = response.text.strip()
        data = json.loads(raw_text)

        action = data.get("action", "KEEP").upper()
        if action not in ["KEEP", "REWRITE"]:
            action = "KEEP"

        final_query = data.get("query", cleaned_query).strip()
        if action == "KEEP" or not final_query:
            final_query = cleaned_query

        reason = data.get("reason", "Query evaluated by query_rewriter tool").strip()

        return {
            "action": action,
            "query": final_query,
            "reason": reason
        }

    except Exception as e:
        # Fallback safely to KEEP with original query
        return {
            "action": "KEEP",
            "query": cleaned_query,
            "reason": f"Fallback to original query due to query_rewriter exception: {str(e)}"
        }


def get_query_rewriter_tool() -> BaseTool:
    """
    Returns the standard LangChain query rewriter tool instance.
    """
    return query_rewriter


def resolve_conversational_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convenience function that invokes the query_rewriter tool.
    Maintains drop-in compatibility for functional callers.
    """
    return query_rewriter.invoke({"query": query, "history": history})
