"""
LangChain Agent Workflow Pipeline for Cadet Readiness Advisor (Step 5).
Chains retrieval, grounded generation, and verification into a modular LangChain runnable pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.runnables import RunnableLambda
import config
from agent.state import AgentState
from agent.retrieval_tool import get_retrieval_tool
from agent.query_rewriter_tool import get_query_rewriter_tool
from agent.verification import verify_grounding


def execute_query_resolution(state: AgentState) -> AgentState:
    """
    Agent step that analyzes user query with a single LLM call to decide
    whether to KEEP the query as-is or REWRITE it for effective retrieval.
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "query": "",
            "resolved_query": "",
            "resolution_action": "KEEP",
            "resolution_reason": "Empty query",
            "status": "EMPTY_QUERY"
        }

    history = state.get("history")
    tool = get_query_rewriter_tool()
    resolution_data = tool.invoke({"query": query, "history": history})
    action = resolution_data.get("action", "KEEP")
    resolved_query = resolution_data.get("query", query)
    reason = resolution_data.get("reason", "")

    tools_used = list(state.get("tools_used", []))
    if action == "REWRITE":
        if tool.name not in tools_used:
            tools_used.append(tool.name)

    return {
        **state,
        "query": query,
        "resolved_query": resolved_query,
        "resolution_action": action,
        "resolution_reason": reason,
        "tools_used": tools_used,
        "status": "QUERY_RESOLVED"
    }


def execute_retrieval(state: AgentState) -> AgentState:
    """
    Agent step that invokes the registered LangChain retrieval tool
    using the resolved query and places retrieved evidence into the agent state.
    """
    if state.get("status") == "EMPTY_QUERY":
        return {
            **state,
            "evidence": [],
            "tools_used": state.get("tools_used", []),
            "status": "EMPTY_QUERY"
        }

    retrieval_query = (state.get("resolved_query") or state.get("query", "")).strip()
    if not retrieval_query:
        return {
            **state,
            "evidence": [],
            "tools_used": state.get("tools_used", []),
            "status": "EMPTY_QUERY"
        }

    tool = get_retrieval_tool()
    evidence_chunks = tool.invoke({"query": retrieval_query})

    tools_used = list(state.get("tools_used", []))
    if tool.name not in tools_used:
        tools_used.append(tool.name)

    top_score = evidence_chunks[0]["score"] if evidence_chunks else None

    return {
        **state,
        "evidence": evidence_chunks,
        "tools_used": tools_used,
        "top_score": top_score,
        "status": "EVIDENCE_RETRIEVED"
    }


def execute_generation(state: AgentState) -> AgentState:
    """
    Agent step that connects the existing Gemini generation logic,
    generating a grounded response with citations or a natural refusal.
    """
    query = state.get("query", "").strip()
    if not query or state.get("status") == "EMPTY_QUERY":
        return {
            **state,
            "answer": "No query provided.",
            "raw_answer": "No query provided.",
            "citations": [],
            "is_grounded": False,
            "status": "EMPTY_QUERY"
        }

    evidence = state.get("evidence", [])
    top_score = state.get("top_score")

    # Reuse existing generation and citation modules
    from generation.generator import generate_natural_refusal, format_context_prompt, get_gemini_client, SYSTEM_PROMPT
    from citation.citation_engine import extract_citations, format_citations_block
    from google.genai import types

    generation_query = (state.get("resolved_query") or query).strip()

    # If no evidence passed threshold, generate natural refusal
    if not evidence:
        refusal_text = generate_natural_refusal(generation_query)
        return {
            **state,
            "answer": refusal_text,
            "raw_answer": refusal_text,
            "citations": [],
            "is_grounded": False,
            "status": "NOT_IN_CORPUS"
        }

    from agent.prompts import build_generation_messages, messages_to_gemini_args
    history = state.get("history")
    gen_messages = build_generation_messages(generation_query, evidence, history=history)
    system_instruction, contents = messages_to_gemini_args(gen_messages)
    client = get_gemini_client()

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
        answer_text = response.text.strip()
    except Exception as e:
        answer_text = f"Error generating LLM response: {str(e)}"

    def is_refusal_answer(text: str) -> bool:
        normalized = text.strip().lower()
        refusal_markers = [
            "not in corpus",
            "not contain enough information",
            "do not contain enough facts",
            "does not contain information",
            "cannot be answered using the provided",
            "no information provided",
            "not mentioned in the provided",
            "not found in the provided"
        ]
        return any(marker in normalized for marker in refusal_markers)

    is_refusal = is_refusal_answer(answer_text)

    if is_refusal:
        return {
            **state,
            "answer": answer_text,
            "raw_answer": answer_text,
            "citations": [],
            "is_grounded": False,
            "is_answerable": False,
            "evidence_sufficient": False,
            "evidence": evidence,
            "status": "INSUFFICIENT_EVIDENCE"
        }

    citations = extract_citations(evidence)
    citations_text = format_citations_block(citations)
    full_answer = f"{answer_text}{citations_text}"

    return {
        **state,
        "answer": full_answer,
        "raw_answer": answer_text,
        "citations": citations,
        "is_grounded": True,
        "is_answerable": True,
        "evidence_sufficient": True,
        "evidence": evidence,
        "status": "GROUNDED"
    }


def execute_verification(state: AgentState) -> AgentState:
    """
    Agent step that validates the factual grounding of the generated answer
    against the retrieved evidence chunks.
    """
    if state.get("status") == "EMPTY_QUERY":
        return {
            **state,
            "is_verified": False,
            "verification_status": "EMPTY_QUERY",
            "verification_details": "No query to verify."
        }

    verification_query = (state.get("resolved_query") or state.get("query", "")).strip()
    raw_answer = state.get("raw_answer", "")
    evidence = state.get("evidence", [])

    verification_result = verify_grounding(
        query=verification_query,
        raw_answer=raw_answer,
        evidence=evidence
    )

    return {
        **state,
        "is_verified": verification_result.get("is_verified", False),
        "verification_status": verification_result.get("verification_status", "UNKNOWN"),
        "verification_details": verification_result.get("verification_details", "")
    }


def build_agent_pipeline():
    """
    Constructs the pure LangChain Runnable execution pipeline for Phase 8 Step 1.
    Pipeline: execute_query_resolution | execute_retrieval | execute_generation | execute_verification
    """
    return (
        RunnableLambda(execute_query_resolution)
        | RunnableLambda(execute_retrieval)
        | RunnableLambda(execute_generation)
        | RunnableLambda(execute_verification)
    )


# Singleton pipeline instance
_AGENT_PIPELINE = None


def get_agent_pipeline():
    """Returns a singleton instance of the LangChain Agent Pipeline."""
    global _AGENT_PIPELINE
    if _AGENT_PIPELINE is None:
        _AGENT_PIPELINE = build_agent_pipeline()
    return _AGENT_PIPELINE
