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


def execute_orchestrator(state: AgentState) -> AgentState:
    """
    Autonomous Orchestrator decision node.
    Inspects user query and conversation history, reasons about user intent
    against registered tool capabilities, and decides:
    - Tool invocation (with arguments)
    - Or direct response
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "query": "",
            "status": "EMPTY_QUERY"
        }

    from agent.prompts import build_orchestrator_decision_messages, messages_to_gemini_args
    from generation.generator import generate_structured_json

    history = state.get("history")
    messages = build_orchestrator_decision_messages(query, history=history)
    sys_instruction, contents = messages_to_gemini_args(messages)

    try:
        decision_data = generate_structured_json(contents=contents, system_instruction=sys_instruction)
        action = decision_data.get("action", "call_tool")
        tool_name = decision_data.get("tool_name")
        args = decision_data.get("arguments", {})
        direct_answer = decision_data.get("direct_answer")
        reason = decision_data.get("reason", "")
    except Exception as e:
        # Fallback safely to standard factual retrieval flow
        action = "call_tool"
        tool_name = "retrieve_corpus_evidence"
        args = {"query": query}
        direct_answer = None
        reason = f"Fallback to retrieval: {str(e)}"

    tools_used = list(state.get("tools_used", []))

    # Case 1: Direct response (no tool needed)
    if action == "direct_response" and direct_answer:
        return {
            **state,
            "answer": direct_answer,
            "raw_answer": direct_answer,
            "is_grounded": False,
            "is_answerable": False,
            "status": "DIRECT_RESPONSE",
            "resolution_reason": reason
        }

    # Case 2: Question generation request
    if tool_name == "generate_corpus_questions":
        from agent.question_generator_tool import get_question_generator_tool
        q_tool = get_question_generator_tool()
        
        q_args = args if isinstance(args, dict) else {}
        if "num_questions" in q_args:
            try:
                q_args["num_questions"] = int(q_args["num_questions"])
            except (ValueError, TypeError):
                q_args["num_questions"] = 5
        
        gen_result = q_tool.invoke(q_args)
        
        if q_tool.name not in tools_used:
            tools_used.append(q_tool.name)

        questions = gen_result.get("questions", [])
        q_status = gen_result.get("status", "SUCCESS")
        coverage = gen_result.get("coverage", {})
        docs_covered = coverage.get("documents", [])
        
        if questions:
            exec_summary = (
                f"Generated {len(questions)} assessment questions grounded in corpus reference documents "
                f"({', '.join(docs_covered) if docs_covered else 'all corpus documents'})."
            )
        else:
            exec_summary = "Unable to generate verified questions: insufficient evidence in the reference documents."

        return {
            **state,
            "answer": exec_summary,
            "raw_answer": exec_summary,
            "questions": questions,
            "generation_metadata": gen_result,
            "tools_used": tools_used,
            "is_grounded": True if questions else False,
            "is_answerable": True if questions else False,
            "is_verified": True if questions else False,
            "verification_status": "VERIFIED_SUPPORTED" if questions else "INSUFFICIENT_EVIDENCE",
            "verification_details": f"Generated {len(questions)} questions strictly verified against corpus chunks.",
            "status": q_status
        }

    # Case 3: Query rewriter needed
    if tool_name == "query_rewriter":
        tool = get_query_rewriter_tool()
        rewriter_query = args.get("query", query) if isinstance(args, dict) else query
        resolution_data = tool.invoke({"query": rewriter_query, "history": history})
        rewriter_action = resolution_data.get("action", "KEEP")
        resolved_q = resolution_data.get("query", query)
        rewriter_reason = resolution_data.get("reason", "")

        if rewriter_action == "REWRITE":
            if tool.name not in tools_used:
                tools_used.append(tool.name)

        return {
            **state,
            "query": query,
            "resolved_query": resolved_q,
            "resolution_action": rewriter_action,
            "resolution_reason": rewriter_reason,
            "tools_used": tools_used,
            "status": "QUERY_RESOLVED"
        }

    # Case 4: Normal factual retrieval flow (tool_name == "retrieve_corpus_evidence" or default)
    retrieval_query = args.get("query", query) if isinstance(args, dict) else query
    return {
        **state,
        "query": query,
        "resolved_query": retrieval_query,
        "resolution_action": "KEEP",
        "resolution_reason": reason or "Standalone factual query routed to retrieval",
        "tools_used": tools_used,
        "status": "QUERY_RESOLVED"
    }


def execute_query_resolution(state: AgentState) -> AgentState:
    """Legacy alias for backward compatibility."""
    return execute_orchestrator(state)


def execute_retrieval(state: AgentState) -> AgentState:
    """
    Agent step that invokes the registered LangChain retrieval tool
    using the resolved query and places retrieved evidence into the agent state.
    """
    # If orchestrator already completed task (e.g. question generation or direct response), pass through
    if state.get("questions") is not None or state.get("status") in ["SUCCESS", "PARTIAL", "DIRECT_RESPONSE"]:
        return state

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
    # If orchestrator already completed task (e.g. question generation or direct response), pass through
    if state.get("questions") is not None or state.get("status") in ["SUCCESS", "PARTIAL", "DIRECT_RESPONSE"]:
        return state

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
    from generation.generator import generate_structured_json
    history = state.get("history")
    gen_messages = build_generation_messages(generation_query, evidence, history=history)
    system_instruction, contents = messages_to_gemini_args(gen_messages)

    try:
        data = generate_structured_json(contents=contents, system_instruction=system_instruction)
        is_answerable = bool(data.get("is_answerable", False))
        answer_text = str(data.get("answer", "")).strip()
    except Exception as e:
        answer_text = str(e)
        is_answerable = False

    if not is_answerable:
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
    # If orchestrator already completed task (e.g. question generation or direct response), pass through
    if state.get("questions") is not None or state.get("status") in ["SUCCESS", "PARTIAL", "DIRECT_RESPONSE"]:
        return state

    if state.get("status") == "EMPTY_QUERY":
        return {
            **state,
            "is_verified": False,
            "verification_status": "EMPTY_QUERY",
            "verification_details": "No query to verify."
        }

    if state.get("is_answerable") is False or state.get("status") in ["NOT_IN_CORPUS", "INSUFFICIENT_EVIDENCE"]:
        return {
            **state,
            "is_verified": True,
            "verification_status": "REFUSAL_CONFIRMED",
            "verification_details": "Verified refusal: Reference documents do not contain sufficient facts to answer this question."
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
