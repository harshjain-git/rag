"""Agent workflow step implementations for Cadet Readiness Advisor.

Defines the individual execution step functions coordinated by AgentPipeline:
- execute_orchestrator: Autonomous routing via Gemini Native Tool Calling
- execute_retrieval: Evidence chunk retrieval from ChromaDB vector store
- execute_generation: Grounded answer generation with citations via Gemini
- execute_verification: Fact verification against retrieved evidence
- execute_srag_database: Structured query execution via remote SRAG agent
- execute_question_generator: Assessment question generation from corpus
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.models import AgentState
from agent.verification import verify_grounding


def execute_orchestrator(state: AgentState, _depth: int = 0) -> AgentState:
    """Autonomous Orchestrator decision node using Gemini Native Tool Calling.

    Sends query and conversation history to Gemini along with native tool
    declarations. Dispatches execution based on the returned native function call
    or direct text response without requiring prompt-enforced JSON.
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "query": "",
            "status": "EMPTY_QUERY",
        }

    from application.prompts import (
        build_orchestrator_decision_messages,
        messages_to_gemini_args,
    )
    from infrastructure.gemini import get_llm
    from agent.tool_declarations import get_gemini_native_tools

    history = state.get("history")
    messages = build_orchestrator_decision_messages(query, history=history)
    sys_instruction, contents = messages_to_gemini_args(messages)
    tools = get_gemini_native_tools()

    try:
        response_data = get_llm().generate_with_tools(
            contents=contents,
            tools=tools,
            system_instruction=sys_instruction,
            temperature=0.0,
        )
    except Exception as e:
        # Fallback safely to standard factual retrieval flow on model or network error
        return {
            **state,
            "query": query,
            "resolved_query": query,
            "resolution_action": "KEEP",
            "resolution_reason": f"Fallback to retrieval (API error): {str(e)}",
            "tools_used": list(state.get("tools_used", [])),
            "status": "QUERY_RESOLVED",
        }

    call_type = response_data.get("call_type", "text")
    tools_used = list(state.get("tools_used", []))

    # Case 1: Direct text response (model chose not to call any tool)
    if call_type == "text":
        direct_text = response_data.get("text", "").strip()
        if not direct_text:
            # Fallback to retrieval if model returned empty text
            return {
                **state,
                "query": query,
                "resolved_query": query,
                "resolution_action": "KEEP",
                "resolution_reason": "Fallback to retrieval: empty text response from orchestrator",
                "tools_used": tools_used,
                "status": "QUERY_RESOLVED",
            }
        return {
            **state,
            "answer": direct_text,
            "raw_answer": direct_text,
            "is_grounded": False,
            "is_answerable": False,
            "status": "DIRECT_RESPONSE",
            "resolution_reason": "Direct text response from orchestrator model",
        }

    # Case 2: Native Function Call
    tool_name = response_data.get("function_name")
    args = response_data.get("arguments", {})
    if not isinstance(args, dict):
        args = {}

    # Tool A: Structured Database SRAG Agent
    if tool_name == "ask_srag_database":
        srag_query = args.get("query", query) or query
        return execute_srag_database(state, query=srag_query)

    # Tool B: Question Generation Request
    if tool_name == "generate_corpus_questions":
        return execute_question_generator(state, args=args)

    # Tool C: Query Rewriter Needed
    if tool_name == "query_rewriter":
        rewrite_query = args.get("query", query) or query
        from agent.query_rewriter_tool import resolve_conversational_query
        resolution_data = resolve_conversational_query(rewrite_query, history=history)
        rewriter_action = resolution_data.get("action", "KEEP")
        resolved_q = resolution_data.get("query", query)
        rewriter_reason = resolution_data.get("reason", "")

        if rewriter_action == "REWRITE":
            if "query_rewriter" not in tools_used:
                tools_used.append("query_rewriter")

        # Re-evaluate with orchestrator once if query was rewritten to route downstream properly
        if _depth == 0 and rewriter_action == "REWRITE" and resolved_q != query:
            updated_state = {
                **state,
                "query": resolved_q,
                "resolved_query": resolved_q,
                "resolution_action": rewriter_action,
                "resolution_reason": rewriter_reason,
                "tools_used": tools_used,
            }
            return execute_orchestrator(updated_state, _depth=_depth + 1)

        return {
            **state,
            "query": query,
            "resolved_query": resolved_q,
            "resolution_action": rewriter_action,
            "resolution_reason": rewriter_reason,
            "tools_used": tools_used,
            "status": "QUERY_RESOLVED",
        }

    # Tool D: Document Corpus Retrieval (tool_name == "retrieve_corpus_evidence" or fallback)
    if tool_name == "retrieve_corpus_evidence":
        retrieval_query = args.get("query", query) or query
        return {
            **state,
            "query": query,
            "resolved_query": retrieval_query,
            "resolution_action": "KEEP",
            "resolution_reason": "Routed to corpus document retrieval via native tool call",
            "tools_used": tools_used,
            "status": "QUERY_RESOLVED",
        }

    # Unrecognized tool fallback
    return {
        **state,
        "query": query,
        "resolved_query": query,
        "resolution_action": "KEEP",
        "resolution_reason": f"Fallback to retrieval: unrecognized native tool '{tool_name}'",
        "tools_used": tools_used,
        "status": "QUERY_RESOLVED",
    }


def execute_retrieval(state: AgentState) -> AgentState:
    """Agent step that retrieves evidence chunks from vector store using resolved query."""
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

    from infrastructure.chroma import retrieve_evidence
    evidence_chunks = retrieve_evidence(query=retrieval_query)

    tools_used = list(state.get("tools_used", []))
    if "retrieve_corpus_evidence" not in tools_used:
        tools_used.append("retrieve_corpus_evidence")

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
    from infrastructure.gemini import generate_natural_refusal, generate_structured_json
    from citation.citation_engine import extract_citations, format_citations_block

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

    from application.prompts import build_generation_messages, messages_to_gemini_args
    history = state.get("history")
    gen_messages = build_generation_messages(generation_query, evidence, history=history)
    system_instruction, contents = messages_to_gemini_args(gen_messages)

    try:
        data = generate_structured_json(contents=contents, system_instruction=system_instruction)
        if isinstance(data, dict):
            is_answerable = bool(data.get("is_answerable", False))
            answer_text = str(data.get("answer", "")).strip()
        else:
            is_answerable = False
            answer_text = str(data).strip()
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

    tools_used = list(state.get("tools_used", []))
    if "verify_grounding" not in tools_used:
        tools_used.append("verify_grounding")

    return {
        **state,
        "tools_used": tools_used,
        "is_verified": verification_result.get("is_verified", False),
        "verification_status": verification_result.get("verification_status", "UNKNOWN"),
        "verification_details": verification_result.get("verification_details", "")
    }


def execute_srag_database(state: AgentState, query: Optional[str] = None) -> AgentState:
    """Agent step that queries the structured relational database via remote SRAG agent over A2A."""
    from infrastructure.a2a_adapter import ask_srag_database

    q = (query or state.get("resolved_query") or state.get("query", "")).strip()
    answer = ask_srag_database(q)

    tools_used = list(state.get("tools_used", []))
    if "ask_srag_database" not in tools_used:
        tools_used.append("ask_srag_database")

    return {
        **state,
        "answer": answer,
        "raw_answer": answer,
        "query": state.get("query", "") or q,
        "resolved_query": q,
        "tools_used": tools_used,
        "is_grounded": False,
        "is_answerable": True,
        "is_verified": False,
        "resolution_action": "DATABASE_QUERY",
        "resolution_reason": "Structured database question answered via remote SRAG A2A agent",
        "status": "DIRECT_RESPONSE",
    }


def execute_question_generator(state: AgentState, args: Optional[Dict[str, Any]] = None) -> AgentState:
    """Agent step that generates verified assessment questions grounded in corpus reference documents."""
    from agent.question_generator_tool import generate_corpus_questions

    q_args = dict(args or {})
    num_q = q_args.get("num_questions", 5)
    try:
        num_q = int(num_q)
    except (ValueError, TypeError):
        num_q = 5
    domains = q_args.get("domains")

    gen_result = generate_corpus_questions(num_questions=num_q, domains=domains)

    tools_used = list(state.get("tools_used", []))
    if "generate_corpus_questions" not in tools_used:
        tools_used.append("generate_corpus_questions")

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

    evidence_list = []
    for q in questions:
        for c in q.get("citations", []):
            evidence_list.append({
                "text": c.get("quote", ""),
                "source": c.get("source", ""),
                "page": c.get("page", 0),
                "chunk_id": c.get("chunk_id", ""),
            })

    return {
        **state,
        "answer": exec_summary,
        "raw_answer": exec_summary,
        "questions": questions,
        "evidence": evidence_list,
        "generation_metadata": gen_result,
        "tools_used": tools_used,
        "is_grounded": True if questions else False,
        "is_answerable": True if questions else False,
        "is_verified": True if questions else False,
        "verification_status": "VERIFIED_SUPPORTED" if questions else "INSUFFICIENT_EVIDENCE",
        "verification_details": f"Generated {len(questions)} questions strictly verified against corpus chunks.",
        "status": q_status,
    }


__all__ = [
    "execute_orchestrator",
    "execute_retrieval",
    "execute_generation",
    "execute_verification",
    "execute_srag_database",
    "execute_question_generator",
]

