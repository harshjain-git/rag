# agent/tools.py

"""Unified Tool Adapters for Cadet Readiness Advisor.

Wraps step functions and capabilities into the state-aware ``Tool`` protocol
defined in ``core.ports.tool`` for registration in the ``ToolRegistry``.
"""

from typing import Dict, Any, Optional
from core.models import AgentState


class OrchestratorTool:
    """Registered as ``"orchestrator"`` in the ToolRegistry."""

    name = "orchestrator"
    description = "Decides which tool to invoke or whether to respond directly."

    def invoke(self, state: AgentState, inputs: Optional[Dict[str, Any]] = None) -> AgentState:
        from agent.chain import execute_orchestrator
        return execute_orchestrator(state)


class RetrievalTool:
    """Registered as ``"retrieve_corpus_evidence"`` in the ToolRegistry."""

    name = "retrieve_corpus_evidence"
    description = "Retrieves relevant evidence chunks from the corpus vector store."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        if inputs is None and isinstance(state_or_inputs, dict) and "evidence" not in state_or_inputs:
            from infrastructure.chroma import retrieve_evidence
            q = state_or_inputs.get("query", "")
            return retrieve_evidence(query=q)

        state: AgentState = state_or_inputs
        from agent.chain import execute_retrieval
        return execute_retrieval(state)


class GenerationTool:
    """Registered as ``"generate_answer"`` in the ToolRegistry."""

    name = "generate_answer"
    description = "Generates a grounded answer with citations from retrieved evidence."

    def invoke(self, state: AgentState, inputs: Optional[Dict[str, Any]] = None) -> AgentState:
        from agent.chain import execute_generation
        return execute_generation(state)


class VerificationTool:
    """Registered as ``"verify_grounding"`` and ``"verification"`` in the ToolRegistry."""

    name = "verify_grounding"
    description = "Verifies factual grounding of generated answer against retrieved evidence."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        if inputs is None and isinstance(state_or_inputs, dict) and "status" not in state_or_inputs:
            from agent.verification import verify_grounding
            return verify_grounding(
                query=state_or_inputs.get("query", ""),
                raw_answer=state_or_inputs.get("raw_answer", ""),
                evidence=state_or_inputs.get("evidence", []),
                is_answerable=state_or_inputs.get("is_answerable"),
            )

        state: AgentState = state_or_inputs
        from agent.chain import execute_verification
        return execute_verification(state)


class QueryRewriterTool:
    """Registered as ``"query_rewriter"`` in the ToolRegistry."""

    name = "query_rewriter"
    description = "Resolves conversational context and rewrites user query if needed."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        from agent.query_rewriter_tool import get_query_rewriter_tool
        tool = get_query_rewriter_tool()

        if inputs is None and isinstance(state_or_inputs, dict) and "status" not in state_or_inputs:
            return tool.invoke(state_or_inputs)

        state: AgentState = state_or_inputs
        tool_inputs = inputs or {
            "query": state.get("query", ""),
            "history": state.get("history", []),
        }
        resolution_data = tool.invoke(tool_inputs)
        rewriter_action = resolution_data.get("action", "KEEP")
        resolved_q = resolution_data.get("query", state.get("query", ""))
        rewriter_reason = resolution_data.get("reason", "")

        tools_used = list(state.get("tools_used", []))
        if rewriter_action == "REWRITE" and self.name not in tools_used:
            tools_used.append(self.name)

        return {
            **state,
            "resolved_query": resolved_q,
            "resolution_action": rewriter_action,
            "resolution_reason": rewriter_reason,
            "tools_used": tools_used,
            "status": "QUERY_RESOLVED",
        }


class QuestionGeneratorTool:
    """Registered as ``"generate_corpus_questions"`` in the ToolRegistry."""

    name = "generate_corpus_questions"
    description = "Generates verified assessment questions directly from the corpus."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        from agent.question_generator_tool import generate_corpus_questions

        if inputs is None and isinstance(state_or_inputs, dict) and "status" not in state_or_inputs:
            count = state_or_inputs.get("count", 5)
            domains = state_or_inputs.get("domains")
            return generate_corpus_questions(num_questions=count, domains=domains)

        state: AgentState = state_or_inputs
        count = (inputs or {}).get("count", 5)
        domains = (inputs or {}).get("domains")
        q_result = generate_corpus_questions(num_questions=count, domains=domains)
        questions = q_result.get("questions", [])
        q_status = q_result.get("status", "SUCCESS")
        tools_used = list(state.get("tools_used", []))
        if self.name not in tools_used:
            tools_used.append(self.name)

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
            "questions": questions,
            "evidence": evidence_list,
            "answer": f"Generated {len(questions)} verified assessment questions from the reference corpus.",
            "tools_used": tools_used,
            "is_grounded": True if questions else False,
            "is_answerable": True if questions else False,
            "is_verified": True if questions else False,
            "verification_status": "VERIFIED_SUPPORTED" if questions else "INSUFFICIENT_EVIDENCE",
            "verification_details": f"Generated {len(questions)} questions strictly verified against corpus chunks.",
            "status": q_status,
        }


__all__ = [
    "OrchestratorTool",
    "RetrievalTool",
    "GenerationTool",
    "VerificationTool",
    "QueryRewriterTool",
    "QuestionGeneratorTool",
]
