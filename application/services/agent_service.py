# application/services/agent_service.py

"""Application Service facade for the Cadet Readiness Advisor.

Provides a unified, high-level API for UI clients (such as the Streamlit app)
to interact with the underlying AgentPipeline without directly coupling to
internal pipeline states, tool registries, or orchestration logic.
"""

from typing import Dict, Any, List, Optional
from core.models import AgentState
from agent.pipeline import AgentPipeline, get_agent_pipeline


class AgentService:
    """Service facade encapsulating agent pipeline execution and response formatting."""

    def __init__(self, pipeline: Optional[AgentPipeline] = None) -> None:
        self._pipeline = pipeline or get_agent_pipeline()

    def query(
        self,
        user_query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute a query through the advisor agent and return a formatted message dict.

        Args:
            user_query: The raw string query from the user.
            history: Optional list of past chat messages.

        Returns:
            A clean dictionary formatted for UI consumption and session persistence:
            - 'role': 'assistant'
            - 'content': generated answer text
            - 'original_query': original user input
            - 'resolved_query': rewritten or normalized query
            - 'resolution_action': 'KEEP' or 'REWRITE'
            - 'resolution_reason': reason string
            - 'is_grounded': bool
            - 'is_answerable': bool
            - 'is_verified': bool
            - 'verification_status': e.g. 'VERIFIED_SUPPORTED'
            - 'verification_details': string explanation
            - 'status': terminal status (e.g. 'GROUNDED', 'DIRECT_RESPONSE')
            - 'tools_used': list of tools invoked
            - 'top_score': float distance or similarity
            - 'evidence': list of evidence chunks
            - 'questions': list of quiz questions if generated
            - 'generation_metadata': dictionary of extra metadata
        """
        raw_state: AgentState = self._pipeline.run(query=user_query, history=history)
        return self._format_assistant_message(raw_state, user_query)

    def _format_assistant_message(
        self,
        state: AgentState,
        original_query: str,
    ) -> Dict[str, Any]:
        """Format an AgentState into the standardized UI message dictionary."""
        is_grounded = bool(state.get("is_grounded", False))
        is_answerable = state.get("is_answerable")
        if is_answerable is None:
            is_answerable = is_grounded

        return {
            "role": "assistant",
            "content": state.get("answer", "") or "",
            "original_query": state.get("query") or original_query,
            "resolved_query": state.get("resolved_query"),
            "resolution_action": state.get("resolution_action", "KEEP"),
            "resolution_reason": state.get("resolution_reason", ""),
            "is_grounded": is_grounded,
            "is_answerable": bool(is_answerable),
            "is_verified": bool(state.get("is_verified", False)),
            "verification_status": state.get("verification_status", "") or "",
            "verification_details": state.get("verification_details", "") or "",
            "status": state.get("status", "") or "",
            "tools_used": list(state.get("tools_used", [])),
            "top_score": state.get("top_score"),
            "evidence": list(state.get("evidence", [])),
            "questions": list(state.get("questions", [])) if state.get("questions") is not None else [],
            "generation_metadata": dict(state.get("generation_metadata", {})),
        }


# Module-level singleton
_AGENT_SERVICE: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """Return the singleton instance of AgentService."""
    global _AGENT_SERVICE
    if _AGENT_SERVICE is None:
        _AGENT_SERVICE = AgentService()
    return _AGENT_SERVICE
