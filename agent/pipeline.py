# agent/pipeline.py

"""Agent Execution Pipeline for Cadet Readiness Advisor.

Coordinates end-to-end query execution:
1. Autonomous orchestrator routing (Gemini Native Tool Calling)
2. Terminal early-exit handling (direct responses, quiz generation, empty queries)
3. Standard document retrieval-augmented workflow (retrieve -> generate -> verify)
"""

from typing import Dict, Any, List, Optional

from core.models import AgentState
from core.registry import ToolRegistry


class AgentPipeline:
    """Registry-driven agent execution pipeline.

    The pipeline coordinates execution by looking up step callables from the
    ``ToolRegistry``. The orchestrator decides the initial routing; if retrieval
    is required, the standard sequence (retrieve -> generate -> verify) executes.
    """

    # Standard workflow executed after the orchestrator decision.
    # Each name must correspond to a tool registered in ToolRegistry.
    STANDARD_WORKFLOW: List[str] = [
        "retrieve_corpus_evidence",
        "generate_answer",
        "verify_grounding",
    ]

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        """Initialise the pipeline.

        Args:
            registry: Optional ``ToolRegistry`` class/instance. Falls back to
                the global ``ToolRegistry``.
        """
        self._registry = registry or ToolRegistry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentState:
        """Execute the full agent pipeline for a user query.

        Args:
            query: The user's question.
            history: Optional conversation history (list of message dicts).

        Returns:
            Completed ``AgentState`` dict with answer, evidence, status, etc.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return {
                "query": "",
                "resolved_query": "",
                "evidence": [],
                "tools_used": [],
                "status": "EMPTY_QUERY",
            }

        state: AgentState = {
            "query": cleaned_query,
            "resolved_query": None,
            "history": history if history is not None else [],
            "evidence": [],
            "tools_used": [],
            "status": "START",
        }

        # Step 1: Orchestrator decision
        state = self._run_orchestrator(state)

        # If orchestrator already completed the task (e.g. greeting or quiz), return early
        if self._is_terminal(state):
            return state

        # Step 2: Execute the standard workflow (retrieval -> generation -> verification)
        for tool_name in self.STANDARD_WORKFLOW:
            step_func = self._registry.get(tool_name)
            state = step_func(state)

            # If a step resulted in an early-exit condition, break cleanly
            if self._is_terminal(state):
                break

        return state

    def invoke(self, state: AgentState) -> AgentState:
        """LangChain-compatible execution method accepting an initial state dict."""
        return self.run(
            query=state.get("query", ""),
            history=state.get("history", []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_orchestrator(self, state: AgentState) -> AgentState:
        """Invoke the registered orchestrator tool callable."""
        orchestrator = self._registry.get("orchestrator")
        return orchestrator(state)

    @staticmethod
    def _is_terminal(state: AgentState) -> bool:
        """Check whether the pipeline should terminate early."""
        terminal_statuses = {
            "DIRECT_RESPONSE",
            "EMPTY_QUERY",
        }
        if state.get("status") in terminal_statuses:
            return True
        if state.get("questions") is not None:
            return True
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_AGENT_PIPELINE: AgentPipeline | None = None


def get_agent_pipeline() -> AgentPipeline:
    """Return a module-level singleton ``AgentPipeline`` instance."""
    global _AGENT_PIPELINE
    if _AGENT_PIPELINE is None:
        _AGENT_PIPELINE = AgentPipeline()
    return _AGENT_PIPELINE
