# agent/tools/retrieval_adapter.py

"""Retrieval tool adapter.

Wraps the retrieval functionality into the ``Tool`` protocol so it can be registered
in the ``ToolRegistry`` and invoked generically by the ``AgentPipeline``.
"""

from typing import Dict, Any, Optional
from core.models import AgentState


class RetrievalTool:
    """Registered as ``"retrieve_corpus_evidence"`` in the ToolRegistry."""

    name = "retrieve_corpus_evidence"
    description = "Retrieves relevant evidence chunks from the corpus vector store."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute retrieval.

        Supports both:
        1. LangChain tool invocation: ``invoke({"query": "..."})`` -> ``List[Dict]``
        2. State-aware pipeline invocation: ``invoke(state, inputs)`` -> ``AgentState``
        """
        # Case 1: Caller passed inputs dict directly (e.g. from chain.py execute_retrieval)
        if inputs is None and isinstance(state_or_inputs, dict) and "evidence" not in state_or_inputs:
            from retrieval.retriever import retrieve_evidence
            q = state_or_inputs.get("query", "")
            return retrieve_evidence(query=q)

        # Case 2: State-aware pipeline execution
        state: AgentState = state_or_inputs
        from agent.chain import execute_retrieval
        return execute_retrieval(state)
