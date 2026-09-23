# agent/tools/query_rewriter_adapter.py

"""Query rewriter tool adapter.

Wraps the query rewriter into the ``Tool`` protocol so it can be registered
in the ``ToolRegistry``.
"""

from typing import Dict, Any, Optional
from core.models import AgentState


class QueryRewriterTool:
    """Registered as ``"query_rewriter"`` in the ToolRegistry."""

    name = "query_rewriter"
    description = "Resolves conversational context and rewrites user query if needed."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute query rewriting.

        Supports both:
        1. LangChain inputs dict: ``invoke({"query": ..., "history": ...})`` -> ``Dict``
        2. State-aware invocation: ``invoke(state, inputs)`` -> ``AgentState``
        """
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
