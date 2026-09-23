# agent/tools/verification_tool.py

"""Verification tool adapter.

Wraps the ``execute_verification`` step function from ``agent.chain`` into the
``Tool`` protocol so it can be registered in the ``ToolRegistry``.
"""

from typing import Dict, Any, Optional
from core.models import AgentState


class VerificationTool:
    """Registered as ``"verify_grounding"`` and ``"verification"`` in the ToolRegistry."""

    name = "verify_grounding"
    description = "Verifies factual grounding of generated answer against retrieved evidence."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute verification on the agent state.

        Supports:
        1. State-aware pipeline invocation: ``invoke(state, inputs)`` -> ``AgentState``
        2. Direct verification args: ``invoke({"query": ..., "raw_answer": ..., "evidence": ...})``
        """
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
