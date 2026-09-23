# Core contracts for tools

"""Tool contract used by the AgentPipeline.

All concrete tools (retrieval, query rewrites, generation, verification, etc.)
must implement this protocol. The contract is **state‑aware**: the tool receives
the current `AgentState` and a dictionary of tool‑specific inputs, and returns
the (potentially) mutated `AgentState`.

This design allows the orchestrator to request a tool by name without knowing
its concrete implementation, and the pipeline can invoke any registered tool
generically.
"""

from typing import Protocol, Dict, Any
from core.models import AgentState


class Tool(Protocol):
    """Protocol that all tool implementations must follow.

    The `invoke` method receives the full mutable `AgentState` and a dict of
    `inputs` specific to the tool. It returns the updated `AgentState` (or the
    same object if no changes are needed).
    """

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        """Execute the tool.

        Args:
            state: Current agent state that may be read or mutated.
            inputs: Tool‑specific parameters (e.g., query string for a retriever).

        Returns:
            Updated `AgentState` after the tool has performed its work.
        """
        ...
