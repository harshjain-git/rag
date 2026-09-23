# agent/tools/generation_tool.py

"""Generation tool adapter.

Wraps the ``execute_generation`` step function from ``agent.chain`` into the
``Tool`` protocol so it can be registered in the ``ToolRegistry``.
"""

from typing import Dict, Any
from core.models import AgentState


class GenerationTool:
    """Registered as ``"generate_answer"`` in the ToolRegistry."""

    name = "generate_answer"
    description = "Generates a grounded answer with citations from retrieved evidence."

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        from agent.chain import execute_generation
        return execute_generation(state)
