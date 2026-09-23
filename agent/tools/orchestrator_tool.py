# agent/tools/orchestrator_tool.py

"""Orchestrator tool adapter.

Wraps the ``execute_orchestrator`` step function from ``agent.chain`` into the
``Tool`` protocol so it can be registered in the ``ToolRegistry``.
"""

from typing import Dict, Any
from core.models import AgentState


class OrchestratorTool:
    """Registered as ``"orchestrator"`` in the ToolRegistry."""

    name = "orchestrator"
    description = "Decides which tool to invoke or whether to respond directly."

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        from agent.chain import execute_orchestrator
        return execute_orchestrator(state)
