"""
LangChain Agent Foundation for Cadet Readiness Advisor (Step 1).
Provides the baseline LangChain agent management structure for orchestrating tools and responses.
"""

from typing import Dict, Any, List
from langchain_core.tools import BaseTool


class CadetAgentManager:
    """
    Manages the lifecycle, tools, and execution of the LangChain-based Cadet Advisor Agent.
    In Step 1, establishes the foundational structure and tools registry.
    """
    def __init__(self, tools: List[BaseTool] = None):
        self.tools: List[BaseTool] = tools if tools is not None else []

    def get_registered_tools(self) -> List[BaseTool]:
        """Returns the list of currently registered LangChain tools."""
        return self.tools

    def register_tool(self, tool: BaseTool) -> None:
        """Registers a new LangChain tool with the agent manager."""
        self.tools.append(tool)

    def run(self, query: str) -> Dict[str, Any]:
        """
        Baseline agent execution runner.
        In Step 1: Validates query, reports registered tools, and returns initial structured output.
        Subsequent steps will attach the retrieval tool and executor.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return {
                "query": "",
                "answer": "No query provided.",
                "tools_used": [],
                "status": "EMPTY_QUERY"
            }

        return {
            "query": cleaned_query,
            "answer": "Agent foundation initialized. Ready for tool integration.",
            "tools_used": [t.name for t in self.tools],
            "status": "INITIALIZED"
        }


# Singleton manager instance for the application
_AGENT_MANAGER = None


def get_agent_manager() -> CadetAgentManager:
    """Returns a singleton instance of CadetAgentManager."""
    global _AGENT_MANAGER
    if _AGENT_MANAGER is None:
        _AGENT_MANAGER = CadetAgentManager()
    return _AGENT_MANAGER
