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

    def run(self, query: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the agent graph workflow for a user query.
        Phase 8 Step 1: Resolves query using history -> LangChain retrieval tool -> Grounded Generation -> Verification.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return {
                "query": "",
                "resolved_query": "",
                "evidence": [],
                "tools_used": [],
                "status": "EMPTY_QUERY"
            }

        from agent.pipeline import get_agent_pipeline

        pipeline = get_agent_pipeline()
        result_state = pipeline.run(query=cleaned_query, history=history)
        return result_state


# Singleton manager instance for the application
_AGENT_MANAGER = None


def get_agent_manager() -> CadetAgentManager:
    """Returns a singleton instance of CadetAgentManager with registered tools."""
    global _AGENT_MANAGER
    if _AGENT_MANAGER is None:
        from agent.retrieval_tool import get_retrieval_tool
        from agent.query_rewriter_tool import get_query_rewriter_tool
        from agent.question_generator_tool import get_question_generator_tool
        _AGENT_MANAGER = CadetAgentManager(tools=[
            get_retrieval_tool(),
            get_query_rewriter_tool(),
            get_question_generator_tool()
        ])
    return _AGENT_MANAGER

