"""
Agent module for Cadet Readiness Advisor.
Orchestrates agentic RAG workflows using the LangChain agent and tools ecosystem.
"""

from agent.agent_manager import CadetAgentManager, get_agent_manager

__all__ = ["CadetAgentManager", "get_agent_manager"]
