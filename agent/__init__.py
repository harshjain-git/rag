"""
Agent module for Cadet Readiness Advisor.
Orchestrates agentic RAG workflows using the generic AgentPipeline and ToolRegistry.
"""

from agent.agent_manager import CadetAgentManager, get_agent_manager
from agent.retrieval_tool import retrieve_corpus_evidence, get_retrieval_tool
from core.models import AgentState
from agent.pipeline import AgentPipeline, get_agent_pipeline
from agent.chain import build_agent_pipeline
from agent.verification import verify_grounding
import core.bootstrap

__all__ = [
    "CadetAgentManager",
    "get_agent_manager",
    "retrieve_corpus_evidence",
    "get_retrieval_tool",
    "AgentState",
    "AgentPipeline",
    "build_agent_pipeline",
    "get_agent_pipeline",
    "verify_grounding",
]
