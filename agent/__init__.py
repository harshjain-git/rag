"""
Agent module for Cadet Readiness Advisor.
Orchestrates agentic RAG workflows using the generic AgentPipeline and ToolRegistry.
"""

from core.models import AgentState
from agent.pipeline import AgentPipeline, get_agent_pipeline
from agent.chain import build_agent_pipeline
from agent.verification import verify_grounding
import core.bootstrap

__all__ = [
    "AgentState",
    "AgentPipeline",
    "build_agent_pipeline",
    "get_agent_pipeline",
    "verify_grounding",
]
