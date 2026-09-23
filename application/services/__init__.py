# application.services package

"""Application services for Cadet Readiness Advisor."""

from .agent_service import AgentService, get_agent_service

__all__ = ["AgentService", "get_agent_service"]
