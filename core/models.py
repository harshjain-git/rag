# core/models.py

"""Domain models for Cadet Readiness Advisor.

Defines the core AgentState TypedDict schema used throughout the application.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """Schema for the LangChain agent workflow state."""

    query: str
    resolved_query: Optional[str]
    resolution_action: Optional[str]
    resolution_reason: Optional[str]
    history: Optional[List[Dict[str, Any]]]
    evidence: List[Dict[str, Any]]
    tools_used: List[str]
    status: str
    metadata: Dict[str, Any]
    # Response fields
    answer: Optional[str]
    raw_answer: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    top_score: Optional[float]
    is_grounded: Optional[bool]
    is_answerable: Optional[bool]
    evidence_sufficient: Optional[bool]
    is_verified: Optional[bool]
    verification_status: Optional[str]
    verification_details: Optional[str]
    questions: Optional[List[Dict[str, Any]]]
    generation_metadata: Optional[Dict[str, Any]]


__all__ = ["AgentState"]
