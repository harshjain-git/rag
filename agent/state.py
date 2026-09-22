"""
Agent State Definition for Cadet Readiness Advisor.
Defines the TypedDict schema for the LangChain agent workflow state.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    Schema for the LangChain Agent workflow state across all stages.
    """
    query: str
    evidence: List[Dict[str, Any]]
    tools_used: List[str]
    status: str
    metadata: Dict[str, Any]
    # Reserved for subsequent migration steps:
    answer: Optional[str]
    raw_answer: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    top_score: Optional[float]
    is_grounded: Optional[bool]
    is_verified: Optional[bool]
    verification_status: Optional[str]
    verification_details: Optional[str]
