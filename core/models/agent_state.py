# Core domain models

"""AgentState model definition used across the application.

We keep the original TypedDict schema (the existing code already uses a TypedDict),
but relocate it into the `core.models` package so that higher‑level layers depend only
on this stable contract.

If later we decide to migrate to a Pydantic BaseModel, the change will be isolated
here without touching the rest of the codebase.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """Schema for the LangChain agent workflow state.

    The keys mirror the fields used throughout the current pipeline. All fields are
    optional to allow step‑wise population of the state.
    """
    query: str
    resolved_query: Optional[str]
    resolution_action: Optional[str]
    resolution_reason: Optional[str]
    history: Optional[List[Dict[str, Any]]]
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
    is_answerable: Optional[bool]
    evidence_sufficient: Optional[bool]
    is_verified: Optional[bool]
    verification_status: Optional[str]
    verification_details: Optional[str]
    questions: Optional[List[Dict[str, Any]]]
    generation_metadata: Optional[Dict[str, Any]]
