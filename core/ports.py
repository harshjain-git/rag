# core/ports.py

"""Core Protocols (Ports) for Cadet Readiness Advisor.

Defines provider-agnostic interfaces for Tools, LLMs, and Retrievers.
All external adapters conform to these runtime-checkable contracts.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from core.models import AgentState


@runtime_checkable
class Tool(Protocol):
    """Protocol for state-aware tools invoked by the agent pipeline."""

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        """Execute the tool against the current AgentState."""
        ...


@runtime_checkable
class LLM(Protocol):
    """Minimal abstraction over any LLM provider."""

    def generate_json(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a prompt and receive a parsed JSON dict back."""
        ...

    def generate_text(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt and receive plain text back."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Minimal abstraction over any vector store retrieval backend."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence chunks passing the similarity threshold."""
        ...

    def retrieve_raw(
        self,
        query: str,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K results without distance filtering."""
        ...


__all__ = ["Tool", "LLM", "Retriever"]
