"""Central registry that maps tool names to concrete tool implementations."""

from typing import Dict, List, Optional
from core.ports import Tool


class ToolRegistry:
    """Global registry for tool instances.

    Usage:
        ToolRegistry.register("retrieve_corpus_evidence", my_tool)
        tool = ToolRegistry.get("retrieve_corpus_evidence")
    """

    _registry: Dict[str, Tool] = {}

    @classmethod
    def register(cls, name: str, tool: Tool) -> None:
        """Register a tool under *name*."""
        cls._registry[name] = tool

    @classmethod
    def get(cls, name: str) -> Tool:
        """Retrieve a registered tool by *name*."""
        if name not in cls._registry:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry.")
        return cls._registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a tool is registered."""
        return name in cls._registry

    @classmethod
    def list_tools(cls) -> List[str]:
        """Return a sorted list of registered tool names."""
        return sorted(cls._registry.keys())

    @classmethod
    def unregister(cls, name: str) -> Optional[Tool]:
        """Remove and return a tool by *name* if registered."""
        return cls._registry.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools."""
        cls._registry.clear()
