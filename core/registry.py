"""Central registry that maps tool names directly to callable functions."""

from typing import Dict, List, Optional, Callable, Any


class ToolRegistry:
    """Global registry for tool functions and callables.

    Usage:
        ToolRegistry.register("retrieve_corpus_evidence", execute_retrieval)
        func = ToolRegistry.get("retrieve_corpus_evidence")
        result = func(state)
    """

    _registry: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, tool: Callable[..., Any]) -> None:
        """Register a callable tool under *name*."""
        cls._registry[name] = tool

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Retrieve a registered tool callable by *name*."""
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
    def unregister(cls, name: str) -> Optional[Callable[..., Any]]:
        """Remove and return a tool by *name* if registered."""
        return cls._registry.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools."""
        cls._registry.clear()

