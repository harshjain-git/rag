# Tool Registry for the application

"""Central registry that maps tool names to concrete tool implementations.

All tools must conform to the ``Tool`` protocol defined in ``core.ports.tool``.
The registry is a simple singleton class with class‑level storage.
"""

from typing import Dict
from core.ports.tool import Tool


class ToolRegistry:
    """Global registry for tool instances.

    Usage:
        ToolRegistry.register("retrieve_corpus_evidence", my_tool)
        tool = ToolRegistry.get("retrieve_corpus_evidence")
    """

    _registry: Dict[str, Tool] = {}

    @classmethod
    def register(cls, name: str, tool: Tool) -> None:
        """Register a tool under *name*.

        If a tool with the same name already exists it will be overwritten.
        """
        cls._registry[name] = tool

    @classmethod
    def get(cls, name: str) -> Tool:
        """Retrieve a registered tool by *name*.

        Raises ``KeyError`` if the name is not present.
        """
        return cls._registry[name]
