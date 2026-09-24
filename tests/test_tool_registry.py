"""Tests for core.registry.ToolRegistry and Tool protocol conformance."""

import unittest
from typing import Dict, Any
from core.models import AgentState
from core.ports import Tool
from core.registry import ToolRegistry
from core.bootstrap import bootstrap_tools


class DummyTool:
    """Mock tool conforming to the Tool protocol."""
    name = "dummy_tool"

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        state["status"] = "DUMMY_INVOKED"
        return state


class InvalidTool:
    """Class missing the invoke method, thus not conforming to Tool protocol."""
    name = "invalid_tool"


class TestToolRegistry(unittest.TestCase):
    """Test suite for ToolRegistry singleton and lifecycle."""

    def setUp(self) -> None:
        """Ensure fresh bootstrap state for each test."""
        ToolRegistry.clear()
        bootstrap_tools()

    def tearDown(self) -> None:
        """Reset registry after each test."""
        ToolRegistry.clear()
        bootstrap_tools()

    def test_default_tools_registered(self) -> None:
        """Verify that default application tools are registered upon bootstrap."""
        expected_tools = [
            "generate_answer",
            "generate_corpus_questions",
            "orchestrator",
            "query_rewriter",
            "retrieve_corpus_evidence",
            "verification",
            "verify_grounding",
        ]
        registered = ToolRegistry.list_tools()
        for tool_name in expected_tools:
            self.assertIn(tool_name, registered)
            self.assertTrue(ToolRegistry.is_registered(tool_name))

    def test_register_and_get(self) -> None:
        """Verify registering a new tool and retrieving it."""
        dummy = DummyTool()
        ToolRegistry.register("custom_dummy", dummy)

        retrieved = ToolRegistry.get("custom_dummy")
        self.assertIs(retrieved, dummy)
        self.assertTrue(ToolRegistry.is_registered("custom_dummy"))

    def test_get_nonexistent_tool_raises_key_error(self) -> None:
        """Verify getting an unregistered tool raises KeyError."""
        with self.assertRaises(KeyError):
            ToolRegistry.get("non_existent_tool_xyz")

    def test_unregister_tool(self) -> None:
        """Verify unregistering a tool removes it from registry."""
        dummy = DummyTool()
        ToolRegistry.register("temp_tool", dummy)
        self.assertTrue(ToolRegistry.is_registered("temp_tool"))

        unregistered = ToolRegistry.unregister("temp_tool")
        self.assertIs(unregistered, dummy)
        self.assertFalse(ToolRegistry.is_registered("temp_tool"))

        # Unregistering non-existent tool returns None
        self.assertIsNone(ToolRegistry.unregister("temp_tool"))

    def test_clear_registry(self) -> None:
        """Verify clear empties the entire registry."""
        ToolRegistry.clear()
        self.assertEqual(len(ToolRegistry.list_tools()), 0)

    def test_tool_protocol_conformance(self) -> None:
        """Verify runtime protocol conformance check with isinstance."""
        dummy = DummyTool()
        self.assertIsInstance(dummy, Tool)

        invalid = InvalidTool()
        self.assertNotIsInstance(invalid, Tool)

    def test_registered_tools_conform_to_protocol(self) -> None:
        """Verify all bootstrapped tools implement the Tool protocol."""
        for tool_name in ToolRegistry.list_tools():
            tool = ToolRegistry.get(tool_name)
            self.assertIsInstance(tool, Tool, f"Tool '{tool_name}' must conform to Tool protocol")


if __name__ == "__main__":
    unittest.main()
