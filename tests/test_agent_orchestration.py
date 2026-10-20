"""
Unit tests for Agent Orchestration, ToolRegistry, AgentPipeline execution,
terminal early-exit handling, and AgentService formatting.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.models import AgentState
from core.registry import ToolRegistry
from agent.pipeline import AgentPipeline
from application.services import AgentService


class TestAgentOrchestration(unittest.TestCase):
    """Test suite verifying pipeline orchestration, tool dispatch, and registry behavior."""

    def setUp(self):
        # Save a copy of registry state
        self._original_registry = dict(ToolRegistry._registry)

    def tearDown(self):
        # Restore registry state
        ToolRegistry._registry = dict(self._original_registry)

    # 1. ToolRegistry Registration and Lookup
    def test_tool_registry_lifecycle(self):
        def dummy_tool(state: AgentState) -> AgentState:
            return {**state, "status": "DUMMY_DONE"}

        ToolRegistry.register("test_dummy", dummy_tool)
        self.assertTrue(ToolRegistry.is_registered("test_dummy"))
        self.assertIn("test_dummy", ToolRegistry.list_tools())
        retrieved = ToolRegistry.get("test_dummy")
        self.assertEqual(retrieved({"query": "hi"}), {"query": "hi", "status": "DUMMY_DONE"})

        ToolRegistry.unregister("test_dummy")
        self.assertFalse(ToolRegistry.is_registered("test_dummy"))
        with self.assertRaises(KeyError):
            ToolRegistry.get("test_dummy")

    # 2. Empty and Whitespace Query Handling
    def test_pipeline_empty_query(self):
        pipeline = AgentPipeline()
        for empty_val in ["", "   ", None]:
            state = pipeline.run(query=empty_val)
            self.assertEqual(state["status"], "EMPTY_QUERY")
            self.assertEqual(len(state["evidence"]), 0)

    # 3. Terminal Early Exit (Direct Response without Retrieval)
    def test_pipeline_terminal_direct_response(self):
        def mock_orchestrator(state: AgentState) -> AgentState:
            return {
                **state,
                "answer": "Hello! How can I assist you with cadet readiness today?",
                "status": "DIRECT_RESPONSE",
                "is_grounded": False,
            }

        retrieval_called = False
        def mock_retrieval(state: AgentState) -> AgentState:
            nonlocal retrieval_called
            retrieval_called = True
            return state

        ToolRegistry.register("orchestrator", mock_orchestrator)
        ToolRegistry.register("retrieve_corpus_evidence", mock_retrieval)

        pipeline = AgentPipeline()
        result = pipeline.run("Hello")

        self.assertEqual(result["status"], "DIRECT_RESPONSE")
        self.assertIn("Hello!", result["answer"])
        self.assertFalse(retrieval_called, "Retrieval must not be invoked for terminal direct responses.")

    # 4. End-to-End Standard Workflow Dispatch (Retrieve -> Generate -> Verify)
    def test_pipeline_standard_workflow_execution(self):
        execution_order = []

        def mock_orch(state: AgentState) -> AgentState:
            execution_order.append("orchestrator")
            return {**state, "status": "QUERY_RESOLVED"}

        def mock_ret(state: AgentState) -> AgentState:
            execution_order.append("retrieval")
            return {
                **state,
                "evidence": [{"text": "Sample chunk", "source": "asvab1.pdf", "page": 1, "score": 0.15}],
                "status": "EVIDENCE_RETRIEVED",
            }

        def mock_gen(state: AgentState) -> AgentState:
            execution_order.append("generation")
            return {
                **state,
                "answer": "Grounded answer text.",
                "raw_answer": "Grounded answer text.",
                "is_grounded": True,
                "is_answerable": True,
                "status": "SUCCESS",
            }

        def mock_ver(state: AgentState) -> AgentState:
            execution_order.append("verification")
            return {
                **state,
                "is_verified": True,
                "verification_status": "VERIFIED_SUPPORTED",
            }

        ToolRegistry.register("orchestrator", mock_orch)
        ToolRegistry.register("retrieve_corpus_evidence", mock_ret)
        ToolRegistry.register("generate_answer", mock_gen)
        ToolRegistry.register("verify_grounding", mock_ver)

        pipeline = AgentPipeline()
        state = pipeline.run("Explain ASVAB scoring")

        self.assertEqual(
            execution_order,
            ["orchestrator", "retrieval", "generation", "verification"],
            "Pipeline must execute steps in strict sequential order.",
        )
        self.assertEqual(state["answer"], "Grounded answer text.")
        self.assertTrue(state["is_grounded"])
        self.assertTrue(state["is_verified"])

    # 5. AgentService Facade Output Formatting
    def test_agent_service_message_formatting(self):
        mock_pipeline = MagicMock(spec=AgentPipeline)
        mock_pipeline.run.return_value = {
            "query": "What is AFQT?",
            "resolved_query": "AFQT components",
            "resolution_action": "REWRITE",
            "resolution_reason": "Context expanded",
            "answer": "AFQT is derived from four ASVAB subtests.",
            "is_grounded": True,
            "is_answerable": True,
            "is_verified": True,
            "verification_status": "VERIFIED_SUPPORTED",
            "verification_details": "Supported by asvab1.pdf",
            "status": "SUCCESS",
            "tools_used": ["retrieve_corpus_evidence"],
            "top_score": 0.145,
            "evidence": [{"source": "asvab1.pdf", "page": 2}],
            "questions": [],
            "generation_metadata": {"model": "gemini-3.8-flash"},
        }

        service = AgentService(pipeline=mock_pipeline)
        msg = service.query("What is AFQT?")

        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "AFQT is derived from four ASVAB subtests.")
        self.assertEqual(msg["resolved_query"], "AFQT components")
        self.assertEqual(msg["resolution_action"], "REWRITE")
        self.assertTrue(msg["is_grounded"])
        self.assertTrue(msg["is_verified"])
        self.assertEqual(msg["verification_status"], "VERIFIED_SUPPORTED")
        self.assertEqual(len(msg["evidence"]), 1)
        self.assertEqual(msg["top_score"], 0.145)


if __name__ == "__main__":
    unittest.main()
