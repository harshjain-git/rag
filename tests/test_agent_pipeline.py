"""Tests for agent.pipeline.AgentPipeline execution and routing flows."""

import unittest
from typing import Dict, Any
from core.models import AgentState
from core.registry import ToolRegistry
from agent.pipeline import AgentPipeline


class MockOrchestratorTool:
    """Mock orchestrator tool."""
    def __init__(self, action: str = "PROCEED_RAG", direct_answer: str = "") -> None:
        self.action = action
        self.direct_answer = direct_answer
        self.invoked = False

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        self.invoked = True
        state["tools_used"].append("orchestrator")
        if self.action == "DIRECT_RESPONSE":
            state["status"] = "DIRECT_RESPONSE"
            state["answer"] = self.direct_answer
        elif self.action == "QUESTIONS":
            state["questions"] = [{"question": "Q1?", "options": ["A", "B"], "answer": "A"}]
        return state


class MockRetrievalTool:
    """Mock retrieval tool."""
    def __init__(self) -> None:
        self.invoked = False

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        self.invoked = True
        state["tools_used"].append("retrieve_corpus_evidence")
        state["evidence"] = [
            {"text": "ASVAB stands for Armed Services Vocational Aptitude Battery.", "source": "asvab.pdf", "page": 1}
        ]
        state["top_score"] = 0.85
        return state


class MockGenerationTool:
    """Mock generation tool."""
    def __init__(self) -> None:
        self.invoked = False

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        self.invoked = True
        state["tools_used"].append("generate_answer")
        state["answer"] = "ASVAB is an aptitude battery."
        state["is_grounded"] = True
        return state


class MockVerificationTool:
    """Mock verification tool."""
    def __init__(self, verified: bool = True) -> None:
        self.verified = verified
        self.invoked = False

    def invoke(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        self.invoked = True
        state["tools_used"].append("verify_grounding")
        state["is_verified"] = self.verified
        state["verification_status"] = "VERIFIED_SUPPORTED" if self.verified else "UNGROUNDED"
        state["status"] = "GROUNDED" if self.verified else "UNVERIFIED"
        return state


class CustomRegistry:
    """Isolated mock registry for testing dependency injection."""
    def __init__(self, tools: Dict[str, Any]) -> None:
        self.tools = tools

    def get(self, name: str) -> Any:
        return self.tools[name]


class TestAgentPipeline(unittest.TestCase):
    """Test suite for AgentPipeline execution logic."""

    def test_empty_query_returns_early(self) -> None:
        """Pipeline returns status EMPTY_QUERY immediately for blank or whitespace queries."""
        pipeline = AgentPipeline()
        for empty_val in ["", "   ", "\t\n", None]:
            result = pipeline.run(empty_val)
            self.assertEqual(result.get("status"), "EMPTY_QUERY")
            self.assertEqual(result.get("evidence"), [])
            self.assertEqual(result.get("tools_used"), [])

    def test_standard_workflow_success(self) -> None:
        """Pipeline runs orchestrator -> retrieve -> generate -> verify in sequence."""
        orch = MockOrchestratorTool(action="PROCEED_RAG")
        ret = MockRetrievalTool()
        gen = MockGenerationTool()
        ver = MockVerificationTool(verified=True)

        custom_reg = CustomRegistry({
            "orchestrator": orch,
            "retrieve_corpus_evidence": ret,
            "generate_answer": gen,
            "verify_grounding": ver,
        })

        pipeline = AgentPipeline(registry=custom_reg)
        state = pipeline.run("What is ASVAB?")

        self.assertTrue(orch.invoked)
        self.assertTrue(ret.invoked)
        self.assertTrue(gen.invoked)
        self.assertTrue(ver.invoked)

        self.assertEqual(
            state["tools_used"],
            ["orchestrator", "retrieve_corpus_evidence", "generate_answer", "verify_grounding"],
        )
        self.assertEqual(state["status"], "GROUNDED")
        self.assertTrue(state["is_verified"])
        self.assertEqual(len(state["evidence"]), 1)
        self.assertIn("ASVAB is an aptitude battery.", state["answer"])

    def test_direct_response_short_circuits_workflow(self) -> None:
        """When orchestrator resolves directly (e.g. greeting), standard workflow is skipped."""
        orch = MockOrchestratorTool(action="DIRECT_RESPONSE", direct_answer="Hello Cadet! How can I assist you today?")
        ret = MockRetrievalTool()
        gen = MockGenerationTool()
        ver = MockVerificationTool()

        custom_reg = CustomRegistry({
            "orchestrator": orch,
            "retrieve_corpus_evidence": ret,
            "generate_answer": gen,
            "verify_grounding": ver,
        })

        pipeline = AgentPipeline(registry=custom_reg)
        state = pipeline.run("Hello")

        self.assertTrue(orch.invoked)
        self.assertFalse(ret.invoked)
        self.assertFalse(gen.invoked)
        self.assertFalse(ver.invoked)

        self.assertEqual(state["status"], "DIRECT_RESPONSE")
        self.assertEqual(state["answer"], "Hello Cadet! How can I assist you today?")
        self.assertEqual(state["tools_used"], ["orchestrator"])

    def test_quiz_questions_short_circuits_workflow(self) -> None:
        """When orchestrator generates questions, subsequent RAG workflow tools are skipped."""
        orch = MockOrchestratorTool(action="QUESTIONS")
        ret = MockRetrievalTool()
        gen = MockGenerationTool()
        ver = MockVerificationTool()

        custom_reg = CustomRegistry({
            "orchestrator": orch,
            "retrieve_corpus_evidence": ret,
            "generate_answer": gen,
            "verify_grounding": ver,
        })

        pipeline = AgentPipeline(registry=custom_reg)
        state = pipeline.run("Generate 5 quiz questions on navigation")

        self.assertTrue(orch.invoked)
        self.assertFalse(ret.invoked)
        self.assertFalse(gen.invoked)
        self.assertFalse(ver.invoked)
        self.assertIsNotNone(state.get("questions"))
        self.assertEqual(len(state["questions"]), 1)

    def test_invoke_matches_run(self) -> None:
        """Pipeline invoke() compatibility wrapper behaves identically to run()."""
        pipeline = AgentPipeline()
        result = pipeline.invoke({"query": "", "history": []})
        self.assertEqual(result.get("status"), "EMPTY_QUERY")


if __name__ == "__main__":
    unittest.main()
