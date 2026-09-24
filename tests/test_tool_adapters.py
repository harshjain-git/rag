"""Tests for individual tool adapters in agent/tools/."""

import unittest
from unittest.mock import patch, MagicMock
from core.models import AgentState
from agent.tools import (
    OrchestratorTool,
    RetrievalTool,
    GenerationTool,
    VerificationTool,
    QueryRewriterTool,
    QuestionGeneratorTool,
)


class TestToolAdapters(unittest.TestCase):
    """Test suite for tool adapters wrapping core functions into the Tool protocol."""

    def test_orchestrator_tool_invoke(self) -> None:
        """OrchestratorTool delegates state to execute_orchestrator."""
        tool = OrchestratorTool()
        state: AgentState = {"query": "hello", "tools_used": [], "status": "START"}

        with patch("agent.chain.execute_orchestrator") as mock_exec:
            mock_exec.return_value = {**state, "status": "DIRECT_RESPONSE", "answer": "Hello cadet!"}
            result = tool.invoke(state, {})

            mock_exec.assert_called_once_with(state)
            self.assertEqual(result["status"], "DIRECT_RESPONSE")
            self.assertEqual(result["answer"], "Hello cadet!")

    def test_retrieval_tool_invoke_state(self) -> None:
        """RetrievalTool delegates full AgentState to execute_retrieval."""
        tool = RetrievalTool()
        state: AgentState = {
            "query": "ASVAB requirements",
            "evidence": [],
            "tools_used": [],
            "status": "START",
        }

        with patch("agent.chain.execute_retrieval") as mock_exec:
            mock_exec.return_value = {
                **state,
                "evidence": [{"text": "chunk 1", "source": "s.pdf", "page": 1}],
                "tools_used": ["retrieve_corpus_evidence"],
            }
            result = tool.invoke(state, {})

            mock_exec.assert_called_once_with(state)
            self.assertEqual(len(result["evidence"]), 1)
            self.assertIn("retrieve_corpus_evidence", result["tools_used"])

    def test_retrieval_tool_invoke_dict(self) -> None:
        """RetrievalTool supports LangChain-style dict invocation."""
        tool = RetrievalTool()
        with patch("infrastructure.chroma.retrieve_evidence") as mock_ret:
            mock_ret.return_value = [{"text": "evidence chunk"}]
            result = tool.invoke({"query": "ASVAB score"})

            mock_ret.assert_called_once_with(query="ASVAB score")
            self.assertEqual(result, [{"text": "evidence chunk"}])

    def test_generation_tool_invoke(self) -> None:
        """GenerationTool delegates state to execute_generation."""
        tool = GenerationTool()
        state: AgentState = {
            "query": "ASVAB score",
            "evidence": [{"text": "score details", "source": "doc.pdf", "page": 1}],
            "tools_used": ["retrieve_corpus_evidence"],
        }

        with patch("agent.chain.execute_generation") as mock_exec:
            mock_exec.return_value = {
                **state,
                "answer": "Minimum score is 31.",
                "tools_used": ["retrieve_corpus_evidence", "generate_answer"],
            }
            result = tool.invoke(state, {})

            mock_exec.assert_called_once_with(state)
            self.assertEqual(result["answer"], "Minimum score is 31.")
            self.assertIn("generate_answer", result["tools_used"])

    def test_verification_tool_invoke_state(self) -> None:
        """VerificationTool delegates full AgentState to execute_verification."""
        tool = VerificationTool()
        state: AgentState = {
            "query": "ASVAB score",
            "raw_answer": "Score is 31.",
            "evidence": [{"text": "score is 31"}],
            "status": "GENERATION_DONE",
            "tools_used": ["generate_answer"],
        }

        with patch("agent.chain.execute_verification") as mock_exec:
            mock_exec.return_value = {
                **state,
                "is_verified": True,
                "verification_status": "VERIFIED_SUPPORTED",
                "status": "GROUNDED",
            }
            result = tool.invoke(state, {})

            mock_exec.assert_called_once_with(state)
            self.assertTrue(result["is_verified"])
            self.assertEqual(result["status"], "GROUNDED")

    def test_verification_tool_invoke_dict(self) -> None:
        """VerificationTool supports raw dict verification arguments."""
        tool = VerificationTool()
        with patch("agent.verification.verify_grounding") as mock_vg:
            mock_vg.return_value = {"is_grounded": True, "verification_status": "VERIFIED_SUPPORTED"}
            result = tool.invoke({
                "query": "test query",
                "raw_answer": "test answer",
                "evidence": [],
            })
            mock_vg.assert_called_once_with(
                query="test query",
                raw_answer="test answer",
                evidence=[],
                is_answerable=None,
            )
            self.assertTrue(result["is_grounded"])

    def test_query_rewriter_adapter(self) -> None:
        """QueryRewriterTool resolves conversational pronouns and records usage."""
        tool = QueryRewriterTool()
        state: AgentState = {
            "query": "What about its score?",
            "history": [{"role": "user", "content": "Tell me about ASVAB"}],
            "tools_used": [],
        }

        mock_tool_instance = MagicMock()
        mock_tool_instance.invoke.return_value = {
            "action": "REWRITE",
            "query": "What is the ASVAB score requirement?",
            "reason": "Resolved 'its' to ASVAB",
        }

        with patch("agent.query_rewriter_tool.get_query_rewriter_tool", return_value=mock_tool_instance):
            result = tool.invoke(state, {})

            self.assertEqual(result["resolved_query"], "What is the ASVAB score requirement?")
            self.assertEqual(result["resolution_action"], "REWRITE")
            self.assertEqual(result["resolution_reason"], "Resolved 'its' to ASVAB")
            self.assertIn("query_rewriter", result["tools_used"])

    def test_question_generator_adapter(self) -> None:
        """QuestionGeneratorTool formats quiz questions, extracts evidence, and updates state."""
        tool = QuestionGeneratorTool()
        state: AgentState = {
            "query": "Generate quiz",
            "tools_used": [],
        }

        mock_q_data = {
            "status": "SUCCESS",
            "questions": [
                {
                    "question": "What is the minimum score?",
                    "options": ["31", "50", "20", "10"],
                    "answer": "31",
                    "citations": [
                        {"quote": "Minimum is 31", "source": "guide.pdf", "page": 5, "chunk_id": "c1"}
                    ],
                }
            ],
        }

        with patch("agent.question_generator_tool.generate_corpus_questions", return_value=mock_q_data):
            result = tool.invoke(state, {"count": 1})

            self.assertEqual(len(result["questions"]), 1)
            self.assertEqual(len(result["evidence"]), 1)
            self.assertEqual(result["evidence"][0]["source"], "guide.pdf")
            self.assertTrue(result["is_grounded"])
            self.assertTrue(result["is_verified"])
            self.assertIn("generate_corpus_questions", result["tools_used"])


if __name__ == "__main__":
    unittest.main()
