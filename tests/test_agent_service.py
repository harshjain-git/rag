"""Tests for application.services.AgentService facade and UI payload formatting."""

import unittest
from unittest.mock import MagicMock
from core.models import AgentState
from application.services import AgentService, get_agent_service


class TestAgentService(unittest.TestCase):
    """Test suite for AgentService."""

    def test_query_formats_standard_response(self) -> None:
        """Service formats full pipeline state into a standardized UI message dict."""
        mock_pipeline = MagicMock()
        mock_state: AgentState = {
            "query": "What is ASVAB?",
            "resolved_query": "What is Armed Services Vocational Aptitude Battery?",
            "resolution_action": "REWRITE",
            "resolution_reason": "Expanded abbreviation",
            "history": [],
            "evidence": [
                {"text": "Sample text", "source": "guide.pdf", "page": 2, "score": 0.9}
            ],
            "answer": "ASVAB is a standardized military test. [guide.pdf — Page 2]",
            "is_grounded": True,
            "is_answerable": True,
            "is_verified": True,
            "verification_status": "VERIFIED_SUPPORTED",
            "verification_details": "Directly supported by guide.pdf page 2",
            "tools_used": ["orchestrator", "retrieve_corpus_evidence", "generate_answer", "verify_grounding"],
            "top_score": 0.9,
            "status": "GROUNDED",
            "questions": None,
            "generation_metadata": {"model": "gemini-2.5-flash"},
        }
        mock_pipeline.run.return_value = mock_state

        service = AgentService(pipeline=mock_pipeline)
        result = service.query("What is ASVAB?")

        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], mock_state["answer"])
        self.assertEqual(result["original_query"], "What is ASVAB?")
        self.assertEqual(result["resolved_query"], "What is Armed Services Vocational Aptitude Battery?")
        self.assertEqual(result["resolution_action"], "REWRITE")
        self.assertEqual(result["resolution_reason"], "Expanded abbreviation")
        self.assertTrue(result["is_grounded"])
        self.assertTrue(result["is_answerable"])
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["verification_status"], "VERIFIED_SUPPORTED")
        self.assertEqual(result["status"], "GROUNDED")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["tools_used"], ["orchestrator", "retrieve_corpus_evidence", "generate_answer", "verify_grounding"])
        self.assertEqual(result["questions"], [])
        self.assertEqual(result["generation_metadata"], {"model": "gemini-2.5-flash"})

    def test_null_safety_and_defaults(self) -> None:
        """Service provides safe defaults when pipeline returns sparse/empty state."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "query": "hello",
            "status": "EMPTY_QUERY",
        }

        service = AgentService(pipeline=mock_pipeline)
        result = service.query("hello")

        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "")
        self.assertFalse(result["is_grounded"])
        self.assertFalse(result["is_answerable"])
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["status"], "EMPTY_QUERY")
        self.assertEqual(result["tools_used"], [])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["questions"], [])
        self.assertEqual(result["generation_metadata"], {})

    def test_quiz_questions_payload(self) -> None:
        """Service correctly serializes quiz questions when generated."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "query": "quiz",
            "status": "QUESTIONS_GENERATED",
            "questions": [
                {"question": "Q1?", "options": ["A", "B", "C", "D"], "answer": "A"}
            ],
            "answer": "Here are 1 quiz questions:",
        }

        service = AgentService(pipeline=mock_pipeline)
        result = service.query("quiz")

        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["question"], "Q1?")
        self.assertEqual(result["content"], "Here are 1 quiz questions:")

    def test_singleton_retrieval(self) -> None:
        """get_agent_service returns a singleton instance."""
        s1 = get_agent_service()
        s2 = get_agent_service()
        self.assertIs(s1, s2)
        self.assertIsInstance(s1, AgentService)


if __name__ == "__main__":
    unittest.main()
