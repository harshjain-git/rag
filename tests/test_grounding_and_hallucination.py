"""
End-to-End and unit tests for hallucination prevention, empty input handling,
vector similarity threshold filtering, citation anchoring, and grounding verification.
"""

import unittest
from application.services import get_agent_service
from agent.verification import verify_grounding
from infrastructure.chroma import get_retriever
import config


class TestHallucinationAndGrounding(unittest.TestCase):
    """Test suite validating anti-hallucination safeguards and edge-case query handling."""

    @classmethod
    def setUpClass(cls):
        cls.service = get_agent_service()
        cls.retriever = get_retriever()

    # 1. Empty and Whitespace User Input Handling
    def test_empty_user_input(self):
        for empty_val in ["", "   ", "\t\n", None]:
            res = self.service.query(empty_val)
            self.assertEqual(res["status"], "EMPTY_QUERY")
            self.assertEqual(len(res["evidence"]), 0)
            self.assertFalse(res["is_grounded"])
            self.assertFalse(res["is_verified"])

    # 2. Vector Threshold Filters Out-of-Corpus Queries
    def test_vector_threshold_filters_out_of_corpus(self):
        out_of_corpus_query = "How do I bake a chocolate cake at home?"
        # Raw distance must exceed threshold
        raw = self.retriever.retrieve_raw(out_of_corpus_query, top_k=1)
        self.assertTrue(len(raw) > 0)
        top_score = raw[0]["score"]
        self.assertGreater(
            top_score,
            config.SIMILARITY_THRESHOLD,
            f"Expected out-of-corpus query distance ({top_score:.4f}) to exceed threshold ({config.SIMILARITY_THRESHOLD})",
        )

        # Filtered retrieval must yield zero evidence chunks
        filtered_evidence = self.retriever.retrieve(out_of_corpus_query)
        self.assertEqual(len(filtered_evidence), 0)

    # 3. Vector Threshold Retains In-Corpus Answerable Queries
    def test_vector_threshold_retains_answerable_query(self):
        answerable_query = "What are the standard score mean and standard deviation for ASVAB test scores?"
        raw = self.retriever.retrieve_raw(answerable_query, top_k=1)
        self.assertTrue(len(raw) > 0)
        top_score = raw[0]["score"]
        self.assertLessEqual(
            top_score,
            config.SIMILARITY_THRESHOLD,
            f"Expected answerable query distance ({top_score:.4f}) to be <= threshold ({config.SIMILARITY_THRESHOLD})",
        )

        filtered = self.retriever.retrieve(answerable_query)
        self.assertGreater(len(filtered), 0)

    # 4. Out-of-Corpus Query Produces Refusal Without Citations or Hallucinations
    def test_out_of_corpus_query_refusal_and_no_citations(self):
        query = "What is the recipe for Neapolitan pizza dough?"
        res = self.service.query(query)

        # Must not be marked grounded
        self.assertFalse(res["is_grounded"])
        self.assertEqual(len(res["evidence"]), 0)
        # Content should not be empty and must not invent document citations
        self.assertTrue(len(res["content"]) > 0)
        self.assertNotIn("**Sources:**", res["content"])
        # Verification confirms safe refusal
        self.assertIn(
            res.get("verification_status", ""),
            ["REFUSAL_CONFIRMED", "VERIFIED_SUPPORTED", ""],
        )

    # 5. In-Corpus Answerable Query Returns Grounded Answer with Citations
    def test_answerable_query_grounded_with_citations(self):
        query = "What are the standard score mean and standard deviation for ASVAB test scores?"
        mock_grounded_json = {
            "is_answerable": True,
            "answer": "Scores on the ASVAB tests are reported as standard scores with a mean of 50 and a standard deviation of 10."
        }
        mock_verification_json = {
            "is_supported": True,
            "explanation": "Answer is strictly supported by asvab2.pdf excerpts."
        }

        from unittest.mock import patch
        with patch("infrastructure.gemini.GeminiLLM.generate_json", return_value=mock_grounded_json), \
             patch("infrastructure.gemini.generate_structured_json", side_effect=[mock_grounded_json, mock_verification_json]):
            res = self.service.query(query)

        self.assertTrue(res["is_grounded"])
        self.assertTrue(res["is_answerable"])
        self.assertGreater(len(res["evidence"]), 0)
        # Top chunk must come from asvab2.pdf
        top_source = res["evidence"][0]["source"]
        self.assertEqual(top_source, "asvab2.pdf")
        # Must contain answer facts (mean 50, sd 10)
        content_lower = res["content"].lower()
        self.assertTrue("50" in content_lower and "10" in content_lower)
        # Must contain citation block and source reference
        self.assertIn("**Sources:**", res["content"])
        self.assertIn("asvab2.pdf", res["content"])
        # Verification audit should pass
        self.assertTrue(res["is_verified"])
        self.assertEqual(res["verification_status"], "VERIFIED_SUPPORTED")

    # 6. Direct Verification Node Test for Unsupported Hallucination
    def test_verify_grounding_detects_unsupported_hallucination(self):
        query = "What is the ASVAB passing score?"
        mock_evidence = [
            {
                "text": "Psychological counseling is provided to personnel experiencing stress in combat training.",
                "source": "military_psyc1.pdf",
                "page": 5,
            }
        ]
        fabricated_answer = "The passing score for ASVAB is 95 points, which guarantees promotion to Captain."

        from unittest.mock import patch
        with patch("infrastructure.gemini.generate_structured_json", return_value={"is_supported": False, "explanation": "Evidence does not mention passing scores or promotion."}):
            verification = verify_grounding(
                query=query,
                raw_answer=fabricated_answer,
                evidence=mock_evidence,
                is_answerable=True,
            )

        self.assertFalse(verification["is_verified"])
        self.assertEqual(verification["verification_status"], "UNSUPPORTED_HALLUCINATION")


if __name__ == "__main__":
    unittest.main()
