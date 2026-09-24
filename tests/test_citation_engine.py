"""Tests for citation.citation_engine module."""

import unittest
from citation.citation_engine import (
    extract_citations,
    format_citations_block,
    attach_citations,
)


class TestCitationEngine(unittest.TestCase):
    """Test suite for citation extraction, sorting, formatting, and attachment."""

    def test_extract_citations_empty(self) -> None:
        """Empty evidence list returns empty citation list."""
        self.assertEqual(extract_citations([]), [])

    def test_extract_citations_deduplication_and_sorting(self) -> None:
        """Citations are deduplicated and sorted by source filename then page number."""
        chunks = [
            {"source": "navy_guide.pdf", "page": 10, "text": "chunk 1"},
            {"source": "army_guide.pdf", "page": 5, "text": "chunk 2"},
            {"source": "navy_guide.pdf", "page": 2, "text": "chunk 3"},
            {"source": "army_guide.pdf", "page": 5, "text": "chunk 4 (duplicate)"},
        ]
        citations = extract_citations(chunks)

        self.assertEqual(len(citations), 3)
        self.assertEqual(citations[0]["source"], "army_guide.pdf")
        self.assertEqual(citations[0]["page"], 5)
        self.assertEqual(citations[0]["citation"], "army_guide.pdf — Page 5")

        self.assertEqual(citations[1]["source"], "navy_guide.pdf")
        self.assertEqual(citations[1]["page"], 2)

        self.assertEqual(citations[2]["source"], "navy_guide.pdf")
        self.assertEqual(citations[2]["page"], 10)

    def test_format_citations_block(self) -> None:
        """Formats citation list into clean markdown block."""
        self.assertEqual(format_citations_block([]), "")

        citations = [
            {"source": "docA.pdf", "page": 1, "citation": "docA.pdf — Page 1"},
            {"source": "docB.pdf", "page": 3, "citation": "docB.pdf — Page 3"},
        ]
        block = format_citations_block(citations)
        self.assertIn("**Sources:**", block)
        self.assertIn("• docA.pdf — Page 1", block)
        self.assertIn("• docB.pdf — Page 3", block)

    def test_attach_citations_grounded(self) -> None:
        """Appends formatted citations when is_grounded is True."""
        answer = "Enlistment requirements include passing the physical test."
        chunks = [{"source": "standards.pdf", "page": 12, "text": "physical fitness"}]

        result = attach_citations(answer, chunks, is_grounded=True)
        self.assertTrue(result.startswith(answer))
        self.assertIn("**Sources:**", result)
        self.assertIn("standards.pdf — Page 12", result)

    def test_attach_citations_ungrounded(self) -> None:
        """Does not append citations when is_grounded is False."""
        answer = "I could not find information regarding this question in the official materials."
        chunks = [{"source": "standards.pdf", "page": 12, "text": "physical fitness"}]

        result = attach_citations(answer, chunks, is_grounded=False)
        self.assertEqual(result, answer)
        self.assertNotIn("**Sources:**", result)


if __name__ == "__main__":
    unittest.main()
