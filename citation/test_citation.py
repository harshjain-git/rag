"""
Phase 10 Citation Generation Engine Verification Test Script.
Tests automatic citation extraction, formatting, and attachment across queries.
"""

import os
import sys
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from generation.generator import generate_answer
from citation.citation_engine import extract_citations, format_citations_block


def run_citation_tests():
    print("=" * 80)
    print("PHASE 10: CITATION GENERATION ENGINE TEST")
    print("Required Format: 'Document_Name.pdf — Page X'")
    print("=" * 80)

    # Live End-to-End Query Verification with Real ChromaDB & Gemini Generation
    test_queries = [
        {
            "category": "In-Corpus Query (AFQT Subtests)",
            "query": "Which subtests make up the Armed Forces Qualification Test (AFQT) composite score?",
            "expect_citations": True
        },
        {
            "category": "Out-of-Corpus Query (General Knowledge)",
            "query": "What is the capital city of France and its population?",
            "expect_citations": False
        }
    ]

    for idx, test in enumerate(test_queries, 1):
        q_text = test["query"]
        category = test["category"]
        expect_cit = test["expect_citations"]

        print(f"\n[INTEGRATION TEST {idx}] {category}")
        print(f"  • Query: \"{q_text}\"")

        res = generate_answer(q_text)
        citations = res.get("citations", [])

        print(f"  • Status: {res['status']}")
        print(f"  • Citations Extracted ({len(citations)}): {[c['citation'] for c in citations]}")
        print(f"  • Full Generated Answer with Citations:\n{res['answer']}")
        print("-" * 80)

        if expect_cit:
            assert len(citations) > 0, "Expected citations for grounded query!"
            assert "Sources:" in res["answer"], "Sources block missing from answer!"
        else:
            assert len(citations) == 0, "Out-of-corpus query should have 0 citations!"

    print("\n✅ Phase 10 Verification Complete: Citation Generation Engine fully operational.")


if __name__ == "__main__":
    run_citation_tests()
