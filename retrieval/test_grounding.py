"""
Phase 8 Grounding & "Not in Corpus" Logic Test Script.
Verifies distance threshold filtering and fallback messaging on in-corpus vs out-of-corpus queries.
"""

import sys
import time
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retriever import evaluate_grounding


def run_grounding_tests():
    print("=" * 80)
    print("PHASE 8: GROUNDING & 'NOT IN CORPUS' LOGIC VERIFICATION")
    print(f"Similarity Threshold (Max L2 Distance): {config.SIMILARITY_THRESHOLD}")
    print("=" * 80)

    test_queries = [
        {
            "category": "In-Corpus (ASVAB / Aptitude)",
            "query": "How do standard scores in ASVAB relate to percentile ranks?",
            "expected_grounded": True
        },
        {
            "category": "In-Corpus (Military Psychology / Resilience)",
            "query": "What is psychological resilience in military personnel?",
            "expected_grounded": True
        },
        {
            "category": "In-Corpus (APA / Validity & Reliability)",
            "query": "What are test validity and reliability standards in psychological assessment?",
            "expected_grounded": True
        },
        {
            "category": "Out-of-Corpus (Unrelated General Knowledge)",
            "query": "What is the capital city of France and its population?",
            "expected_grounded": False
        },
        {
            "category": "Out-of-Corpus (Irrelevant Sports Query)",
            "query": "Who won the 2022 FIFA World Cup?",
            "expected_grounded": False
        },
        {
            "category": "Out-of-Corpus (Quantum Physics Query)",
            "query": "Explain quantum entanglement in quantum thermodynamics.",
            "expected_grounded": False
        }
    ]

    passed_count = 0

    for idx, test in enumerate(test_queries, 1):
        q_text = test["query"]
        category = test["category"]
        expected = test["expected_grounded"]

        start_time = time.time()
        res = evaluate_grounding(q_text)
        elapsed_ms = (time.time() - start_time) * 1000

        is_grounded = res["is_grounded"]
        status_pass = (is_grounded == expected)
        if status_pass:
            passed_count += 1

        print(f"\nTest #{idx} [{category}]")
        print(f"  • Query: \"{q_text}\"")
        print(f"  • Status: {res['status']} (Grounded={is_grounded}) | Execution Time: {elapsed_ms:.2f} ms")
        print()

        if is_grounded:
            best_chunk = res["evidence"][0]
            print(f"  • Top Chunk Match: Source={best_chunk['source']} | Page={best_chunk['page']} | L2 Distance={res['top_score']:.4f}")
            print(f"  • Content Snippet: {repr(best_chunk['text'][:120])}...")
        else:
            print(f"  • Output Message: \"{res['fallback_message']}\"")
            print(f"  • Grounding Check: Zero valid chunks within distance threshold {config.SIMILARITY_THRESHOLD}")
            print(f"  • Nearest Chunk L2 Distance: {res['top_score']:.4f} (Exceeds threshold 0.85)")

        print(f"  • Result: {'[PASSED]' if status_pass else '[FAILED]'}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"GROUNDING EVALUATION TEST SUMMARY: {passed_count}/{len(test_queries)} PASSED")
    print("=" * 80)

    # Verify exact required message string
    assert config.NOT_IN_CORPUS_MESSAGE == "Not in corpus — the provided documents do not contain enough information to answer this question.", "Fallback message mismatch!"
    print("Verified: Fallback message string strictly matches required specification.")


if __name__ == "__main__":
    run_grounding_tests()
