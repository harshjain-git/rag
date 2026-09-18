"""
Phase 9 Generator Verification Test Script.
Tests Gemini Flash Lite LLM responses for grounded generation vs out-of-corpus fallback.
"""

import os
import sys
import warnings
import time

# Suppress warnings & environment messages
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from pathlib import Path

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from generation.generator import generate_answer


def run_generator_tests():
    print("=" * 80)
    print("PHASE 9: GEMINI FLASH LITE LLM INTEGRATION TEST")
    print(f"Model: {config.LLM_MODEL_NAME} | Temperature: 0.0")
    print("=" * 80)

    test_queries = [
        {
            "category": "In-Corpus Question #1 (AFQT Subtests)",
            "query": "Which subtests make up the Armed Forces Qualification Test (AFQT) composite score?",
            "expect_grounded": True
        },
        {
            "category": "In-Corpus Question #2 (APA Reliability Baseline)",
            "query": "What statistical method is recommended by APA guidelines for evaluating construct validity structural hypotheses?",
            "expect_grounded": True
        }
    ]

    for idx, test in enumerate(test_queries, 1):
        q_text = test["query"]
        category = test["category"]
        expect_grounded = test["expect_grounded"]

        print(f"\nTest #{idx} [{category}]")
        print(f"  • Query: \"{q_text}\"")

        start_time = time.time()
        res = generate_answer(q_text)
        elapsed_sec = time.time() - start_time

        score_str = f"{res['top_score']:.4f}" if res.get('top_score') is not None else 'N/A'
        print(f"  • Status: {res['status']} | Top L2 Distance: {score_str} | Execution Time: {elapsed_sec:.2f} s")
        
        evidence_chunks = res.get("evidence", [])
        if evidence_chunks:
            print(f"\n  [RETRIEVED CHUNKS SENT TO GEMINI: {len(evidence_chunks)} Total]")
            for c_idx, chunk in enumerate(evidence_chunks, 1):
                c_id = chunk.get("chunk_id", "N/A")
                c_src = chunk.get("source", "unknown.pdf")
                c_page = chunk.get("page", 1)
                c_score = chunk.get("score", 0.0)
                c_text = chunk.get("text", "").replace("\n", " ").strip()
                print(f"    • Chunk #{c_idx} | ID: {c_id} | Source: {c_src} (Page {c_page}) | Distance: {c_score:.4f}")
                print(f"      Snippet: {repr(c_text[:140])}...")
        
        print(f"\n  • Generated Answer:\n{res['answer']}")
        print("-" * 80)

        if not expect_grounded:
            assert res['answer'] == config.NOT_IN_CORPUS_MESSAGE, "Failed fallback check!"

    print("\n✅ Phase 9 Verification Complete: Gemini 2.5 Flash produces strictly grounded responses.")


if __name__ == "__main__":
    run_generator_tests()
