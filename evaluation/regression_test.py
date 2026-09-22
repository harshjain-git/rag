"""
Step 7 — Comprehensive Regression Testing Suite for Cadet Readiness Advisor.
Performs side-by-side evaluation comparing the original direct RAG pipeline
and the new LangChain Agentic pipeline across the benchmark dataset.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from generation.generator import generate_answer as original_generate_answer
from agent import get_agent_manager


def run_regression_suite():
    print("=" * 78)
    print("STEP 7: COMPREHENSIVE REGRESSION TESTING (Original RAG vs LangChain Agent)")
    print("=" * 78)

    eval_file = config.BASE_DIR / "evaluation" / "evaluation_set.json"
    if not eval_file.exists():
        print(f"❌ Error: Evaluation dataset not found at {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} Benchmark Evaluation Questions across 3 test categories:\n"
          f"  • Answerable (In-Corpus)\n"
          f"  • Unanswerable (Out-of-Corpus Refusals)\n"
          f"  • Comparison (Multi-Document Synthesis)\n")

    agent_mgr = get_agent_manager()

    total_count = len(questions)
    retrieval_matches = 0
    grounding_matches = 0
    verification_passed = 0
    refusal_correct = 0
    in_corpus_count = 0
    out_corpus_count = 0

    results_table = []

    print("-" * 78)
    print(f"{'ID':<4} | {'Type':<12} | {'Retrieval':<10} | {'Grounding':<10} | {'Verified':<10} | {'Status':<12}")
    print("-" * 78)

    for q in questions:
        q_id = q["id"]
        q_type = q["type"]
        query_text = q["question"]

        is_unanswerable = (q_type == "unanswerable")
        if is_unanswerable:
            out_corpus_count += 1
        else:
            in_corpus_count += 1

        # 1. Run Original Direct Pipeline
        t0 = time.time()
        orig_res = original_generate_answer(query_text)
        orig_lat = time.time() - t0

        # 2. Run LangChain Agent Pipeline
        t1 = time.time()
        agent_res = agent_mgr.run(query_text)
        agent_lat = time.time() - t1

        # 3. Parity Checks
        # A. Retrieval parity
        orig_evidence = orig_res.get("evidence", [])
        agent_evidence = agent_res.get("evidence", [])
        retrieval_ok = (len(orig_evidence) == len(agent_evidence))
        if retrieval_ok and orig_evidence:
            retrieval_ok = all(
                o["chunk_id"] == a["chunk_id"] and abs(o["score"] - a["score"]) < 1e-4
                for o, a in zip(orig_evidence, agent_evidence)
            )
        if retrieval_ok:
            retrieval_matches += 1

        # B. Grounding decision parity
        grounding_ok = (orig_res.get("is_grounded") == agent_res.get("is_grounded"))
        if grounding_ok:
            grounding_matches += 1

        # C. Refusal check for unanswerable
        if is_unanswerable:
            if not agent_res.get("is_grounded") and len(agent_res.get("citations", [])) == 0:
                refusal_correct += 1

        # D. Verification status
        verified_ok = bool(agent_res.get("is_verified", False))
        if verified_ok:
            verification_passed += 1

        row_status = "✅ PASS" if (retrieval_ok and grounding_ok and verified_ok) else "⚠️ REVIEW"
        
        print(f"{q_id:<4} | {q_type:<12} | {'MATCH' if retrieval_ok else 'DIFF':<10} | "
              f"{'MATCH' if grounding_ok else 'DIFF':<10} | {'YES' if verified_ok else 'NO':<10} | {row_status:<12}")

        results_table.append({
            "id": q_id,
            "type": q_type,
            "question": query_text,
            "retrieval_ok": retrieval_ok,
            "grounding_ok": grounding_ok,
            "verified_ok": verified_ok,
            "agent_status": agent_res.get("status"),
            "verification_status": agent_res.get("verification_status"),
            "agent_answer": agent_res.get("answer", "")[:120]
        })

    # Summary Report
    retrieval_rate = (retrieval_matches / total_count) * 100
    grounding_rate = (grounding_matches / total_count) * 100
    verification_rate = (verification_passed / total_count) * 100
    refusal_rate = (refusal_correct / out_corpus_count * 100) if out_corpus_count > 0 else 100.0

    print("=" * 78)
    print("REGRESSION TESTING SUMMARY METRICS")
    print("=" * 78)
    print(f"  • Total Benchmark Queries Tested:       {total_count}")
    print(f"  • Retrieval Parity Rate:                 {retrieval_rate:.1f}% ({retrieval_matches}/{total_count})")
    print(f"  • Grounding Classification Parity:       {grounding_rate:.1f}% ({grounding_matches}/{total_count})")
    print(f"  • Zero-Hallucination Refusal Accuracy:   {refusal_rate:.1f}% ({refusal_correct}/{out_corpus_count})")
    print(f"  • Fact Verification Pass Rate:           {verification_rate:.1f}% ({verification_passed}/{total_count})")
    print("=" * 78)

    if retrieval_rate == 100.0 and grounding_rate == 100.0 and refusal_rate == 100.0:
        print("🎉 REGRESSION PASSED: The LangChain Agent pipeline preserves 100% functionality of the original RAG pipeline!")
    else:
        print("⚠️ REGRESSION WARNING: Discrepancies detected between pipelines.")
    print("=" * 78)


if __name__ == "__main__":
    run_regression_suite()
