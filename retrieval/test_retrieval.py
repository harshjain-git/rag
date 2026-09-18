"""
Retrieval Testing & Verification Script for Phase 7.
Evaluates Top-K dense semantic retrieval performance against ground-truth evaluation questions.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from retrieval.retriever import retrieve_evidence


def run_retrieval_tests():
    print("=" * 70)
    print("PHASE 7: DENSE SEMANTIC RETRIEVAL TESTING (ChromaDB + BGE Embeddings)")
    print("=" * 70)

    eval_file = config.BASE_DIR / "evaluation" / "evaluation_set.json"
    if not eval_file.exists():
        print(f"❌ Error: Evaluation dataset not found at {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} Ground-Truth Evaluation Questions from benchmark dataset.\n")

    total_latency = 0
    hit_count = 0
    answerable_total = 0

    for q in questions:
        q_id = q["id"]
        q_type = q["type"].upper()
        query_text = q["question"]
        target_docs = q.get("target_documents", [])

        # Measure retrieval execution time
        start_time = time.time()
        retrieved_chunks = retrieve_evidence(query_text, top_k=config.INITIAL_TOP_K)
        latency = time.time() - start_time
        total_latency += latency

        retrieved_sources = list(set(c["source"] for c in retrieved_chunks))

        # Check document recall for answerable/comparison questions
        hit = False
        if target_docs:
            answerable_total += 1
            if any(doc in retrieved_sources for doc in target_docs):
                hit = True
                hit_count += 1

        print(f"[{q_type}] Q#{q_id}: {query_text}")
        print(f"  • Retrieval Time: {latency * 1000:.2f} ms")
        print(f"  • Top-{len(retrieved_chunks)} Retrieved Sources: {retrieved_sources}")
        if target_docs:
            print(f"  • Target Docs Expected: {target_docs} → {'✅ HIT' if hit else '❌ MISS'}")
        print(f"  • Top Result Snippet: {repr(retrieved_chunks[0]['text'][:120]) if retrieved_chunks else 'None'}...")
        print("-" * 70)

    avg_latency = (total_latency / len(questions)) * 1000 if questions else 0
    recall = (hit_count / answerable_total * 100) if answerable_total > 0 else 0

    print("RETRIEVAL SUMMARY METRICS:")
    print(f"  • Total Questions Tested: {len(questions)}")
    print(f"  • Average Retrieval Latency: {avg_latency:.2f} ms (Target < 100 ms)")
    print(f"  • Document Target Recall: {recall:.1f}% ({hit_count}/{answerable_total})")
    print("=" * 70)


if __name__ == "__main__":
    run_retrieval_tests()
