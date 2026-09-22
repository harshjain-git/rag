"""
Step 3 Verification Script: Connect Retrieval Tool to Agent/Graph.
Verifies that:
1. User queries enter the LangChain agent pipeline.
2. The agent executes the retrieval tool node.
3. Retrieved evidence is placed directly into the agent state.
4. Parity is maintained across all 18 benchmark evaluation questions.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from retrieval.retriever import retrieve_evidence
from agent import get_agent_manager, get_agent_pipeline


def run_step3_tests():
    print("=" * 70)
    print("STEP 3 VERIFICATION: AGENT RETRIEVAL WORKFLOW TESTS")
    print("=" * 70)

    manager = get_agent_manager()

    # 1. Test empty query handling
    print("[1/3] Testing empty query handling...")
    empty_result = manager.run("")
    assert empty_result["status"] == "EMPTY_QUERY", f"Unexpected status for empty query: {empty_result['status']}"
    assert empty_result["evidence"] == [], "Evidence should be empty for blank query"
    print("✅ Empty query handled correctly.\n")

    # 2. Test single query execution & state inspection
    print("[2/3] Testing single query execution & state output...")
    sample_query = "What is the minimum Cronbach's Alpha coefficient required for operational readiness?"
    result = manager.run(sample_query)

    print(f"  • Query:      {result.get('query')}")
    print(f"  • Status:     {result.get('status')}")
    print(f"  • Tools Used: {result.get('tools_used')}")
    print(f"  • Top Score:  {result.get('top_score')}")
    print(f"  • Chunks:     {len(result.get('evidence', []))}")

    assert result.get("status") == "EVIDENCE_RETRIEVED", f"Unexpected status: {result.get('status')}"
    assert "retrieve_corpus_evidence" in result.get("tools_used", []), "retrieve_corpus_evidence tool missing from tools_used"
    assert len(result.get("evidence", [])) > 0, "No evidence retrieved in agent state"
    print("✅ Single query state inspection passed.\n")

    # 3. Test parity across evaluation benchmark set
    print("[3/3] Running Agent State vs Direct Retriever parity tests on benchmark set...")
    eval_file = config.BASE_DIR / "evaluation" / "evaluation_set.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    all_passed = True
    for q in questions:
        q_id = q["id"]
        query_text = q["question"]

        direct_evidence = retrieve_evidence(query_text, top_k=config.INITIAL_TOP_K)
        agent_state = manager.run(query_text)
        agent_evidence = agent_state.get("evidence", [])

        if len(direct_evidence) != len(agent_evidence):
            print(f"❌ Mismatch count for Q#{q_id}: direct={len(direct_evidence)}, agent={len(agent_evidence)}")
            all_passed = False
            continue

        for idx, (d_chunk, a_chunk) in enumerate(zip(direct_evidence, agent_evidence)):
            if d_chunk["text"] != a_chunk["text"] or d_chunk["score"] != a_chunk["score"]:
                print(f"❌ Chunk mismatch in Q#{q_id} chunk #{idx}")
                all_passed = False

        status = "✅ OK" if all_passed else "❌ FAIL"
        print(f"Q#{q_id} [{q['type']}]: '{query_text[:40]}...' -> {len(agent_evidence)} chunks ({status})")

    print("-" * 70)
    if all_passed:
        print("🎉 ALL STEP 3 AGENT RETRIEVAL WORKFLOW TESTS PASSED SUCCESSFULLY!")
    else:
        print("❌ SOME STEP 3 TESTS FAILED.")
    print("=" * 70)


if __name__ == "__main__":
    run_step3_tests()
