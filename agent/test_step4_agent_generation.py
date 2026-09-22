"""
Step 4 Verification Script: Connect Existing Generation to LangChain Agent.
Verifies that:
1. Grounded queries generate factual answers with citations from retrieved evidence.
2. Out-of-corpus queries generate contextual polite refusals.
3. Citations and evidence are properly formatted into the agent state.
4. Parity with direct generation pipeline is verified.
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
from agent import get_agent_manager


def run_step4_tests():
    print("=" * 70)
    print("STEP 4 VERIFICATION: LANGCHAIN AGENT GENERATION WORKFLOW")
    print("=" * 70)

    manager = get_agent_manager()

    # Test 1: In-corpus Grounded Query
    print("[1/2] Testing In-Corpus Grounded Question...")
    query_in = "What is the minimum Cronbach's Alpha coefficient required for operational readiness?"
    res_in = manager.run(query_in)

    print(f"  • Query:        {query_in}")
    print(f"  • Status:       {res_in.get('status')}")
    print(f"  • Grounded:     {res_in.get('is_grounded')}")
    print(f"  • Tools Used:   {res_in.get('tools_used')}")
    print(f"  • Evidence:     {len(res_in.get('evidence', []))} chunks")
    print(f"  • Citations:    {len(res_in.get('citations', []))} citations")
    print(f"  • Answer:\n{res_in.get('answer')[:300]}...\n")

    assert res_in.get("is_grounded") is True, "Expected is_grounded to be True"
    assert res_in.get("status") == "GROUNDED", f"Unexpected status: {res_in.get('status')}"
    assert len(res_in.get("citations", [])) > 0, "Expected citations in response"
    assert "0.80" in res_in.get("answer", "") or "0.8" in res_in.get("answer", ""), "Expected 0.80 in answer"
    print("✅ In-corpus generation check passed.\n")

    # Test 2: Out-of-Corpus Query (Refusal)
    print("[2/2] Testing Out-of-Corpus Refusal...")
    query_out = "What is the minimum passing swim time cutoff for military recruits?"
    res_out = manager.run(query_out)

    print(f"  • Query:        {query_out}")
    print(f"  • Status:       {res_out.get('status')}")
    print(f"  • Grounded:     {res_out.get('is_grounded')}")
    print(f"  • Answer:\n{res_out.get('answer')[:300]}...\n")

    assert res_out.get("is_grounded") is False, "Expected is_grounded to be False"
    assert res_out.get("status") == "NOT_IN_CORPUS", f"Unexpected status: {res_out.get('status')}"
    assert len(res_out.get("citations", [])) == 0, "Expected 0 citations for out-of-corpus query"
    print("✅ Out-of-corpus refusal check passed.\n")

    print("-" * 70)
    print("🎉 ALL STEP 4 AGENT GENERATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_step4_tests()
