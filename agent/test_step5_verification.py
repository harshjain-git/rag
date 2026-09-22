"""
Step 5 Verification Script: Add Verification / Grounding Node.
Verifies that:
1. Answers generated from retrieved evidence pass factual grounding verification.
2. Contextual out-of-corpus refusals are verified as valid refusals.
3. Verification status and details are properly populated into the agent state.
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


def run_step5_tests():
    print("=" * 70)
    print("STEP 5 VERIFICATION: GROUNDING & VERIFICATION NODE TESTS")
    print("=" * 70)

    manager = get_agent_manager()

    # Test 1: Grounded In-Corpus Question
    print("[1/2] Testing Verification on Grounded In-Corpus Query...")
    query_in = "What is the minimum Cronbach's Alpha coefficient required for operational readiness?"
    res_in = manager.run(query_in)

    print(f"  • Query:                {query_in}")
    print(f"  • Grounded:             {res_in.get('is_grounded')}")
    print(f"  • Is Verified:          {res_in.get('is_verified')}")
    print(f"  • Verification Status:  {res_in.get('verification_status')}")
    print(f"  • Verification Details: {res_in.get('verification_details')}")

    assert res_in.get("is_verified") is True, "Expected generated answer to be verified as supported"
    assert res_in.get("verification_status") in ("VERIFIED_SUPPORTED", "HEURISTIC_VERIFIED")
    print("✅ In-corpus verification passed.\n")

    # Test 2: Out-of-Corpus Refusal Verification
    print("[2/2] Testing Verification on Out-of-Corpus Refusal...")
    query_out = "What is the minimum passing swim time cutoff for military recruits?"
    res_out = manager.run(query_out)

    print(f"  • Query:                {query_out}")
    print(f"  • Grounded:             {res_out.get('is_grounded')}")
    print(f"  • Is Verified:          {res_out.get('is_verified')}")
    print(f"  • Verification Status:  {res_out.get('verification_status')}")
    print(f"  • Verification Details: {res_out.get('verification_details')}")

    assert res_out.get("is_verified") is True, "Expected refusal to be verified as valid"
    assert res_out.get("verification_status") == "REFUSAL_VERIFIED"
    print("✅ Out-of-corpus refusal verification passed.\n")

    print("-" * 70)
    print("🎉 ALL STEP 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_step5_tests()
