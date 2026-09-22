"""
Verification and Parity Test Script for Step 2: LangChain Retrieval Tool.
Compares direct retriever outputs with tool execution outputs to ensure 100% parity.
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
from agent.retrieval_tool import retrieve_corpus_evidence, get_retrieval_tool
from agent.agent_manager import get_agent_manager


def run_retrieval_tool_tests():
    print("=" * 70)
    print("STEP 2 VERIFICATION: LANGCHAIN RETRIEVAL TOOL PARITY TESTS")
    print("=" * 70)

    # 1. Verify Tool Registration and Meta
    tool = get_retrieval_tool()
    print(f"Tool Name:        {tool.name}")
    print(f"Tool Description: {tool.description[:80]}...")
    assert tool.name == "retrieve_corpus_evidence", f"Unexpected tool name: {tool.name}"
    print("✅ Tool definition & metadata check passed.\n")

    # 2. Test Agent Manager Registration
    manager = get_agent_manager()
    registered = manager.get_registered_tools()
    registered_names = [t.name for t in registered]
    print(f"Agent Manager Registered Tools: {registered_names}")
    assert "retrieve_corpus_evidence" in registered_names, "Tool not registered in Agent Manager"
    print("✅ Agent Manager tool registration check passed.\n")

    # 3. Parity Tests against Evaluation Dataset
    eval_file = config.BASE_DIR / "evaluation" / "evaluation_set.json"
    if not eval_file.exists():
        print(f"❌ Error: Evaluation dataset not found at {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running parity tests across {len(questions)} evaluation questions...\n")

    all_passed = True
    for q in questions:
        q_id = q["id"]
        query_text = q["question"]

        # Direct retrieval output
        direct_evidence = retrieve_evidence(query_text, top_k=config.INITIAL_TOP_K)

        # Tool execution output
        tool_evidence = tool.invoke({"query": query_text})

        # Compare outputs
        if len(direct_evidence) != len(tool_evidence):
            print(f"❌ Mismatch in chunk count for Q#{q_id}: direct={len(direct_evidence)}, tool={len(tool_evidence)}")
            all_passed = False
            continue

        for idx, (d_chunk, t_chunk) in enumerate(zip(direct_evidence, tool_evidence)):
            if d_chunk["text"] != t_chunk["text"]:
                print(f"❌ Text mismatch in Q#{q_id} chunk #{idx}")
                all_passed = False
            if d_chunk["source"] != t_chunk["source"]:
                print(f"❌ Source mismatch in Q#{q_id} chunk #{idx}")
                all_passed = False
            if d_chunk["page"] != t_chunk["page"]:
                print(f"❌ Page mismatch in Q#{q_id} chunk #{idx}")
                all_passed = False
            if d_chunk["score"] != t_chunk["score"]:
                print(f"❌ Score mismatch in Q#{q_id} chunk #{idx}")
                all_passed = False

        status_str = "✅ PARITY OK" if all_passed else "❌ PARITY FAILED"
        print(f"Q#{q_id} [{q['type']}]: '{query_text[:45]}...' -> {len(tool_evidence)} chunks ({status_str})")

    print("-" * 70)
    if all_passed:
        print("🎉 ALL STEP 2 RETRIEVAL TOOL PARITY TESTS PASSED SUCCESSFULLY!")
    else:
        print("❌ SOME PARITY TESTS FAILED.")
    print("=" * 70)


if __name__ == "__main__":
    run_retrieval_tool_tests()
