# core/bootstrap.py

"""Bootstrap module to register concrete tool implementations with the ToolRegistry.

This file is imported automatically from ``agent/__init__.py`` so the
registration runs when the ``agent`` package is first imported.
"""

from core.registry import ToolRegistry
from agent.tools import (
    OrchestratorTool,
    RetrievalTool,
    GenerationTool,
    VerificationTool,
    QueryRewriterTool,
    QuestionGeneratorTool,
)

# 1. Orchestrator tool
ToolRegistry.register("orchestrator", OrchestratorTool())

# 2. Retrieval tool
ToolRegistry.register("retrieve_corpus_evidence", RetrievalTool())

# 3. Generation tool
ToolRegistry.register("generate_answer", GenerationTool())

# 4. Verification tool (registered under both names for compatibility)
verification_tool = VerificationTool()
ToolRegistry.register("verify_grounding", verification_tool)
ToolRegistry.register("verification", verification_tool)

# 5. Query Rewriter tool
ToolRegistry.register("query_rewriter", QueryRewriterTool())

# 6. Question Generator tool
ToolRegistry.register("generate_corpus_questions", QuestionGeneratorTool())
