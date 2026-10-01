# core/bootstrap.py

"""Bootstrap module to register tool callables with the ToolRegistry.

This file is imported automatically from ``agent/__init__.py`` so the
registration runs when the ``agent`` package is first imported.
"""

from core.registry import ToolRegistry
from agent.steps import (
    execute_orchestrator,
    execute_retrieval,
    execute_generation,
    execute_verification,
    execute_srag_database,
    execute_question_generator,
)
from agent.query_rewriter_tool import resolve_conversational_query


def bootstrap_tools() -> None:
    """Register all application tool callables in the ToolRegistry."""
    # 1. Orchestrator (Native Tool Calling)
    ToolRegistry.register("orchestrator", execute_orchestrator)

    # 2. Retrieval
    ToolRegistry.register("retrieve_corpus_evidence", execute_retrieval)

    # 3. Generation
    ToolRegistry.register("generate_answer", execute_generation)

    # 4. Verification
    ToolRegistry.register("verify_grounding", execute_verification)

    # 5. Query Rewriter
    ToolRegistry.register("query_rewriter", resolve_conversational_query)

    # 6. Question Generator
    ToolRegistry.register("generate_corpus_questions", execute_question_generator)

    # 7. Structured Database SRAG tool
    ToolRegistry.register("ask_srag_database", execute_srag_database)


# Execute on import for zero-config startup
bootstrap_tools()

