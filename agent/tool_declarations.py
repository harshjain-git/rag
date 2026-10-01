# agent/tool_declarations.py

"""Gemini Native Tool Declarations for Cadet Readiness Advisor.

Provides schema definitions conforming to ``google.genai.types.Tool`` and
``types.FunctionDeclaration`` for native tool calling with Gemini models.
"""

from typing import List
from google.genai import types


def get_gemini_native_tools() -> List[types.Tool]:
    """Build and return the list of Gemini Tool definitions for native tool calling.

    Exposes the 4 core routing capabilities:
    1. retrieve_corpus_evidence: for questions answered from the PDF/document corpus.
    2. ask_srag_database: for structured/database questions handled by the remote SRAG agent.
    3. generate_corpus_questions: for generating assessment items from the document corpus.
    4. query_rewriter: for resolving ambiguous/follow-up queries using conversation history.
    """
    declarations = [
        types.FunctionDeclaration(
            name="retrieve_corpus_evidence",
            description=(
                "Retrieves factual and conceptual evidence from the reference document corpus "
                "(PDF documents covering ASVAB testing standards, psychometric evaluation, "
                "psychological resilience, and cadet readiness guidelines). Use this tool whenever "
                "the user asks reference, factual, conceptual, or guideline questions that should "
                "be answered from reference documentation."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "The specific search query to retrieve evidence chunks for from the document corpus."
                        ),
                    }
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="ask_srag_database",
            description=(
                "Queries the structured relational database via the remote Structured RAG (SRAG) agent. "
                "Use this tool for structured data questions, relational database lookups, employee counts, "
                "cadet personnel database queries, tables, and quantitative database records (for example: "
                "'How many employees are currently hired?', database tables, employee statistics, personnel records)."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "The natural language database query to send to the remote SRAG agent."
                        ),
                    }
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="generate_corpus_questions",
            description=(
                "Generates verified assessment questions, quizzes, or practice items grounded directly "
                "in the reference document corpus. Use this tool when the user explicitly requests to "
                "create questions, generate a quiz, practice test, or study items based on the corpus."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "num_questions": {
                        "type": "INTEGER",
                        "description": "The number of questions to generate (default is 5).",
                    },
                    "domains": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Optional list of domains to sample from, e.g. ['asvab', 'psychometrics'].",
                    },
                },
            },
        ),
        types.FunctionDeclaration(
            name="query_rewriter",
            description=(
                "Resolves conversational context, ambiguous pronouns ('them', 'it', 'those'), or "
                "conversational ellipses by reformulating the user's query into a clear standalone query "
                "based on recent conversation history. Use this when the query is ambiguous and cannot "
                "be accurately routed without resolving prior dialogue context."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "The ambiguous or contextual user query that needs conversational context resolution."
                        ),
                    }
                },
                "required": ["query"],
            },
        ),
    ]

    return [types.Tool(function_declarations=declarations)]


__all__ = ["get_gemini_native_tools"]
