# application/prompts package

"""Prompts and message builders for Cadet Readiness Advisor."""

from .cadet_prompts import (
    CADET_ADVISOR_BASE_INSTRUCTIONS,
    QUERY_RESOLUTION_DIRECTIVES,
    VERIFICATION_DIRECTIVES,
    QUESTION_GENERATION_DIRECTIVES,
    QUESTION_BATCH_VERIFICATION_DIRECTIVES,
    ORCHESTRATOR_SYSTEM_PROMPT,
    GENERATION_DIRECTIVES,
    format_conversation_context,
    get_base_system_message,
    build_query_resolution_messages,
    build_generation_messages,
    build_verification_messages,
    build_question_generation_messages,
    build_question_verification_messages,
    build_orchestrator_decision_messages,
    messages_to_gemini_args,
)

__all__ = [
    "CADET_ADVISOR_BASE_INSTRUCTIONS",
    "QUERY_RESOLUTION_DIRECTIVES",
    "VERIFICATION_DIRECTIVES",
    "QUESTION_GENERATION_DIRECTIVES",
    "QUESTION_BATCH_VERIFICATION_DIRECTIVES",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "GENERATION_DIRECTIVES",
    "format_conversation_context",
    "get_base_system_message",
    "build_query_resolution_messages",
    "build_generation_messages",
    "build_verification_messages",
    "build_question_generation_messages",
    "build_question_verification_messages",
    "build_orchestrator_decision_messages",
    "messages_to_gemini_args",
]
