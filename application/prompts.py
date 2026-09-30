# application/prompts.py

"""Centralized Prompt and Message Definitions for Cadet Readiness Advisor.

Standardizes system instructions, grounding constraints, and message builders
using LangChain core message classes (SystemMessage, HumanMessage, AIMessage, ToolMessage).
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


# =============================================================================
# 1. CORE CADET READINESS ADVISOR SYSTEM INSTRUCTIONS
# =============================================================================

CADET_ADVISOR_BASE_INSTRUCTIONS = """You are the Cadet Readiness Advisor reference assistant.
Strictly grounded in corpus PDFs covering: APA Assessment Standards, ASVAB Norms/Scores, and Military Psychology.

RULES:
1. Rely EXCLUSIVELY on provided corpus excerpts. Never use external knowledge or invent citations.
2. ZERO-HALLUCINATION REFUSAL: If excerpts lack sufficient evidence, politely state you cannot answer based on reference documents, and offer help on covered topics (ASVAB, psychometrics, readiness).

ACTIVE TOOLS:
- `retrieve_corpus_evidence`: Semantic search for factual/conceptual questions.
- `generate_corpus_questions`: Generates assessment/practice questions grounded in corpus chunks.
- `query_rewriter`: Resolves ambiguous conversational follow-ups into standalone queries."""


# =============================================================================
# 2. TASK-SPECIFIC INSTRUCTION DIRECTIVES (LEAN & TOKEN-EFFICIENT)
# =============================================================================

QUERY_RESOLUTION_DIRECTIVES = """TASK: QUERY RESOLUTION
Decide KEEP (query is clear standalone) or REWRITE (query has ambiguous pronouns/ellipsis like "them", "the guidelines", "give me those", "explain").
If REWRITE, resolve pronouns/context using chat history into a specific retrieval query targeting body text and specific details (e.g., "APA Guidelines for Psychological Assessment and Evaluation specific guidelines standards principles text").
Output JSON:
{"action": "KEEP" | "REWRITE", "query": "<standalone query>", "reason": "<brief reason>"}"""

VERIFICATION_DIRECTIVES = """TASK: FACT VERIFICATION
Check if the generated answer is completely supported by the provided excerpts.
Output JSON:
{"is_supported": true | false, "explanation": "<brief rationale>"}"""

QUESTION_GENERATION_DIRECTIVES = """TASK: QUESTION GENERATION
Generate assessment questions grounded strictly in the provided excerpts.
Every question must have a factual answer and cite the exact 'grounding_chunk_id' from the excerpts.
Output JSON array:
[{"question": "...", "answer": "...", "difficulty": "basic"|"intermediate"|"advanced", "question_type": "conceptual"|"definition"|"application"|"scenario"|"comparison"|"factual", "grounding_chunk_id": "<exact_id>"}]"""

QUESTION_BATCH_VERIFICATION_DIRECTIVES = """TASK: BATCH QUESTION VERIFICATION
Verify each candidate question is strictly supported and answerable by its assigned grounding chunk text.
Output JSON:
{"verdicts": [{"index": 0, "is_valid": true | false, "reason": "..."}]}"""

ORCHESTRATOR_SYSTEM_PROMPT = f"""{CADET_ADVISOR_BASE_INSTRUCTIONS}

TASK: AUTONOMOUS ROUTING & TOOL CALLING
Analyze the user input in the context of recent conversation history.
Determine whether to call one of the native tools or provide a direct natural language response.

ROUTING CRITERIA:
1. `retrieve_corpus_evidence`: Call when the question asks about facts, definitions, standards, scoring, or concepts covered in the reference document corpus (ASVAB subtests, psychometrics, psychological resilience, cadet readiness).
2. `ask_srag_database`: Call when the question asks about structured, tabular, or database records (e.g., employee counts, database tables, personnel counts, records, 'How many employees are currently hired?').
3. `generate_corpus_questions`: Call when the user explicitly asks to generate, create, or synthesize assessment questions, quizzes, or practice items from the reference corpus.
4. `query_rewriter`: Call when the user query is an ambiguous follow-up, uses unresolved pronouns ('it', 'them', 'these'), or depends heavily on previous dialogue context to be reformulated into a standalone query.

DIRECT RESPONSE CRITERIA:
- If the user is asking for solutions, answers, or explanations to previously generated questions in the conversation history (e.g. "give also the answer", "answer question 2", "show solutions"), DO NOT invoke retrieval or database tools. Provide the answer directly in natural text.
- If the user offers a simple greeting or non-domain conversational pleasantry, respond directly in natural text.
"""

GENERATION_DIRECTIVES = """TASK: GROUNDED ANSWER GENERATION
1. Summarize and detail all specific guidelines, principles, standards, and facts found in the provided excerpts that address the user's question.
2. If the excerpts contain relevant details or partial facts: {"is_answerable": true, "answer": "<factual answer grounded ONLY in excerpts>"}
3. Only if the excerpts contain ZERO relevant facts for the topic: {"is_answerable": false, "answer": "<polite refusal naming topic and offering help on ASVAB/psychometrics/readiness>"}"""


# =============================================================================
# 3. MESSAGE BUILDERS USING LANGCHAIN CORE CLASSES
# =============================================================================


def build_query_resolution_messages(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[BaseMessage]:
    """Single-call query resolution: checks for ambiguous pronouns/ellipsis."""
    messages: List[BaseMessage] = [SystemMessage(content=QUERY_RESOLUTION_DIRECTIVES)]

    if history:
        for turn in history[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                if len(content) > 200:
                    content = content[:200] + "..."
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=f"LATEST QUERY: {query.strip()}"))
    return messages


def build_generation_messages(
    query: str,
    evidence: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[BaseMessage]:
    """Constructs messages for grounded answer generation."""
    sys_content = f"{CADET_ADVISOR_BASE_INSTRUCTIONS}\n\n{GENERATION_DIRECTIVES}"
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]

    context_str = "EXCERPTS:\n\n"
    for idx, chunk in enumerate(evidence, 1):
        source = chunk.get("source", "unknown.pdf")
        page = chunk.get("page", 1)
        text = chunk.get("text", "").strip()
        context_str += f"[{source} p.{page}]: {text}\n\n"

    human_content = f"{context_str}QUESTION: {query.strip()}"
    messages.append(HumanMessage(content=human_content))
    return messages


def build_verification_messages(
    query: str,
    raw_answer: str,
    evidence: List[Dict[str, Any]],
) -> List[BaseMessage]:
    """Constructs messages for factual grounding verification."""
    messages: List[BaseMessage] = [SystemMessage(content=VERIFICATION_DIRECTIVES)]

    evidence_text = "\n\n".join(
        f"[{c.get('source', 'unknown')} p.{c.get('page', 1)}]: {c.get('text', '')}"
        for c in evidence
    )

    human_content = f"QUESTION: {query.strip()}\n\nEXCERPTS:\n{evidence_text}\n\nANSWER:\n{raw_answer.strip()}"
    messages.append(HumanMessage(content=human_content))
    return messages


def build_question_generation_messages(
    evidence_pack: List[Dict[str, Any]],
    num_questions: int = 5,
    difficulty: str = "mixed",
    question_types: Optional[List[str]] = None,
) -> List[BaseMessage]:
    """Constructs messages for batch question generation."""
    sys_content = f"{CADET_ADVISOR_BASE_INSTRUCTIONS}\n\n{QUESTION_GENERATION_DIRECTIVES}"
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]

    excerpts = []
    for idx, c in enumerate(evidence_pack, 1):
        cid = c.get("chunk_id", f"chunk_{idx}")
        src = c.get("source", "unknown.pdf")
        page = c.get("page", 1)
        text = c.get("text", "").strip()
        excerpts.append(f"[{cid} | {src} p.{page}]: {text}")

    evidence_text = "\n\n".join(excerpts)
    q_types_str = ", ".join(question_types) if question_types else "conceptual, definition, application"

    prompt_text = f"EXCERPTS:\n{evidence_text}\n\nINSTRUCTION: Generate {num_questions} questions. Difficulty: {difficulty}. Types: {q_types_str}."
    messages.append(HumanMessage(content=prompt_text))
    return messages


def build_question_verification_messages(
    candidates: List[Dict[str, Any]],
    evidence_pack: List[Dict[str, Any]],
) -> List[BaseMessage]:
    """Constructs messages for batch question grounding verification."""
    import json
    messages: List[BaseMessage] = [SystemMessage(content=QUESTION_BATCH_VERIFICATION_DIRECTIVES)]

    chunk_map = {c.get("chunk_id", ""): c.get("text", "") for c in evidence_pack}
    verification_items = []
    for idx, cand in enumerate(candidates):
        cid = cand.get("grounding_chunk_id", "")
        chunk_content = chunk_map.get(cid, "ERROR: CHUNK NOT FOUND")
        verification_items.append({
            "index": idx,
            "question": cand.get("question", ""),
            "answer": cand.get("answer", ""),
            "chunk_id": cid,
            "chunk_text": chunk_content,
        })

    prompt_text = f"CANDIDATES TO VERIFY:\n{json.dumps(verification_items, indent=2)}"
    messages.append(HumanMessage(content=prompt_text))
    return messages


def build_orchestrator_decision_messages(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[BaseMessage]:
    """Constructs LangChain messages for Gemini native tool calling."""
    messages: List[BaseMessage] = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)]

    if history:
        recent = history[-4:]
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                q_list = msg.get("questions", [])
                if q_list:
                    q_str = "\n".join(
                        f"Q{i}: {q.get('question')}\nAnswer: {q.get('answer')}"
                        for i, q in enumerate(q_list[:5], 1)
                    )
                    messages.append(AIMessage(content=f"{content}\n\nGENERATED QUESTIONS & ANSWERS:\n{q_str}"))
                else:
                    if len(content) > 300:
                        content = content[:300] + "..."
                    messages.append(AIMessage(content=content))

    prompt_text = f"USER INPUT: {query.strip()}"
    messages.append(HumanMessage(content=prompt_text))
    return messages


# =============================================================================
# 4. HELPER: EXTRACT SYSTEM INSTRUCTION & CONTENTS FOR GEMINI CLIENT
# =============================================================================

def messages_to_gemini_args(messages: List[BaseMessage]) -> Tuple[str, str]:
    """Extracts (system_instruction, contents) from a list of LangChain messages."""
    system_parts = []
    content_parts = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        elif isinstance(msg, HumanMessage):
            content_parts.append(msg.content)
        elif isinstance(msg, AIMessage):
            content_parts.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            content_parts.append(f"Tool Result [{msg.name}]: {msg.content}")
        else:
            content_parts.append(str(msg.content))

    system_instruction = "\n\n".join(system_parts)
    contents = "\n\n".join(content_parts)
    return system_instruction, contents


__all__ = [
    "CADET_ADVISOR_BASE_INSTRUCTIONS",
    "QUERY_RESOLUTION_DIRECTIVES",
    "VERIFICATION_DIRECTIVES",
    "QUESTION_GENERATION_DIRECTIVES",
    "QUESTION_BATCH_VERIFICATION_DIRECTIVES",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "GENERATION_DIRECTIVES",
    "build_query_resolution_messages",
    "build_generation_messages",
    "build_verification_messages",
    "build_question_generation_messages",
    "build_question_verification_messages",
    "build_orchestrator_decision_messages",
    "messages_to_gemini_args",
]
