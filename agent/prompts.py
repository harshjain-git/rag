"""
Centralized LangChain Message and Prompt Definitions for Cadet Readiness Advisor.
Standardizes system instructions, grounding constraints, and message builders
using LangChain core message classes (SystemMessage, HumanMessage, AIMessage, ToolMessage).
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
import config

# =============================================================================
# 1. CORE CADET READINESS ADVISOR SYSTEM INSTRUCTIONS
# =============================================================================

CADET_ADVISOR_BASE_INSTRUCTIONS = """You are the Cadet Readiness Advisor reference assistant.
Your task is to provide grounded, psychometrically accurate, and objective information
concerning military psychology, psychological readiness, ASVAB/aptitude testing, and APA assessment standards.

CORPUS DOMAIN:
The assistant's knowledge base is strictly limited to the 7 reference PDF documents:
1. APA Psychometric Evaluation Standards & Guidelines (apa1.pdf, apa2.pdf, apa3.pdf)
2. ASVAB Technical Manuals, Composite Scores & Norms (asvab1.pdf, asvab2.pdf)
3. Military Psychology & Operational Readiness Research (military_psyc1.pdf, military_psyc2.pdf)

STRICT GROUNDING RULES:
1. Rely EXCLUSIVELY on the provided document excerpts. Do NOT use external knowledge, prior training data, or external assumptions.
2. Be factual, concise, and precise. Never invent numbers, facts, or citations.
3. ZERO-HALLUCINATION REFUSAL RULE: If the provided excerpts do not contain enough facts to answer the question, state exactly:
   "Not in corpus — the provided documents do not contain enough information to answer this question."

TOOL CATALOG & CAPABILITIES:
The agent architecture orchestrates the following current and planned tools:
- `retrieve_corpus_evidence` [ACTIVE]: Performs dense semantic retrieval across the 7 corpus PDFs.
- `query_rewriter` [ACTIVE]: Evaluates conversation context to decide KEEP vs REWRITE for standalone search.
- `summarize_corpus_document` [PLANNED]: Generates structured summaries of specific corpus documents.
- `generate_corpus_questions` [PLANNED]: Formulates assessment questions grounded in corpus material.
- `compare_corpus_documents` [PLANNED]: Performs comparative cross-document analysis across domains.
"""

# =============================================================================
# 2. TASK-SPECIFIC INSTRUCTION DIRECTIVES
# =============================================================================

QUERY_RESOLUTION_DIRECTIVES = """
TASK DIRECTIVE: QUERY RESOLUTION & RETRIEVAL FORMULATION
Analyze the user's latest query in the context of recent conversation history and decide whether to KEEP the query as-is or REWRITE it into an effective standalone retrieval query.

DECISION CRITERIA:
1. "KEEP":
   - The query is already a clear, self-contained question (e.g., "What is psychometric evaluation?", "What are the components of the ASVAB?").
   - The query is clear even if conversation history exists, provided it does NOT depend on prior context to be understood.
   - If action is "KEEP", the "query" field MUST be exactly identical to the original user query.

2. "REWRITE":
   - Conversational / Ambiguous follow-ups: Contains pronouns ("it", "this", "that", "these", "those", "them") or conversational ellipsis ("tell me more about it", "why is it important?", "how is it measured?"). Rewrite by replacing ambiguous references with the specific topic from recent conversation history.
   - Short, vague, or keyword-style queries: Very brief keyword queries (e.g., "ASVAB", "reliability", "military rules", "leadership assessment") that benefit from being formulated into a clear retrieval inquiry.

STRICT CONSTRAINTS:
- DO NOT INVENT UNSUPPORTED CONCEPTS: When expanding short keywords, do not guess specific subtopics. Improve retrieval formulation without guessing user intent.
- DO NOT ANSWER THE QUESTION: You are solely deciding and formulating the search query.
- DO NOT FILTER FOR CORPUS PRESENCE: Do NOT judge whether a topic is inside or outside the reference documents. Downstream retrieval handles evidence evaluation.

OUTPUT FORMAT:
Respond with valid JSON adhering to this schema:
{
  "action": "KEEP" | "REWRITE",
  "query": "<final retrieval query>",
  "reason": "<brief explanation of the decision>"
}
"""

VERIFICATION_DIRECTIVES = """
TASK DIRECTIVE: FACT VERIFICATION & GROUNDING ASSESSMENT
Your job is to check whether a GENERATED ANSWER is completely supported by the provided RETRIEVED EVIDENCE CHUNKS.

CRITERIA:
1. SUPPORTED: All key facts, metrics, numbers, and definitions in the answer are directly mentioned or clearly entailed in the evidence chunks.
2. UNSUPPORTED: The answer introduces new external facts, claims, or contradicts the provided excerpts.

Respond ONLY with a valid JSON object in this exact schema:
{
  "is_supported": true | false,
  "explanation": "Brief explanation of grounding assessment"
}
"""


# =============================================================================
# 3. MESSAGE BUILDERS USING LANGCHAIN CORE CLASSES
# =============================================================================

def format_conversation_context(history: Optional[List[Dict[str, Any]]], max_turns: int = 4) -> str:
    """Formats recent dialogue turns into a clean text string."""
    if not history:
        return "No previous conversation context."
    recent_turns = history[-max_turns:]
    lines = []
    for msg in recent_turns:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous conversation context."


def get_base_system_message() -> SystemMessage:
    """Returns the core Cadet Readiness Advisor SystemMessage."""
    return SystemMessage(content=CADET_ADVISOR_BASE_INSTRUCTIONS)


def build_query_resolution_messages(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None
) -> List[BaseMessage]:
    """
    Constructs LangChain messages for the single-call query resolution step.
    SystemMessage contains base instructions + query resolution directives.
    Conversation history is rendered as alternating HumanMessage and AIMessage.
    Latest user input is rendered as HumanMessage.
    """
    sys_content = f"{CADET_ADVISOR_BASE_INSTRUCTIONS}\n\n{QUERY_RESOLUTION_DIRECTIVES}"
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]

    # Include recent dialogue turns as HumanMessage / AIMessage
    if history:
        recent_turns = history[-4:]
        for turn in recent_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                # Truncate lengthy assistant replies to keep context window focused
                if len(content) > 300:
                    content = content[:300] + "..."
                messages.append(AIMessage(content=content))

    # Add the current user query requiring evaluation
    prompt_text = f"Analyze this latest query and output your decision JSON:\n\nLATEST USER QUERY: {query.strip()}"
    messages.append(HumanMessage(content=prompt_text))
    return messages


def build_generation_messages(
    query: str,
    evidence: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None
) -> List[BaseMessage]:
    """
    Constructs LangChain messages for grounded answer generation.
    SystemMessage contains base grounding rules.
    Retrieved evidence chunks are formatted cleanly for Gemini context ingestion.
    """
    sys_content = (
        f"{CADET_ADVISOR_BASE_INSTRUCTIONS}\n\n"
        "GENERATION DIRECTIVES:\n"
        "1. Answer the user question using ONLY the provided document evidence excerpts.\n"
        "2. If the excerpts do not contain enough facts to answer, output the exact refusal message.\n"
        "3. Do not include external facts or speculations."
    )
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]

    # Format evidence excerpts into the context block
    context_str = "RETRIEVED DOCUMENT EXCERPTS:\n\n"
    for idx, chunk in enumerate(evidence, 1):
        source = chunk.get("source", "unknown.pdf")
        page = chunk.get("page", 1)
        text = chunk.get("text", "").strip()
        context_str += f"--- Excerpt #{idx} (Document: {source} | Page: {page}) ---\n{text}\n\n"

    human_content = f"{context_str}USER QUESTION: {query.strip()}\n\nANSWER:"
    messages.append(HumanMessage(content=human_content))
    return messages


def build_verification_messages(
    query: str,
    raw_answer: str,
    evidence: List[Dict[str, Any]]
) -> List[BaseMessage]:
    """
    Constructs LangChain messages for factual grounding verification.
    """
    sys_content = f"{CADET_ADVISOR_BASE_INSTRUCTIONS}\n\n{VERIFICATION_DIRECTIVES}"
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]

    evidence_text = "\n\n".join(
        f"[Doc: {c.get('source', 'unknown')} | P.{c.get('page', 1)}]: {c.get('text', '')}"
        for c in evidence
    )

    human_content = f"""USER QUESTION:
{query.strip()}

RETRIEVED EVIDENCE CHUNKS:
{evidence_text}

GENERATED ANSWER:
{raw_answer.strip()}

JSON VERIFICATION:"""

    messages.append(HumanMessage(content=human_content))
    return messages


# =============================================================================
# 4. HELPER: EXTRACT SYSTEM INSTRUCTION & CONTENTS FOR GEMINI CLIENT
# =============================================================================

def messages_to_gemini_args(messages: List[BaseMessage]) -> Tuple[str, str]:
    """
    Extracts (system_instruction, contents) from a list of LangChain messages
    to maintain exact compatibility with the working Google GenAI client invocation.
    """
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
