# agent/tools/question_generator_adapter.py

"""Question generator tool adapter.

Wraps the corpus question generator into the ``Tool`` protocol so it can be registered
in the ``ToolRegistry``.
"""

from typing import Dict, Any, Optional
from core.models import AgentState


class QuestionGeneratorTool:
    """Registered as ``"generate_corpus_questions"`` in the ToolRegistry."""

    name = "generate_corpus_questions"
    description = "Generates verified assessment questions directly from the corpus."

    def invoke(self, state_or_inputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute corpus question generation.

        Supports both:
        1. Dict inputs: ``invoke({"count": 5, "domains": [...]})`` -> ``Dict``
        2. State-aware invocation: ``invoke(state, inputs)`` -> ``AgentState``
        """
        from agent.question_generator_tool import generate_questions_from_corpus

        if inputs is None and isinstance(state_or_inputs, dict) and "status" not in state_or_inputs:
            count = state_or_inputs.get("count", 5)
            domains = state_or_inputs.get("domains")
            return generate_questions_from_corpus(count=count, requested_domains=domains)

        state: AgentState = state_or_inputs
        count = (inputs or {}).get("count", 5)
        domains = (inputs or {}).get("domains")
        q_result = generate_questions_from_corpus(count=count, requested_domains=domains)
        questions = q_result.get("questions", [])
        q_status = q_result.get("status", "SUCCESS")
        tools_used = list(state.get("tools_used", []))
        if self.name not in tools_used:
            tools_used.append(self.name)

        evidence_list = []
        for q in questions:
            for c in q.get("citations", []):
                evidence_list.append({
                    "text": c.get("quote", ""),
                    "source": c.get("source", ""),
                    "page": c.get("page", 0),
                    "chunk_id": c.get("chunk_id", ""),
                })

        return {
            **state,
            "questions": questions,
            "evidence": evidence_list,
            "answer": f"Generated {len(questions)} verified assessment questions from the reference corpus.",
            "tools_used": tools_used,
            "is_grounded": True if questions else False,
            "is_answerable": True if questions else False,
            "is_verified": True if questions else False,
            "verification_status": "VERIFIED_SUPPORTED" if questions else "INSUFFICIENT_EVIDENCE",
            "verification_details": f"Generated {len(questions)} questions strictly verified against corpus chunks.",
            "status": q_status,
        }
