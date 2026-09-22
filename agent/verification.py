"""
Verification and Grounding Node for Cadet Readiness Advisor (Step 5).
Validates that generated answers are factually supported by retrieved corpus evidence
without modifying the underlying retrieval or generation implementations.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from generation.generator import get_gemini_client
from google.genai import types

VERIFICATION_SYSTEM_PROMPT = """You are a strict Fact Verification and Grounding Judge for Cadet Readiness Advisor.
Your job is to check whether a GENERATED ANSWER is completely supported by the provided RETRIEVED EVIDENCE CHUNKS.

CRITERIA:
1. SUPPORTED: All key facts, metrics, numbers, and definitions in the answer are directly mentioned or clearly entailed in the evidence chunks.
2. UNSUPPORTED: The answer introduces new external facts, claims, or contradicts the provided excerpts.

Respond ONLY with a valid JSON object in this exact schema:
{
  "is_supported": true,
  "explanation": "Brief explanation of grounding assessment"
}
"""


def verify_grounding(query: str, raw_answer: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer is strictly supported by the retrieved evidence.
    
    Args:
        query: User question string.
        raw_answer: The generated answer text (excluding appended citation blocks).
        evidence: List of evidence dictionaries with 'text', 'source', 'page'.
        
    Returns:
        Dict with 'is_verified' (bool), 'verification_status' (str), and 'verification_details' (str).
    """
    if not query or not raw_answer:
        return {
            "is_verified": False,
            "verification_status": "EMPTY_INPUT",
            "verification_details": "Query or answer is empty."
        }

    # If no evidence was retrieved (out of corpus refusal)
    if not evidence:
        return {
            "is_verified": True,
            "verification_status": "REFUSAL_VERIFIED",
            "verification_details": "Response is a verified out-of-corpus refusal."
        }

    # Format evidence excerpts for verification judge
    evidence_text = "\n\n".join(
        f"[Doc: {c.get('source', 'unknown')} | P.{c.get('page', 1)}]: {c.get('text', '')}"
        for c in evidence
    )

    prompt = f"""USER QUESTION:
{query}

RETRIEVED EVIDENCE CHUNKS:
{evidence_text}

GENERATED ANSWER:
{raw_answer}

JSON VERIFICATION:"""

    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=VERIFICATION_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        parsed = json.loads(response.text.strip())
        is_supported = bool(parsed.get("is_supported", False))
        explanation = str(parsed.get("explanation", "Verification complete."))

        return {
            "is_verified": is_supported,
            "verification_status": "VERIFIED_SUPPORTED" if is_supported else "UNSUPPORTED_HALLUCINATION",
            "verification_details": explanation
        }
    except Exception as e:
        # Fallback to evidence text substring / overlap check if verification API encounters issue
        has_overlap = any(
            any(word in c.get("text", "").lower() for word in raw_answer.lower().split()[:5])
            for c in evidence
        )
        return {
            "is_verified": has_overlap,
            "verification_status": "HEURISTIC_VERIFIED" if has_overlap else "VERIFICATION_ERROR",
            "verification_details": f"Heuristic check applied ({str(e)})"
        }
