# infrastructure/gemini.py

"""LLM adapter for Google Gemini.

Interacts with the Google Gemini API using the official ``google.genai`` SDK
for structured JSON generation, native tool calling, and text responses.
"""

import json
import warnings
import logging
from typing import Any, Dict, List, Optional

# Suppress noisy genai warnings
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from google import genai
from google.genai import types

import config


class GeminiLLM:
    """Gemini client adapter.

    Features:
    - Singleton client (created once, reused across calls).
    - Structured JSON mode via ``response_mime_type="application/json"``.
    - Native tool calling via ``generate_with_tools``.
    - Plain-text generation for refusals and free-form answers.
    - Configurable model name and API key via ``config.py``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            api_key: Google API key. Falls back to ``config.GEMINI_API_KEY``.
            model_name: Model identifier. Falls back to ``config.LLM_MODEL_NAME``.
        """
        self._api_key = api_key or config.GEMINI_API_KEY
        self._model_name = model_name or config.LLM_MODEL_NAME
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in environment or .env file."
            )
        self._client: genai.Client | None = None

    # ---- lazy singleton client ----------------------------------------

    @property
    def client(self) -> genai.Client:
        """Return (or create) the cached ``genai.Client`` instance."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # ---- LLM Protocol implementation ----------------------------------

    def generate_json(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a prompt and parse the structured JSON response."""
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        raw_text = (
            response.text.strip()
            if hasattr(response, "text") and response.text
            else "{}"
        )
        return json.loads(raw_text)

    def generate_text(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt and return the plain-text response."""
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        return response.text.strip() if hasattr(response, "text") and response.text else ""

    def generate_with_tools(
        self,
        contents: Any,
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a prompt with native tool definitions and return function call or text.

        Uses the existing google.genai.Client, does NOT enforce JSON response mode,
        and leaves automatic_function_calling disabled so the caller maintains full
        lifecycle control.
        """
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "tools": tools,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        function_calls = getattr(response, "function_calls", None)
        if function_calls:
            first_call = function_calls[0]
            call_args = dict(first_call.args) if getattr(first_call, "args", None) else {}
            all_calls = [
                {
                    "name": c.name,
                    "arguments": dict(c.args) if getattr(c, "args", None) else {},
                }
                for c in function_calls
            ]
            return {
                "call_type": "function_call",
                "function_name": first_call.name,
                "arguments": call_args,
                "function_calls": all_calls,
                "text": None,
                "raw_response": response,
            }

        raw_text = (
            response.text.strip()
            if hasattr(response, "text") and response.text
            else ""
        )
        return {
            "call_type": "text",
            "function_name": None,
            "arguments": {},
            "function_calls": [],
            "text": raw_text,
            "raw_response": response,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_GEMINI_LLM: GeminiLLM | None = None


def get_llm() -> GeminiLLM:
    """Return a module-level singleton ``GeminiLLM`` instance.

    Usage::

        from infrastructure.gemini import get_llm
        llm = get_llm()
        result = llm.generate_json(contents="...", system_instruction="...")
    """
    global _GEMINI_LLM
    if _GEMINI_LLM is None:
        _GEMINI_LLM = GeminiLLM()
    return _GEMINI_LLM


def generate_structured_json(
    contents: str,
    system_instruction: str = "",
    temperature: float = 0.0,
) -> Any:
    """Convenience helper for structured JSON generation."""
    return get_llm().generate_json(
        contents=contents,
        system_instruction=system_instruction,
        temperature=temperature,
    )


def generate_with_native_tools(
    contents: Any,
    tools: List[Any],
    system_instruction: str = "",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Convenience helper for Gemini native tool calling."""
    return get_llm().generate_with_tools(
        contents=contents,
        tools=tools,
        system_instruction=system_instruction,
        temperature=temperature,
    )


def generate_natural_refusal(query: str) -> str:
    """Generates a polite refusal when a question cannot be answered from corpus."""
    from application.prompts import build_generation_messages, messages_to_gemini_args

    messages = build_generation_messages(query=query, evidence=[])
    system_instruction, contents = messages_to_gemini_args(messages)

    try:
        return get_llm().generate_text(
            contents=contents,
            system_instruction=system_instruction,
            temperature=0.0,
        )
    except Exception:
        return config.NOT_IN_CORPUS_MESSAGE
