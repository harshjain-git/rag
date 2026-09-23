# infrastructure/gemini/gemini_llm.py

"""Concrete LLM adapter for Google Gemini.

This is the **only** file in the entire project that imports the ``google.genai``
SDK. Every other module depends on the ``core.ports.LLM`` Protocol instead.

Swapping Gemini for another provider (OpenAI, Anthropic, etc.) requires only
creating a new adapter that satisfies the same Protocol — zero changes to agent
code, tools, or pipeline.
"""

import json
import warnings
import logging
from typing import Any, Dict

# Suppress noisy genai warnings
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from google import genai
from google.genai import types

import config


class GeminiLLM:
    """Implements ``core.ports.LLM`` using Google Gemini.

    Features:
    - Singleton client (created once, reused across calls).
    - Structured JSON mode via ``response_mime_type="application/json"``.
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
        """Send a prompt and parse the structured JSON response.

        This is the drop-in replacement for the old
        ``generation.generator.generate_structured_json`` helper.
        """
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
        """Send a prompt and return the plain-text response.

        This is the drop-in replacement for the old
        ``generation.generator.generate_natural_refusal`` and similar
        free-form generation calls.
        """
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


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors the old _LLM_CLIENT pattern)
# ---------------------------------------------------------------------------

_GEMINI_LLM: GeminiLLM | None = None


def get_llm() -> GeminiLLM:
    """Return a module-level singleton ``GeminiLLM`` instance.

    Usage::

        from infrastructure.gemini.gemini_llm import get_llm
        llm = get_llm()
        result = llm.generate_json(contents="...", system_instruction="...")
    """
    global _GEMINI_LLM
    if _GEMINI_LLM is None:
        _GEMINI_LLM = GeminiLLM()
    return _GEMINI_LLM
