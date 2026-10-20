# infrastructure/gemini.py

"""LLM adapter for Google Gemini.

Interacts with the Google Gemini API using the official ``google.genai`` SDK
for structured JSON generation, native tool calling, and text responses.
"""

import json
import logging
import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

# Suppress noisy genai warnings
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from google import genai
from google.genai import types

import config


class GeminiDailyQuotaExhausted(Exception):
    """Raised immediately when Google Gemini daily quota (RPD / per-day limit) is exhausted."""
    pass


def is_daily_quota_error(error: Exception) -> bool:
    """Check if the exception indicates that the daily quota has been exhausted."""
    msg = str(error).lower()
    daily_indicators = (
        "per day",
        "per_day",
        "rpd",
        "daily",
        "free_tier_requests_per_day",
        "requests per day",
        "requestsperday",
        "generate_content_free_tier_requests",
    )
    if any(indicator in msg for indicator in daily_indicators):
        return True

    # If the server requests a retry delay longer than 5 minutes (e.g. hours or >300s),
    # it is a daily quota exhaustion, not a transient spike.
    delay_match = re.search(
        r"retry\s+(?:in|after)\s+(?:approximately\s+)?(?:(\d+)h|(\d+)m|(\d+(?:\.\d+)?)\s*s)",
        msg,
    )
    if delay_match:
        hours, minutes, seconds = delay_match.groups()
        if hours or minutes:
            return True
        if seconds and float(seconds) > 300.0:
            return True

    retry_delay_match = re.search(r"'retrydelay':\s*'(\d+)s'", msg)
    if retry_delay_match and float(retry_delay_match.group(1)) > 300.0:
        return True

    return False


def is_retryable_gemini_error(error: Exception) -> bool:
    """Determine if a Gemini API error is temporary and eligible for retry (429 or 503)."""
    # Daily quota exhaustion cannot be resolved by immediate retries
    if is_daily_quota_error(error):
        return False

    code = getattr(error, "code", None)
    if code in (429, 503):
        return True

    msg = str(error).lower()
    retryable_terms = (
        "429",
        "503",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "unavailable",
        "overloaded",
        "service unavailable",
    )
    return any(term in msg for term in retryable_terms)


def extract_retry_delay(error: Exception, fallback: float = 2.0) -> float:
    """Extract server-specified retry delay in seconds if available in the error message."""
    msg = str(error)
    patterns = [
        r"retry in (?:approximately )?(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)",
        r"retry after (?:approximately )?(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            try:
                # Add a 1.0 second buffer over the server-reported duration
                return float(match.group(1)) + 1.0
            except (ValueError, IndexError):
                pass
    return fallback


class GeminiLLM:
    """Gemini client adapter.

    Features:
    - Singleton client (created once, reused across calls).
    - Structured JSON mode via ``response_mime_type="application/json"``.
    - Native tool calling via ``generate_with_tools``.
    - Plain-text generation for refusals and free-form answers.
    - Configurable model name and API key via ``config.py``.
    - Robust retry logic with exponential backoff for temporary 429 and 503 errors.
    - Fast-fail protection on daily quota exhaustion.
    """

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_DELAY = 1.0
    DEFAULT_BACKOFF_FACTOR = 2.0
    MAX_DELAY = 60.0

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

    # ---- retry execution wrapper --------------------------------------

    def _call_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a Gemini API call with exponential backoff on retryable 429/503 errors.

        - Respects server-provided retry delays when present.
        - Raises GeminiDailyQuotaExhausted immediately on daily quota exhaustion without retrying.
        - Retries up to DEFAULT_MAX_RETRIES times for transient 429/503 errors.
        - Immediately propagates permanent errors (400, 401, 403, 404, etc.).
        """
        max_retries = self.DEFAULT_MAX_RETRIES
        delay = self.DEFAULT_INITIAL_DELAY

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 1. Daily quota: Fail fast without repeated retries
                if is_daily_quota_error(e):
                    logger.error("Gemini daily quota exhausted: %s", e)
                    raise GeminiDailyQuotaExhausted(
                        f"Gemini daily quota exhausted: {e}"
                    ) from e

                # 2. Check if error is retryable (429, 503) and we have retries left
                if is_retryable_gemini_error(e) and attempt < max_retries:
                    server_delay = extract_retry_delay(e, fallback=delay)
                    wait_time = min(server_delay, self.MAX_DELAY)
                    logger.warning(
                        "Gemini temporary error (attempt %d/%d): %s. Backing off for %.1fs...",
                        attempt + 1,
                        max_retries,
                        e,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    delay = min(delay * self.DEFAULT_BACKOFF_FACTOR, self.MAX_DELAY)
                    continue

                # 3. Permanent error or retries exhausted: Propagate cleanly
                logger.error(
                    "Gemini API call failed permanently or retries exhausted: %s",
                    e,
                )
                raise

    # ---- LLM Protocol implementation ----------------------------------

    def generate_json(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a prompt and parse the structured JSON response."""
        response = self._call_with_retry(
            self.client.models.generate_content,
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
        response = self._call_with_retry(
            self.client.models.generate_content,
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

        response = self._call_with_retry(
            self.client.models.generate_content,
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


__all__ = [
    "GeminiLLM",
    "GeminiDailyQuotaExhausted",
    "is_daily_quota_error",
    "is_retryable_gemini_error",
    "extract_retry_delay",
    "get_llm",
    "generate_structured_json",
    "generate_with_native_tools",
    "generate_natural_refusal",
]
