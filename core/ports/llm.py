# core/ports/llm.py

"""LLM port — a provider-agnostic interface for language model calls.

Any concrete LLM adapter (Gemini, OpenAI, Anthropic, etc.) must implement
this Protocol. The rest of the application depends *only* on this contract,
never on a specific SDK.
"""

from typing import Protocol, Any, Dict, Optional


class LLM(Protocol):
    """Minimal abstraction over any LLM provider.

    Two methods cover all current usage patterns:
    - ``generate_json``: returns a parsed JSON dict (structured output).
    - ``generate_text``: returns a plain-text string (free-form output).
    """

    def generate_json(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a prompt and receive a parsed JSON dict back.

        Args:
            contents: The user/human content to send.
            system_instruction: Optional system-level instruction.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Parsed JSON as a Python dict.
        """
        ...

    def generate_text(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt and receive plain text back.

        Args:
            contents: The user/human content to send.
            system_instruction: Optional system-level instruction.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The model's response as a string.
        """
        ...
