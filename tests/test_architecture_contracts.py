"""Architectural compliance and boundary enforcement tests.

Ensures that the modular layering rules are respected:
1. `core/` has ZERO dependencies on external frameworks (google.genai, chromadb, streamlit).
2. `infrastructure/gemini/gemini_llm.py` is the only internal module importing `google.genai`.
3. All registered tools satisfy the `Tool` Protocol.
4. Infrastructure adapters satisfy their respective Protocol contracts.
"""

import os
import re
import unittest
from pathlib import Path
from core.ports import Tool, LLM, Retriever
from infrastructure.gemini import GeminiLLM
from infrastructure.chroma import ChromaRetriever
from agent.tools import (
    OrchestratorTool,
    RetrievalTool,
    GenerationTool,
    VerificationTool,
    QueryRewriterTool,
    QuestionGeneratorTool,
)


PROJECT_ROOT = Path(__file__).parent.parent


class TestArchitectureContracts(unittest.TestCase):
    """Architectural integrity test suite."""

    def test_core_layer_has_no_framework_dependencies(self) -> None:
        """Domain core layer must not import google.genai, chromadb, or streamlit."""
        core_dir = PROJECT_ROOT / "core"
        forbidden_patterns = [
            re.compile(r"^\s*(from|import)\s+google(\.genai)?", re.MULTILINE),
            re.compile(r"^\s*(from|import)\s+chromadb", re.MULTILINE),
            re.compile(r"^\s*(from|import)\s+streamlit", re.MULTILINE),
        ]

        for py_file in core_dir.rglob("*.py"):
            code = py_file.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                match = pattern.search(code)
                self.assertIsNone(
                    match,
                    f"Core module '{py_file.relative_to(PROJECT_ROOT)}' violates domain purity "
                    f"by importing: {match.group(0) if match else ''}",
                )

    def test_sole_google_genai_importer(self) -> None:
        """infrastructure/gemini.py is sole agent inference LLM caller (ingestion holds embedding pipeline)."""
        genai_import_pattern = re.compile(r"^\s*(from|import)\s+google\.genai|^\s*from\s+google\s+import\s+genai", re.MULTILINE)
        allowed_files = {
            (PROJECT_ROOT / "infrastructure" / "gemini.py").resolve(),
            (PROJECT_ROOT / "ingestion" / "vector_store.py").resolve(),
        }

        # Scan all python files in project (excluding venv, .git, etc.)
        for py_file in PROJECT_ROOT.rglob("*.py"):
            # Skip virtual environments and hidden directories
            if any(part in py_file.parts for part in ["venv", ".git", "__pycache__", ".streamlit"]):
                continue
            if py_file.resolve() in allowed_files:
                continue

            content = py_file.read_text(encoding="utf-8", errors="ignore")
            match = genai_import_pattern.search(content)
            self.assertIsNone(
                match,
                f"File '{py_file.relative_to(PROJECT_ROOT)}' imports google.genai directly. "
                "All LLM interactions must go through core.ports.LLM / infrastructure.gemini.gemini_llm.",
            )

    def test_all_tool_adapters_implement_tool_protocol(self) -> None:
        """Every tool adapter in agent/tools/ must satisfy the Tool Protocol."""
        tool_instances = [
            OrchestratorTool(),
            RetrievalTool(),
            GenerationTool(),
            VerificationTool(),
            QueryRewriterTool(),
            QuestionGeneratorTool(),
        ]
        for tool in tool_instances:
            self.assertIsInstance(
                tool,
                Tool,
                f"{type(tool).__name__} does not satisfy core.ports.tool.Tool protocol",
            )
            self.assertTrue(callable(getattr(tool, "invoke", None)))

    def test_gemini_adapter_implements_llm_protocol(self) -> None:
        """GeminiLLM adapter must conform to core.ports.llm.LLM protocol."""
        adapter = GeminiLLM(api_key="mock_key", model_name="mock_model")
        self.assertIsInstance(adapter, LLM)
        self.assertTrue(callable(getattr(adapter, "generate_json", None)))
        self.assertTrue(callable(getattr(adapter, "generate_text", None)))

    def test_chroma_adapter_implements_retriever_protocol(self) -> None:
        """ChromaRetriever adapter must conform to core.ports.retriever.Retriever protocol."""
        # Using dummy vector_store mock to avoid loading actual SQLite DB during contract check
        adapter = ChromaRetriever(vector_store=object())
        self.assertIsInstance(adapter, Retriever)
        self.assertTrue(callable(getattr(adapter, "retrieve", None)))
        self.assertTrue(callable(getattr(adapter, "retrieve_raw", None)))


if __name__ == "__main__":
    unittest.main()
