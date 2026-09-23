# Architecture Refactoring Progress — Cadet Readiness Advisor

## Current State: Steps 1 through 11 Completed (11/14)

This document tracks the execution progress of the modular architecture refactoring.

---

### Step-by-Step Execution Summary

#### ✅ Step 1: Baseline Inspection & Tracking
- Inspected the repository architecture, dependencies, and execution paths.
- Established tracking artifacts.

#### ✅ Step 2: Core Domain Models
- Created `core/models/agent_state.py` containing the `AgentState` TypedDict definition.
- Exported via `core/models/__init__.py`.

#### ✅ Step 3: Dedicated Prompts Module
- Created `application/prompts/cadet_prompts.py` centralizing all 7 system prompts & directives and all message builders.
- Exported via `application/prompts/__init__.py`.
- Converted `agent/prompts.py` into a backward-compatible re-export facade.

#### ✅ Step 4: LLM Gateway Adapter
- Created `core/ports/llm.py` defining the provider-agnostic `LLM` Protocol (`generate_json`, `generate_text`).
- Created `infrastructure/gemini/gemini_llm.py` (`GeminiLLM`) as the sole module importing `google.genai`.

#### ✅ Step 5: Chroma Retriever Adapter
- Created `core/ports/retriever.py` defining the `Retriever` Protocol (`retrieve`, `retrieve_raw`).
- Created `infrastructure/chroma/chroma_retriever.py` (`ChromaRetriever`) encapsulating all vector database interactions.

#### ✅ Step 6: Tool Contract & ToolRegistry
- Created `core/ports/tool.py` defining the state-aware `Tool` Protocol.
- Created `core/registry/tool_registry.py` defining the singleton `ToolRegistry`.

#### ✅ Step 7: Strangler Fig Facades
- Refactored `generation/generator.py` into a thin facade delegating to `GeminiLLM`.
- Refactored `retrieval/retriever.py` into a thin facade delegating to `ChromaRetriever`.
- Preserved existing signatures so all callers continue working without disruption.

#### ✅ Step 8: Generic AgentPipeline & Tool Adapters
- Created `agent/pipeline.py` implementing `AgentPipeline` (runs `retrieve_corpus_evidence` → `generate_answer` → `verify_grounding` looked up from `ToolRegistry`).
- Created tool adapters in `agent/tools/`:
  - `OrchestratorTool` (`"orchestrator"`)
  - `RetrievalTool` (`"retrieve_corpus_evidence"`)
  - `GenerationTool` (`"generate_answer"`)
  - `VerificationTool` (`"verify_grounding"`, `"verification"`)
  - `QueryRewriterTool` (`"query_rewriter"`)
  - `QuestionGeneratorTool` (`"generate_corpus_questions"`)
- Fixed routing in `agent/chain.py` (`if tool_name == "query_rewriter":`).

#### ✅ Step 9: Composition Root / Bootstrap
- Updated `core/bootstrap.py` to register all 6 application tools in `ToolRegistry` automatically upon import.

#### ✅ Step 10: Service Facade & UI Decoupling
- Created `application/services/agent_service.py` (`AgentService.query()`).
- Updated `app/streamlit_app.py` to consume `AgentService`, decoupling UI from agent internals and database details.

#### ✅ Step 11: Dead Code & Obsolete Dependencies Cleanup
- Removed unused `langchain-google-genai` from `requirements.txt`.
- Removed dead `types` SDK import from `agent/chain.py`.
- Removed unused `get_gemini_client` and `format_context_prompt` imports.
- Removed unused `json` imports and dead prompt strings from `verification.py` and `query_rewriter_tool.py`.
- Updated all modules to import prompts directly from `application.prompts`.

---

### Remaining Steps (3 Steps to 100% Completion)

#### 🔲 Step 12: Automated Unit & Architecture Tests
- Add `tests/` directory with `pytest`.
- Test `ToolRegistry` registration and lookup.
- Test `AgentPipeline` with mocked tools.
- Test `AgentService.query()` response formatting.

#### 🔲 Step 13: Full Regression Testing
- Validate all user query paths:
  - Factual retrieval queries with grounding citations.
  - Out-of-corpus queries with verified polite refusals.
  - Conversational follow-ups with query rewriter.
  - Assessment quiz question generation.

#### 🔲 Step 14: Final Documentation & Wrap-Up
- Finalize developer guides, architecture diagrams, and deployment instructions.
