# Cadet Readiness Advisor — Modular Architecture Plan

## Goal
Establish a **clean, layered architecture** where each functional area (retrieval, generation, query rewriting, question generation, verification, UI, configuration, prompts, etc.) lives behind a **minimal-dependency interface** (Python Protocol). Adding a new capability must never require modifying `AgentPipeline.run()` or `AgentPipeline.__init__()`.

---

## 1. Package Layout

```
project_root/
│
├── core/                     # Domain-only — immutable models & contracts
│   ├── models/               # Data models (AgentState TypedDict)
│   ├── ports/                # Typed Protocols (Tool, LLM, Retriever)
│   ├── registry/             # Central ToolRegistry singleton
│   └── bootstrap.py          # Composition root — wires tools into ToolRegistry
│
├── application/              # Pure business logic
│   ├── services/             # High-level facades for UI (AgentService)
│   └── prompts/              # Centralized prompt templates & message builders
│
├── agent/                    # Agent execution engine & tool adapters
│   ├── pipeline.py           # Registry-driven AgentPipeline
│   ├── tools/                # Thin adapters implementing core.ports.Tool
│   │   ├── orchestrator_tool.py
│   │   ├── retrieval_adapter.py
│   │   ├── generation_tool.py
│   │   ├── verification_tool.py
│   │   ├── query_rewriter_adapter.py
│   │   └── question_generator_adapter.py
│   ├── prompts.py            # Re-export facade to application.prompts
│   └── agent_manager.py      # Backward-compatible manager
│
├── infrastructure/           # Concrete external provider adapters
│   ├── gemini/               # Gemini LLM adapter (implements core.ports.LLM)
│   │   └── gemini_llm.py     # ONLY file importing google.genai SDK
│   └── chroma/               # Chroma vector store adapter (implements core.ports.Retriever)
│       └── chroma_retriever.py
│
├── app/                      # Web user interface (Streamlit)
│   └── streamlit_app.py      # Consumes application.services.AgentService
│
├── config.py                 # Central application configuration
└── requirements.txt          # Production dependencies
```

---

## 2. Layer Responsibilities & Contracts

- **`core/`**: Defines domain models and Protocols (`Tool`, `LLM`, `Retriever`). Has ZERO dependencies on external frameworks like Gemini, Chroma, or Streamlit.
- **`infrastructure/`**: Contains the sole concrete implementations of technology-specific clients (`google.genai`, `chromadb`). Isolates vendor SDKs behind `core.ports`.
- **`application/`**: Contains use-case facades (`AgentService`) and prompt engineering assets (`application/prompts/`).
- **`agent/`**: Contains the execution pipeline (`AgentPipeline`) and adapters that connect step functions to the `core.ports.Tool` contract.
- **`app/`**: Thin UI layer that interacts solely with `AgentService`.

---

## 3. The 14-Step Migration Checklist

| Step | Focus Area | Status | Deliverables |
|:----:|------------|:------:|--------------|
| **1** | Baseline inspection & tracking | ✅ Completed | Initial repository review & progress baseline |
| **2** | Core models & contracts | ✅ Completed | `core/models/agent_state.py` (`AgentState` TypedDict) |
| **3** | Dedicated prompts module | ✅ Completed | `application/prompts/` & `agent/prompts.py` facade |
| **4** | LLM Gateway Adapter | ✅ Completed | `core/ports/llm.py` & `infrastructure/gemini/gemini_llm.py` |
| **5** | Chroma Retriever Adapter | ✅ Completed | `core/ports/retriever.py` & `infrastructure/chroma/chroma_retriever.py` |
| **6** | Tool Contract & Registry | ✅ Completed | `core/ports/tool.py` & `core/registry/tool_registry.py` |
| **7** | Strangler Fig Facades | ✅ Completed | `generation/generator.py` & `retrieval/retriever.py` delegate to adapters |
| **8** | Generic AgentPipeline | ✅ Completed | `agent/pipeline.py` & unified adapters in `agent/tools/` |
| **9** | Composition Root / Bootstrap | ✅ Completed | `core/bootstrap.py` automatically registers all 6 tools |
| **10** | Service Facade & UI Decoupling | ✅ Completed | `application/services/agent_service.py` & streamlined `streamlit_app.py` |
| **11** | Clean Up Dead Code & Dependencies | ✅ Completed | Cleaned `requirements.txt`, removed dead SDK imports |
| **12** | Unit & Architecture Tests | 🔲 Next | `pytest` test suite for registry, pipeline, and adapters |
| **13** | Full Regression Testing | 🔲 Remaining | Verification of all query flows (factual, refusal, quiz, rewriter) |
| **14** | Final Documentation & Wrap-up | 🔲 Remaining | Final architecture guide and developer documentation |
