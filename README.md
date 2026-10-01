# Cadet Readiness Advisor - RAG System

A Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **PyMuPDF**, **ChromaDB**, and **Gemini Flash Lite** (`gemini-2.5-flash-lite`) for Cadet Readiness Assessment and Guidance.

## Project Structure

```
cadet-readiness-advisor/
│
├── data/
│   └── raw/             # Raw PDF documents and data files
│
├── ingestion/           # Data loading, parsing, cleaning, and LangChain text chunking
├── retrieval/           # LangChain Vector store querying & context retrieval
├── evaluation/          # RAG assessment, ground truth benchmarking, and metrics
├── app/                 # Streamlit web application layer
│
├── config.py            # Central configuration & parameters
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation
```


## Setup Instructions

1. Place source PDF documents into `data/raw/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run ingestion and launch application modules:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## System Architecture

```
User Query
    │
    ▼
Streamlit UI (app/streamlit_app.py)
    │
    ▼
AgentService (application/services.py)
    │
    ▼
AgentPipeline (agent/pipeline.py)
    │
    ▼
ToolRegistry (core/registry.py)
    │
    ▼
Orchestrator Step (agent/steps.py)
    │
    ├── [Gemini Native Tool Calling] ──► GeminiLLM.generate_with_tools()
    │
    ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
    │                                                                                     │
    ▼                                                                                     ▼
[Document Corpus Flow]                                                         [Structured Database Flow]
`retrieve_corpus_evidence`                                                     `ask_srag_database`
    │                                                                                     │
    ▼                                                                                     ▼
ChromaDB Vector Store (infrastructure/chroma.py)                               A2A Adapter (infrastructure/a2a_adapter.py)
    │ (Top-K Chunks)                                                                      │ (HTTP / JSON-RPC Agent-to-Agent)
    ▼                                                                                     ▼
`generate_answer` (agent/steps.py)                                             SRAG Server (srag_mcp/a2a_server.py on port 8001)
    │ (Grounded Answer + Citations)                                                       │
    ▼                                                                                     ▼
`verify_grounding` (agent/verification.py)                                     Model Context Protocol (MCP)
    │ (Strict Factual Verification Audit)                                                 │
    ▼                                                                                     ▼
Streamlit UI (Cards, Badges, Citations, Expander)                              PostgreSQL Database
```

### Core Architecture Components

1. **RAG Agent (`agent/pipeline.py` & `core/registry.py`):**
   * Registry-driven, provider-agnostic agent orchestration.
   * State-aware execution pipeline operating on `AgentState` dictionaries.
2. **Gemini Native Tool Calling (`infrastructure/gemini.py` & `agent/tool_declarations.py`):**
   * Uses `google.genai` SDK native tool declarations (`FunctionDeclaration`) to autonomously route user queries without forcing artificial JSON schemas.
   * Returns function call specifications or direct conversational responses. Automatic function execution is disabled so Python maintains full lifecycle control over `AgentState`.
3. **Document Retrieval Flow (`retrieve -> generate -> verify`):**
   * For document queries, the orchestrator routes to `retrieve_corpus_evidence` (ChromaDB similarity search).
   * Downstream generation synthesizes grounded answers with strict citation anchors.
   * A verification step audits every claim against the retrieved text to prevent hallucinations.
4. **Agent-to-Agent (A2A) Connection to SRAG (`infrastructure/a2a_adapter.py`):**
   * Database queries route to `ask_srag_database`.
   * A thread-safe synchronous wrapper connects to the remote Structured RAG (SRAG) microservice via the standard A2A protocol over HTTP.
5. **MCP → PostgreSQL Flow (`srag_mcp/`):**
   * The remote SRAG agent receives queries via A2A, translates natural language into database queries through the Model Context Protocol (MCP), and queries PostgreSQL.
   * No direct PostgreSQL or MCP dependencies exist in the main RAG codebase.
6. **Production Native Orchestration:**
   * Autonomous routing powered solely by Gemini Native Tool Calling.
   * Legacy prompted JSON routing and JSON schema parsing have been completely decommissioned for maximum reliability, speed, and clean separation of concerns.
