# Agentic Migration Plan

## Project

Cadet Readiness Advisor — RAG-based Psychometric Reference Assistant

## Purpose

Incrementally convert the existing RAG pipeline into an agentic architecture using the existing LangChain ecosystem, while preserving the current working application.

The migration must be performed one step at a time.

## Critical Rules

1. Implement ONLY the currently requested step.
2. NEVER automatically proceed to the next step.
3. Preserve the existing working RAG pipeline.
4. Do not rewrite working code unnecessarily.
5. Do not change existing behavior unless the current step explicitly requires it.
6. Do not add new functionality during the migration unless explicitly requested.
7. Keep the application runnable after every step.
8. Before modifying code, inspect the existing implementation and understand how the relevant stage currently works.
9. Reuse existing functions and components instead of duplicating them.
10. After completing each step, explain exactly what changed and STOP.
11. Wait for explicit user approval before starting the next step.
12. If a proposed change could affect existing behavior, explain the risk before making it.
13. Use the current official LangChain documentation and recommended APIs as the implementation reference (pure LangChain tools, agents, and runnables without LangGraph); do not invent or use outdated agent/tool patterns when an official approach exists.

## Target Architecture

The final direction is:

User
  ↓
Agent / Orchestrator
  ↓
Tools / Nodes
  ↓
Existing RAG capabilities
  ↓
Grounded Answer

The goal is NOT to rewrite the RAG system. The goal is to gradually expose the existing capabilities as tools/nodes and orchestrate them through an agentic workflow.

## Migration Steps

### Step 1 — Agentic Foundation

Create the basic agentic structure.

- Inspect the current project.
- Identify the current RAG stages.
- Create the agentic module/structure.
- Define the initial state.
- Create the initial agent/graph foundation.
- Do NOT replace the existing application flow yet.
- Do NOT modify retrieval or generation logic.

Status: COMPLETED

### Step 2 — Existing Retrieval as a Tool

Take the existing retrieval functionality and expose it as an agent tool.

- Reuse the existing Chroma/retriever implementation.
- Do not rewrite the retrieval implementation.
- Do not change embeddings.
- Do not change retrieval behavior unnecessarily.
- Test that the tool returns the same evidence as the existing retrieval pipeline.

Status: COMPLETED

### Step 3 — Connect Retrieval Tool to the Agent

Connect the retrieval tool to the agent/graph.

- User query enters the agent.
- Agent can call the existing retrieval tool.
- Retrieved evidence is placed into the agent state.
- Do not change answer generation yet.

Status: COMPLETED

### Step 4 — Connect Existing Generation

Connect the existing Gemini generation logic to the agentic workflow.

- Reuse the existing generation implementation and grounding prompt.
- Do not redesign the prompt unless explicitly requested.
- Preserve current grounded-answer behavior.
- The agent should use retrieved evidence to generate the answer.

Status: COMPLETED

### Step 5 — Add Verification / Grounding Node

Add a separate verification stage after generation.

- Check whether the generated answer is supported by the available evidence.
- Do not invent missing evidence.
- Keep this isolated from the existing retrieval implementation.

Status: COMPLETED

### Step 6 — Connect the Agentic Workflow to Streamlit

Only after the agentic pipeline works independently:

- Connect Streamlit to the new workflow.
- Preserve the existing UI behavior as much as possible.
- Replace the old direct pipeline only after the new pipeline is tested.

Status: COMPLETED

### Step 7 — Regression Testing

Compare the original and agentic pipelines.

Test:
- normal in-corpus questions
- partially answerable questions
- out-of-corpus questions
- citation/source behavior
- retrieval results
- generated answers

The agentic migration should not silently reduce existing functionality.

Status: COMPLETED

### Step 8+ — Future Capabilities

Only after the basic agentic pipeline is stable, consider additional capabilities such as:

- selected PDF handling
- selected paragraph/text handling
- document summarization
- multi-document analysis
- question generation
- comparison workflows
- additional retrieval strategies
- additional tools

These are FUTURE capabilities and must NOT be implemented during Steps 1–7 unless explicitly requested.

Status: NOT STARTED

## Current Progress

Completed:
- Existing RAG application
- Existing Chroma retrieval
- Existing Gemini integration
- Step 1 — Agentic Foundation
- Step 2 — Existing Retrieval as a Tool
- Step 3 — Connect Retrieval Tool to the Agent
- Step 4 — Connect Existing Generation
- Step 5 — Add Verification / Grounding Node
- Step 6 — Connect the Agentic Workflow to Streamlit
- Step 7 — Regression Testing

Current step:
- Completed (Core Migration Finished)

Next step:
- Step 8+ — Future Capabilities (Optional future extensions)

Do not implement the next step automatically.

## Required Behavior for the Coding Agent

Before every task:

1. Read this file.
2. Check the CURRENT STEP.
3. Inspect the existing code relevant to that step.
4. Make only the changes required for that step.
5. Do not implement future steps.
6. Report changed files and explain the implementation.
7. Stop and wait for explicit user approval.
