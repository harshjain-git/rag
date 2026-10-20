# Evaluation & Verification Guide — Cadet Readiness Advisor

## 1. Overview

The Cadet Readiness Advisor evaluation methodology validates system precision, grounding reliability, and hallucination resistance across both in-corpus and out-of-corpus queries.

---

## 2. Benchmark Evaluation Dataset

The ground-truth evaluation set is defined in `evaluation/evaluation_set.json` and consists of **18 curated evaluation queries**:

- **10 In-Corpus Answerable Queries** covering:
  - APA assessment standards (Cronbach's alpha thresholds, construct validity CFA guidelines, inter-rater reliability metrics).
  - ASVAB norms and subtests (standard score mean/SD, AFQT composite formula, CAT-ASVAB adaptive mechanics).
  - Military psychology (Military Training Mental Toughness Inventory scenarios, operational psychology consultation roles).
- **8 Out-of-Corpus Unanswerable Queries** covering unrelated domains (baking recipes, sports rules, astronomy, gardening, vehicle maintenance) designed to test refusal behavior and anti-hallucination guardrails.

---

## 3. Empirical Vector Similarity Threshold Calibration

The vector similarity threshold determines whether a retrieved chunk from ChromaDB is sufficiently relevant to be passed to the LLM generator.

ChromaDB calculates squared $L^2$ (Euclidean) distance for embeddings:
$$\text{Distance} = \sum (u_i - v_i)^2$$
*(Lower distance indicates higher semantic similarity)*

### Empirical Calibration Matrix

| Query Type | Query Sample | Raw Distance Score | Decision ($Threshold = 0.22$) |
|:---|:---|:---:|:---:|
| **In-Corpus** | ASVAB standard score mean & SD | **0.1309** | **Retained** ($\le 0.22$) |
| **In-Corpus** | AFQT composite subtests | **0.1652** | **Retained** ($\le 0.22$) |
| **In-Corpus** | APA Cronbach's alpha threshold | **0.1970** | **Retained** ($\le 0.22$) |
| **Out-of-Corpus** | Neapolitan pizza dough recipe | **0.2312** | **Filtered Out** ($> 0.22$) |
| **Out-of-Corpus** | How to bake a chocolate cake | **0.2584** | **Filtered Out** ($> 0.22$) |
| **Out-of-Corpus** | FIFA World Cup tournament rules | **0.2741** | **Filtered Out** ($> 0.22$) |

Setting `GEMINI_SIMILARITY_THRESHOLD = 0.22` in `config.py` cleanly separates in-corpus queries (0.13 - 0.197) from out-of-corpus queries ($\ge 0.23$), guaranteeing that out-of-corpus queries produce zero retrieved evidence chunks.

---

## 4. Four-Tier Hallucination Defense

```
User Query
    │
    ▼
[Tier 1: Pre-Retrieval Input Validation]
    • Detects empty, whitespace-only, or non-string inputs.
    • Status: EMPTY_QUERY | Evidence: [] | Grounded: False.
    │
    ▼
[Tier 2: Vector Distance Filtering (Threshold = 0.22)]
    • Chunks with L2 distance > 0.22 are discarded.
    • Out-of-corpus queries yield 0 evidence chunks.
    │
    ▼
[Tier 3: Strict Generation Directives]
    • Prompt instructs Gemini: "Rely EXCLUSIVELY on provided excerpts."
    • If excerpts are empty or insufficient, outputs is_answerable: false.
    • Returns polite domain-scoped refusal without synthetic citations.
    │
    ▼
[Tier 4: Post-Generation Grounding Verification]
    • Verification node independently fact-checks the answer against retrieved chunks.
    • Status: VERIFIED_SUPPORTED, REFUSAL_CONFIRMED, or UNSUPPORTED_HALLUCINATION.
```

---

## 5. Automated Test Suite

The test suite contains **20 unit and integration tests** located in `tests/`:

1. **`tests/test_gemini_retry.py` (9 tests)**:
   - Tests detection of temporary 429 and 503 errors.
   - Tests exponential backoff calculation.
   - Tests parsing of server retry delays (`retry in 7.5 seconds`).
   - Tests immediate fast-fail on daily quota exhaustion (`requests_per_day`).
   - Tests permanent error propagation (400, 401).
2. **`tests/test_grounding_and_hallucination.py` (6 tests)**:
   - Tests empty input handling (`EMPTY_QUERY`).
   - Tests vector distance threshold filtering on out-of-corpus queries.
   - Tests vector distance threshold retention on answerable queries.
   - Tests refusal generation without hallucinated sources for out-of-corpus queries.
   - Tests citation attachment on answerable queries.
   - Tests verification node detection of unsupported hallucinations.
3. **`tests/test_agent_orchestration.py` (5 tests)**:
   - Tests `ToolRegistry` registration, listing, retrieval, and unregistration.
   - Tests `AgentPipeline` empty input handling.
   - Tests terminal early exits (direct text responses).
   - Tests standard workflow sequential execution (`orchestrator` → `retrieval` → `generation` → `verification`).
   - Tests `AgentService` message dictionary formatting.

### Running the Test Suite

```bash
# Run all tests using Python standard unittest
python -m unittest discover -s tests -p "test_*.py"

# Or run individual test modules
python -m unittest tests/test_agent_orchestration.py
python -m unittest tests/test_gemini_retry.py
python -m unittest tests/test_grounding_and_hallucination.py
```
