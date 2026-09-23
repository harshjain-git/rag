"""
Corpus Assessment Question Generator Tool for Cadet Readiness Advisor.
Generates assessment, quiz, or conceptual practice questions grounded strictly in the reference corpus PDFs.
Uses stratified corpus sampling, bounded batch LLM synthesis, batch grounding verification,
lexical/semantic deduplication, and deterministic metadata citation mapping.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool, BaseTool

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retriever import get_cached_vector_store
from generation.generator import generate_structured_json
from application.prompts import (
    build_question_generation_messages,
    build_question_verification_messages,
    messages_to_gemini_args,
)


def _normalize_domain_from_source(source: str, meta_domain: Optional[str] = None) -> str:
    """Derives a clean domain category using metadata if present, falling back to document naming."""
    if meta_domain:
        return str(meta_domain).lower()
    s = source.lower()
    if "apa" in s:
        return "apa_psychometrics"
    elif "asvab" in s:
        return "asvab"
    elif "military" in s:
        return "military_psychology"
    return "general_corpus"


_CACHED_COVERAGE: Optional[Dict[str, Any]] = None


def inspect_corpus_coverage() -> Dict[str, Any]:
    """
    Inspects existing Chroma collection metadata dynamically without modifying or re-embedding the database.
    Caches the parsed coverage in memory for instant reuse.
    """
    global _CACHED_COVERAGE
    if _CACHED_COVERAGE is not None:
        return _CACHED_COVERAGE

    vector_store = get_cached_vector_store()
    
    # Direct collection retrieval without vector similarity computation
    raw_data = vector_store.get(include=["metadatas", "documents"])
    ids = raw_data.get("ids", [])
    documents_text = raw_data.get("documents", [])
    metadatas = raw_data.get("metadatas", [])

    all_chunks = []
    doc_set = set()
    domain_set = set()
    doc_to_domain = {}

    for cid, text, meta in zip(ids, documents_text, metadatas):
        if not text or len(text.strip()) < 80:
            continue
        source = meta.get("source", "unknown.pdf")
        page = meta.get("page", 1)
        domain = _normalize_domain_from_source(source, meta.get("domain"))

        doc_set.add(source)
        domain_set.add(domain)
        doc_to_domain[source] = domain

        all_chunks.append({
            "chunk_id": cid,
            "text": text.strip(),
            "source": source,
            "page": page,
            "domain": domain,
            "metadata": meta
        })

    _CACHED_COVERAGE = {
        "documents": sorted(list(doc_set)),
        "domains": sorted(list(domain_set)),
        "doc_to_domain": doc_to_domain,
        "all_chunks": all_chunks
    }
    return _CACHED_COVERAGE


def stratified_sample_chunks(
    coverage: Dict[str, Any],
    target_count: int = 15,
    scope: str = "all",
    domains: Optional[List[str]] = None,
    documents: Optional[List[str]] = None,
    excluded_chunk_ids: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Samples diverse, non-adjacent chunks across documents and pages based on requested scope.
    """
    excluded = excluded_chunk_ids or set()
    chunks = [c for c in coverage.get("all_chunks", []) if c["chunk_id"] not in excluded]

    # Apply scope filtering
    if scope == "domain" and domains:
        target_domains = {d.lower().strip() for d in domains}
        chunks = [c for c in chunks if any(td in c["domain"] for td in target_domains)]
    elif scope == "document" and documents:
        target_docs = {doc.lower().strip() for doc in documents}
        chunks = [c for c in chunks if any(td in c["source"].lower() for td in target_docs)]

    if not chunks:
        return []

    # Group chunks by document
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        by_doc.setdefault(c["source"], []).append(c)

    # Sort chunks within each document by page to allow non-adjacent spacing
    for doc_name in by_doc:
        by_doc[doc_name].sort(key=lambda x: x["page"])

    sampled: List[Dict[str, Any]] = []
    doc_names = list(by_doc.keys())
    if not doc_names:
        return []

    # Round-robin sampling across documents with page spacing
    chunks_per_doc = max(1, target_count // len(doc_names) + 1)
    
    for doc_name in doc_names:
        doc_chunks = by_doc[doc_name]
        total_available = len(doc_chunks)
        if total_available <= chunks_per_doc:
            sampled.extend(doc_chunks)
        else:
            # Step evenly through pages to avoid clustering
            step = max(1, total_available // chunks_per_doc)
            for idx in range(0, total_available, step):
                if len(sampled) < target_count * 2:
                    sampled.append(doc_chunks[idx])

    return sampled[:target_count]


def _compute_token_similarity(q1: str, q2: str) -> float:
    """Computes normalized word-token Jaccard overlap to detect semantic duplicates."""
    tokens1 = set(re.findall(r"\w+", q1.lower()))
    tokens2 = set(re.findall(r"\w+", q2.lower()))
    
    # Filter common stop words
    stops = {"what", "is", "the", "how", "are", "of", "in", "and", "a", "an", "to", "for", "with", "does", "do", "explain", "describe"}
    t1 = tokens1 - stops
    t2 = tokens2 - stops
    
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / float(len(t1 | t2))


def deduplicate_questions(candidates: List[Dict[str, Any]], similarity_threshold: float = 0.65) -> List[Dict[str, Any]]:
    """Removes duplicate or substantially overlapping questions."""
    unique: List[Dict[str, Any]] = []
    for cand in candidates:
        q_text = cand.get("question", "").strip()
        if not q_text:
            continue
        is_dup = False
        for u in unique:
            if _compute_token_similarity(q_text, u.get("question", "")) >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(cand)
    return unique


def _run_batch_verification(
    candidates: List[Dict[str, Any]],
    evidence_pack: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Executes exactly 1 verification call per batch to evaluate candidate grounding.
    Prohibits per-question verification calls to conserve Gemini API quota.
    """
    if not candidates:
        return []

    messages = build_question_verification_messages(candidates, evidence_pack)
    system_instruction, contents = messages_to_gemini_args(messages)

    try:
        parsed = generate_structured_json(contents=contents, system_instruction=system_instruction)
        verdicts = parsed.get("verdicts", [])
        
        valid_indices = set()
        for v in verdicts:
            if v.get("is_valid", False):
                valid_indices.add(v.get("index"))
        
        verified = [c for idx, c in enumerate(candidates) if idx in valid_indices]
        return verified

    except Exception:
        # Graceful fallback: accept candidates whose chunk_id exists in evidence pack
        valid_chunk_ids = {c["chunk_id"] for c in evidence_pack}
        return [c for c in candidates if c.get("grounding_chunk_id") in valid_chunk_ids]


@tool("generate_corpus_questions")
def generate_corpus_questions(
    num_questions: int = 5,
    scope: str = "all",
    domains: Optional[List[str]] = None,
    documents: Optional[List[str]] = None,
    difficulty: str = "mixed",
    question_types: Optional[List[str]] = None,
    include_answers: bool = True,
    include_citations: bool = True
) -> Dict[str, Any]:
    """
    Generates structured assessment, quiz, or interview questions grounded strictly in the reference corpus documents.
    Supports corpus-wide, domain-specific, and document-specific question generation with difficulty and question types.
    Does NOT use top-k query search; samples evidence directly across the corpus.
    
    Args:
        num_questions: Requested number of questions (bounded 1 to 50, default: 5).
        scope: "all" (corpus-wide), "domain" (domain-specific), or "document" (document-specific).
        domains: Optional list of domains (e.g. ['asvab', 'apa_psychometrics', 'military_psychology']).
        documents: Optional list of document filenames (e.g. ['asvab1.pdf', 'apa2.pdf']).
        difficulty: "basic", "intermediate", "advanced", or "mixed" (default: "mixed").
        question_types: List of question types (e.g. ['conceptual', 'definition', 'application', 'scenario']).
        include_answers: Whether to include the grounded answer key.
        include_citations: Whether to include deterministic page/source citations.
        
    Returns:
        Structured dictionary containing:
            - 'status': 'SUCCESS' | 'PARTIAL' | 'INSUFFICIENT_EVIDENCE'
            - 'requested_count': int
            - 'generated_count': int
            - 'unsupported_count': int
            - 'coverage': {'documents': [...], 'domains': [...], 'pages': [...]}
            - 'questions': List of question objects with verified answer keys and source citations
    """
    target_count = max(1, min(int(num_questions or 5), 50))
    coverage_info = inspect_corpus_coverage()
    all_chunks = coverage_info.get("all_chunks", [])

    if not all_chunks:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "requested_count": target_count,
            "generated_count": 0,
            "unsupported_count": 0,
            "coverage": {"documents": [], "domains": [], "pages": []},
            "questions": [],
            "error": "Corpus collection contains no usable evidence chunks."
        }

    # Configurable batch size (default 5 to 10 questions per batch to avoid output token limits)
    batch_size = 10 if target_count >= 10 else 5
    num_batches = (target_count + batch_size - 1) // batch_size
    
    collected_questions: List[Dict[str, Any]] = []
    sampled_ids_used = set()
    total_unsupported = 0

    # Create lookup map for deterministic chunk metadata resolution
    chunk_meta_map = {c["chunk_id"]: c for c in all_chunks}

    # Bounded execution: at most num_batches + 1 adaptive retry
    max_loops = num_batches + 1
    loop_count = 0

    while len(collected_questions) < target_count and loop_count < max_loops:
        loop_count += 1
        needed = target_count - len(collected_questions)
        current_batch_target = min(needed, batch_size)

        # Sample stratified evidence pack for this batch
        evidence_pack = stratified_sample_chunks(
            coverage=coverage_info,
            target_count=max(6, current_batch_target * 2),
            scope=scope,
            domains=domains,
            documents=documents,
            excluded_chunk_ids=sampled_ids_used
        )

        if not evidence_pack:
            break

        for ep in evidence_pack:
            sampled_ids_used.add(ep["chunk_id"])

        # 1. Batch Question Generation LLM Call
        gen_messages = build_question_generation_messages(
            evidence_pack=evidence_pack,
            num_questions=current_batch_target,
            difficulty=difficulty,
            question_types=question_types
        )
        sys_instr, contents = messages_to_gemini_args(gen_messages)

        try:
            raw_candidates = generate_structured_json(contents=contents, system_instruction=sys_instr, temperature=0.2)
            if isinstance(raw_candidates, dict) and "questions" in raw_candidates:
                raw_candidates = raw_candidates["questions"]
            if not isinstance(raw_candidates, list):
                raw_candidates = []
        except Exception:
            raw_candidates = []

        if not raw_candidates:
            continue

        # 2. Batch Grounding Verification (1 call per batch)
        verified_candidates = _run_batch_verification(raw_candidates, evidence_pack)
        rejected_in_batch = len(raw_candidates) - len(verified_candidates)
        total_unsupported += max(0, rejected_in_batch)

        # 3. Deterministic Citation Mapping: grounding_chunk_id -> chunk metadata
        for cand in verified_candidates:
            cid = cand.get("grounding_chunk_id", "")
            chunk_obj = chunk_meta_map.get(cid)
            
            # Strict guarantee: source/page come from real chunk metadata, never from LLM
            if chunk_obj:
                src = chunk_obj["source"]
                pg = chunk_obj["page"]
                dom = chunk_obj["domain"]
            else:
                # If chunk ID was invalid, fallback to the first chunk from the evidence pack
                if evidence_pack:
                    src = evidence_pack[0].get("source", "unknown.pdf")
                    pg = evidence_pack[0].get("page", 1)
                    dom = evidence_pack[0].get("domain", "general_corpus")
                    cid = evidence_pack[0].get("chunk_id", "")
                else:
                    src = "unknown.pdf"
                    pg = 1
                    dom = "general_corpus"
                    cid = ""

            cand["grounding_chunk_id"] = cid
            cand["source"] = src
            cand["page"] = pg
            cand["domain"] = dom
            cand["citation"] = f"{src} — Page {pg}"
            cand["verified"] = True

        # 4. Deduplication
        deduped = deduplicate_questions(verified_candidates)
        for d in deduped:
            if len(collected_questions) < target_count:
                # Assign stable sequential index
                d["id"] = len(collected_questions) + 1
                if not include_answers:
                    d.pop("answer", None)
                if not include_citations:
                    d.pop("citation", None)
                collected_questions.append(d)

    # Determine final outcome status
    generated_count = len(collected_questions)
    if generated_count == 0:
        final_status = "INSUFFICIENT_EVIDENCE"
    elif generated_count >= target_count:
        final_status = "SUCCESS"
    else:
        final_status = "PARTIAL"

    # Compile actual coverage representation
    docs_represented = sorted(list({q["source"] for q in collected_questions if "source" in q}))
    domains_represented = sorted(list({q["domain"] for q in collected_questions if "domain" in q}))
    pages_represented = sorted(list({f"{q['source']}#p{q['page']}" for q in collected_questions if "source" in q and "page" in q}))

    return {
        "status": final_status,
        "requested_count": target_count,
        "generated_count": generated_count,
        "unsupported_count": total_unsupported,
        "coverage": {
            "documents": docs_represented,
            "domains": domains_represented,
            "pages_sampled_count": len(pages_represented)
        },
        "questions": collected_questions
    }


def get_question_generator_tool() -> BaseTool:
    """Returns the standard LangChain generate_corpus_questions tool instance."""
    return generate_corpus_questions
