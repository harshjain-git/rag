"""
Cadet Readiness Advisor — Streamlit Web Interface (Phase 11)
Interactive RAG application for psychometric, ASVAB, and psychological reference.
"""

import sys
import os
import warnings
import logging
import base64
from pathlib import Path

# Suppress noisy library warnings and HuggingFace Hub messages
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pymupdf

import config
from retrieval.retriever import get_cached_vector_store
from agent import get_agent_manager

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Cadet Readiness Advisor",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge-grounded {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-not-corpus {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .chunk-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data
def get_document_catalog():
    """Returns list of documents in data/raw with metadata and page counts."""
    raw_dir = config.DATA_RAW_DIR
    catalog = []
    if not raw_dir.exists():
        return catalog

    for file_path in sorted(raw_dir.glob("*.pdf")):
        try:
            doc = pymupdf.open(file_path)
            num_pages = len(doc)
            doc.close()
        except Exception:
            num_pages = 0

        catalog.append({
            "name": file_path.name,
            "path": file_path,
            "size_kb": round(file_path.stat().st_size / 1024, 1),
            "pages": num_pages
        })
    return catalog


@st.cache_data
def get_pdf_page_image(pdf_path_str: str, page_number: int):
    """Renders a specific PDF page as high-res PNG image bytes using PyMuPDF."""
    try:
        doc = pymupdf.open(pdf_path_str)
        if page_number < 1 or page_number > len(doc):
            return None
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        return None


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# # Pre-load vector store once in background
# with st.spinner("Initializing Vector Store & Embedding Model..."):
#     get_cached_vector_store()


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/military-medal.png", width=64)
    st.title("Advisor System")
    st.caption("Psychometric & Assessment Reference")

    st.divider()

    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown(f"""
    - **Architecture**: `LangChain Agentic Pipeline`
    - **Orchestration**: `LCEL Runnable Sequence`
    - **LLM**: `{config.LLM_MODEL_NAME}`
    - **Embeddings**: `bge-base-en-v1.5`
    - **Similarity Threshold**: `≤ {config.SIMILARITY_THRESHOLD}`
    - **Top-K Retrieval**: `{config.INITIAL_TOP_K}`
    - **Vector DB**: `ChromaDB` (Persistent)
    """)

    st.divider()

    st.markdown("### 💡 Quick Sample Queries")
    sample_queries = [
        "What is the ASVAB aptitude test and its components?",
        "How do standard scores relate to percentile ranks in ASVAB?",
        "What are the core dimensions of psychological resilience in cadets?",
        "What is quantum thermodynamics in black holes?"  # Out-of-corpus test
    ]
    for sq in sample_queries:
        if st.button(f"📌 {sq[:38]}...", help=sq, width="stretch"):
            st.session_state.current_prompt = sq

    st.divider()
    if st.button("🗑️ Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --- Header ---
st.markdown('<div class="main-header">🎖️ Cadet Readiness Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strictly grounded reference assistant powered by Gemini Flash Lite, BGE embeddings, and ChromaDB.</div>', unsafe_allow_html=True)

# --- Top Navigation ---
nav_selection = st.segmented_control(
    "Navigation",
    options=["💬 Ask Advisor", "📚 Document Library (PDF Viewer)"],
    default="💬 Ask Advisor",
    label_visibility="collapsed"
)

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)


# ==========================================
# VIEW 1: ASK ADVISOR (Chat & Evidence)
# ==========================================
if nav_selection == "💬 Ask Advisor":
    # 1. Check if a sample query was clicked in the sidebar
    if "current_prompt" in st.session_state and st.session_state.current_prompt:
        user_query = st.session_state.current_prompt
        st.session_state.current_prompt = None
    else:
        user_query = None

    # 2. Sticky bottom chat input widget (pinned to bottom of viewport)
    chat_input_text = st.chat_input("Ask a question regarding ASVAB, psychological resilience, or psychometric standards...")
    if chat_input_text:
        user_query = chat_input_text

    # 3. If there is a new query to process, run it and append to state
    if user_query:
        # Capture conversation history prior to current turn
        prior_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("Executing LangChain Agent (Query Resolution → Retrieval Tool → Grounded Generation → Verification)..."):
            agent_mgr = get_agent_manager()
            response_data = agent_mgr.run(user_query, history=prior_history)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data.get("answer", ""),
            "original_query": response_data.get("query", user_query),
            "resolved_query": response_data.get("resolved_query"),
            "resolution_action": response_data.get("resolution_action", "KEEP"),
            "resolution_reason": response_data.get("resolution_reason", ""),
            "is_grounded": response_data.get("is_grounded", False),
            "is_answerable": response_data.get("is_answerable", response_data.get("is_grounded", False)),
            "is_verified": response_data.get("is_verified", False),
            "verification_status": response_data.get("verification_status", ""),
            "verification_details": response_data.get("verification_details", ""),
            "status": response_data.get("status", ""),
            "tools_used": response_data.get("tools_used", []),
            "top_score": response_data.get("top_score"),
            "evidence": response_data.get("evidence", [])
        })

    # 4. Render all conversation messages in sequence
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                is_grounded = msg.get("is_grounded", False)
                is_verified = msg.get("is_verified", False)
                verification_status = msg.get("verification_status", "")
                tools_used = msg.get("tools_used", [])
                top_score = msg.get("top_score")
                status_code = msg.get("status", "GROUNDED")
                
                # Grounding / Answerability and Verification Badges
                col_b1, col_b2 = st.columns([1, 1])
                with col_b1:
                    if is_grounded:
                        score_txt = f" (Distance: {top_score:.4f})" if top_score is not None else ""
                        st.markdown(f'<span class="status-badge-grounded">✅ ANSWERABLE & GROUNDED{score_txt}</span>', unsafe_allow_html=True)
                    else:
                        score_txt = f" (Nearest Doc Distance: {top_score:.4f})" if top_score is not None else ""
                        st.markdown(f'<span class="status-badge-not-corpus">⚠️ INSUFFICIENT EVIDENCE / NOT IN CORPUS{score_txt}</span>', unsafe_allow_html=True)
                
                with col_b2:
                    if verification_status == "REFUSAL_CONFIRMED":
                        st.markdown(f'<span class="status-badge-grounded">🛡️ VERIFIED: Refusal Confirmed</span>', unsafe_allow_html=True)
                    elif is_verified:
                        st.markdown(f'<span class="status-badge-grounded">🛡️ VERIFIED: Factually Supported</span>', unsafe_allow_html=True)
                    elif verification_status:
                        st.markdown(f'<span class="status-badge-not-corpus">⚠️ VERIFICATION: {verification_status}</span>', unsafe_allow_html=True)

                if tools_used:
                    st.caption(f"🔧 **Tools Executed**: `{'`, `'.join(tools_used)}`")

                # Answer Text
                st.markdown(msg["content"])
                
                # Retrieved Evidence Chunks & Query Analysis Expander
                evidence_list = msg.get("evidence", [])
                resolved_q = msg.get("resolved_query")
                orig_q = msg.get("original_query")
                action = msg.get("resolution_action", "KEEP")
                reason = msg.get("resolution_reason", "")
                if evidence_list or resolved_q:
                    with st.expander(f"🔍 Inspect Query Analysis & Retrieved Chunks ({len(evidence_list)} chunks evaluated by agent)"):
                        if resolved_q:
                            action_icon = "🟢" if action == "KEEP" else "🔄"
                            st.markdown(f"##### 🎯 Query Resolution: {action_icon} `{action}`")
                            st.markdown(f"- **Decision Reason**: {reason}")
                            st.markdown(f"- **Original Query**: `{orig_q or 'N/A'}`")
                            if action == "REWRITE":
                                st.markdown(f"- **Reformulated Retrieval Query**: `{resolved_q}`")
                            st.divider()

                        # Retrieval & Answerability Overview
                        st.markdown("##### 📊 Retrieval & Answerability Overview")
                        st.markdown(f"- **Retrieval Query**: `{resolved_q or orig_q}`")
                        st.markdown(f"- **Answerability Status**: `{'ANSWERABLE (GROUNDED)' if is_grounded else 'INSUFFICIENT_EVIDENCE'}`")
                        if top_score is not None:
                            st.markdown(f"- **Nearest Chunk Distance**: `{top_score:.4f}`")
                        st.markdown(f"- **Retrieved Chunks Count**: `{len(evidence_list)} chunks`")
                        st.divider()

                        st.markdown("##### 📄 Retrieved Evidence Chunks (Debug)")
                        for idx, chunk in enumerate(evidence_list, 1):
                            st.markdown(f"**Chunk #{idx}** — `ID: {chunk.get('chunk_id', 'N/A')}`")
                            st.caption(f"📄 Source: **{chunk.get('source')}** | Page: **{chunk.get('page')}** | Distance: **{chunk.get('score', 0):.4f}**")
                            st.code(chunk.get("text", "").strip(), language="text")
                            st.markdown("---")

    # Extra bottom space so the last message is never covered by the fixed bottom input bar
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)


# ==========================================
# VIEW 2: DOCUMENT LIBRARY (PDF Viewer)
# ==========================================
elif nav_selection == "📚 Document Library (PDF Viewer)":
    st.subheader("📚 Corpus Document Library")
    st.caption("Browse, inspect, or download any of the reference PDF files in the corpus.")

    catalog = get_document_catalog()

    if not catalog:
        st.warning("No PDF documents found in `data/raw/`.")
    else:
        # Layout: Left column file selector & info, right column viewer
        col_list, col_viewer = st.columns([1, 2], gap="medium")

        with col_list:
            st.markdown("### Available Documents")
            doc_names = [d["name"] for d in catalog]
            selected_doc_name = st.selectbox("Select document to inspect:", doc_names)

            # Find selected doc info
            selected_doc = next(d for d in catalog if d["name"] == selected_doc_name)

            st.markdown(f"""
            - **Filename**: `{selected_doc['name']}`
            - **Total Pages**: `{selected_doc['pages']}`
            - **File Size**: `{selected_doc['size_kb']} KB`
            - **Path**: `data/raw/{selected_doc['name']}`
            """)

            # Download button
            with open(selected_doc["path"], "rb") as f:
                pdf_data = f.read()

            st.download_button(
                label=f"⬇️ Download {selected_doc['name']}",
                data=pdf_data,
                file_name=selected_doc["name"],
                mime="application/pdf",
                width="stretch"
            )

            st.divider()
            st.markdown("### 📊 Corpus Summary")
            total_pages = sum(d["pages"] for d in catalog)
            total_size_mb = sum(d["size_kb"] for d in catalog) / 1024
            st.metric("Total Documents", len(catalog))
            st.metric("Total Pages", total_pages)
            st.metric("Total Corpus Size", f"{total_size_mb:.2f} MB")

        with col_viewer:
            st.markdown(f"### 📄 Document Viewer: `{selected_doc_name}`")
            total_doc_pages = selected_doc["pages"]
            
            if total_doc_pages > 0:
                c_slider, c_info = st.columns([3, 1])
                with c_slider:
                    current_page = st.slider("Page Navigator:", min_value=1, max_value=total_doc_pages, value=1, step=1)
                with c_info:
                    st.markdown(f"<div style='margin-top: 28px; font-weight: 600;'>Page {current_page} of {total_doc_pages}</div>", unsafe_allow_html=True)

                page_img = get_pdf_page_image(str(selected_doc["path"]), current_page)
                if page_img:
                    st.image(
                        page_img,
                        caption=f"{selected_doc_name} — Page {current_page} / {total_doc_pages}",
                        width="stretch"
                    )
                else:
                    st.warning("Unable to render page.")
            else:
                st.info("This document does not contain readable pages.")
