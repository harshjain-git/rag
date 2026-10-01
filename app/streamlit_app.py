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
from application.services import get_agent_service

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
    /* Prevent page dimming/blurring during rerun/execution */
    div[data-testid="stAppViewBlockContainer"],
    div[data-testid="stAppViewContainer"],
    .stApp {
        opacity: 1 !important;
        filter: none !important;
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


# --- Modal Dialog for PDF Preview ---
if hasattr(st, "dialog"):
    @st.dialog("📄 PDF Document Viewer", width="large")
    def show_pdf_preview(doc_info: dict):
        with open(doc_info["path"], "rb") as f:
            pdf_bytes = f.read()

        c_info, c_dl = st.columns([3, 1])
        with c_info:
            st.markdown(f"#### 📄 `{doc_info['name']}`")
            st.caption(f"📑 **{doc_info['pages']} pages** • 💾 **{doc_info['size_kb']} KB** • *Continuous vertical scroll*")
        with c_dl:
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=doc_info["name"],
                mime="application/pdf",
                key=f"dlg_dl_btn_{doc_info['name']}",
                width="stretch"
            )

        # Native vector PDF rendering with guaranteed 750px height so the full document is clearly visible
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = (
            f'<div style="width: 100%; height: 750px; border-radius: 8px; overflow: hidden; border: 1px solid #334155; background-color: #0f172a;">'
            f'<iframe src="data:application/pdf;base64,{b64_pdf}#toolbar=1&navpanes=0&view=FitH" '
            f'width="100%" height="750px" allowfullscreen="true" '
            f'style="border: none; width: 100%; height: 750px; display: block;"></iframe>'
            f'</div>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.caption("💡 *Scroll down through pages with your mouse or trackpad. Text is crisp vector quality and can be selected and copied.*")


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chat-message.png", width=64)
    st.title("Advisor System")
    st.caption("Psychometric & Assessment Reference")

    st.divider()

    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown(f"""
    - **Architecture**: `LangChain Agentic Pipeline`
    - **Orchestration**: `LCEL Runnable Sequence`
    - **LLM**: `{config.LLM_MODEL_NAME}`
    - **Embeddings**: `{config.GEMINI_EMBEDDING_MODEL if config.EMBEDDING_PROVIDER == 'gemini' else config.EMBEDDING_MODEL_NAME}`
    - **Similarity Threshold**: `≤ {config.SIMILARITY_THRESHOLD}`
    - **Top-K Retrieval**: `{config.INITIAL_TOP_K}`
    - **Vector DB**: `ChromaDB` (Persistent)
    """)

    st.divider()

    catalog = get_document_catalog()
    doc_count = len(catalog)

    # 1. Collapsible Reference PDFs (clean, no hovering black tooltips)
    with st.expander(f"📚 Reference PDFs ({doc_count} files)", expanded=False):
        st.caption("Available corpus reference documents:")
        if not catalog:
            st.info("No PDF documents found in `data/raw/`.")
        else:
            for doc in catalog:
                c_name, c_view, c_dl = st.columns([3, 1, 1])
                with c_name:
                    st.markdown(
                        f"<div style='font-size: 0.85rem; font-weight: 500; padding-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>📄 {doc['name']}</div>",
                        unsafe_allow_html=True
                    )
                with c_view:
                    if st.button("👁️", key=f"sb_view_{doc['name']}"):
                        if hasattr(st, "dialog"):
                            show_pdf_preview(doc)
                with c_dl:
                    with open(doc["path"], "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="⬇️",
                        data=pdf_data,
                        file_name=doc["name"],
                        mime="application/pdf",
                        key=f"sb_dl_{doc['name']}"
                    )
                st.markdown("<div style='margin-bottom: 2px;'></div>", unsafe_allow_html=True)

    # 2. Collapsible Corpus Summary
    total_pages = sum(d["pages"] for d in catalog)
    total_size_mb = sum(d["size_kb"] for d in catalog) / 1024

    with st.expander("📊 Corpus Summary", expanded=False):
        st.caption("Overview & scope of indexed knowledge base:")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.metric("Total Documents", doc_count)
            st.metric("Total Pages", total_pages)
        with c_s2:
            st.metric("Total Size", f"{total_size_mb:.2f} MB")
            st.metric("Indexed Chunks", "999 chunks")

        st.divider()
        st.markdown("""
        **Document Scope**:
        - 📘 **ASVAB Standards**: `asvab1.pdf`, `asvab2.pdf`
        - 📗 **APA Psychometrics**: `apa1.pdf`, `apa2.pdf`, `apa3.pdf`
        - 📙 **Military Psychology**: `military_psyc1.pdf`, `military_psyc2.pdf`
        """)

    st.divider()
    if st.button("🗑️ Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --- Header ---
st.markdown('<div class="main-header">🎖️ Cadet Readiness Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strictly grounded reference assistant powered by Gemini Flash Lite, BGE embeddings, and ChromaDB.</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)


# Helper to render the rich assistant response (badges, answer, cards, evidence)
def render_assistant_card(msg: dict):
    is_grounded = msg.get("is_grounded", False)
    is_verified = msg.get("is_verified", False)
    verification_status = msg.get("verification_status", "")
    tools_used = msg.get("tools_used", [])
    top_score = msg.get("top_score")
    
    # Grounding / Answerability and Verification Badges (rendered ONLY for grounded answers)
    if is_grounded:
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            score_txt = f" (Distance: {top_score:.4f})" if top_score is not None else ""
            st.markdown(f'<span class="status-badge-grounded">✅ ANSWERABLE & GROUNDED{score_txt}</span>', unsafe_allow_html=True)
        with col_b2:
            if is_verified:
                st.markdown(f'<span class="status-badge-grounded">🛡️ VERIFIED: Factually Supported</span>', unsafe_allow_html=True)
            elif verification_status:
                st.markdown(f'<span class="status-badge-not-corpus">⚠️ VERIFICATION: {verification_status}</span>', unsafe_allow_html=True)

        if tools_used:
            st.caption(f"🔧 **Tools Executed**: `{'`, `'.join(tools_used)}`")

    # Answer Text
    st.markdown(msg.get("content", ""))

    # Render Generated Corpus Questions Cards
    questions_list = msg.get("questions", [])
    if questions_list:
        gen_meta = msg.get("generation_metadata", {})
        cov = gen_meta.get("coverage", {})
        docs_cov = cov.get("documents", [])
        domains_cov = cov.get("domains", [])
        
        st.markdown(f"#### 🎯 Generated Assessment Items ({len(questions_list)} Questions)")
        if domains_cov:
            st.caption(f"📁 **Domains**: `{'`, `'.join(domains_cov)}` | 📄 **Documents Represented**: `{'`, `'.join(docs_cov)}`")
        st.divider()

        for q in questions_list:
            qid = q.get("id", "")
            qtext = q.get("question", "")
            qdiff = q.get("difficulty", "intermediate").capitalize()
            qtype = q.get("question_type", "conceptual").capitalize()
            qdom = q.get("domain", "Corpus").replace("_", " ").title()
            qsrc = q.get("source", "unknown.pdf")
            qpage = q.get("page", 1)
            qans = q.get("answer", "")
            qcid = q.get("grounding_chunk_id", "")

            st.markdown(f"##### **Q{qid}: {qtext}**")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.caption(f"🏷️ **Domain**: {qdom}")
            with c2:
                st.caption(f"⚡ **Type**: {qtype}")
            with c3:
                st.caption(f"📊 **Level**: {qdiff}")

            if qans:
                with st.expander(f"💡 View Grounded Answer & Source Reference ({qsrc} - Page {qpage})"):
                    st.markdown(f"**Expected Answer / Evaluation Rubric:**\n\n{qans}")
                    st.markdown(f"---\n📄 **Source Citation**: `{qsrc} — Page {qpage}` *(Chunk ID: `{qcid}`)*")
            st.markdown("<br>", unsafe_allow_html=True)
    
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


# ==========================================
# MAIN INTERFACE: ASK ADVISOR (Chat & Evidence)
# ==========================================
# 1. Render all prior conversation messages FIRST so screen is never blank
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render_assistant_card(msg)

# 2. Check if a sample query was clicked in session state
if "current_prompt" in st.session_state and st.session_state.current_prompt:
    user_query = st.session_state.current_prompt
    st.session_state.current_prompt = None
else:
    user_query = None

# 3. Sticky bottom chat input widget
chat_input_text = st.chat_input("Ask a question regarding ASVAB, psychological resilience, or psychometric standards...")
if chat_input_text:
    user_query = chat_input_text

# 4. If a new query is submitted, display question immediately and stream progress
if user_query:
    prior_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Immediately render the user's message so it NEVER disappears!
    with st.chat_message("user"):
        st.write(user_query)

    # Immediately open the assistant bubble with natural dynamic status updates
    with st.chat_message("assistant"):
        agent_service = get_agent_service()
        if hasattr(st, "status"):
            with st.status("🔍 Searching corpus documents and analyzing...", expanded=True) as status_box:
                st.write("• Consulting psychometric & ASVAB reference documents...")
                assistant_msg = agent_service.query(user_query, history=prior_history)
                st.write("• Verifying factual evidence and grounding...")
                status_box.update(label="✅ Response ready", state="complete", expanded=False)
        else:
            with st.spinner("💬 Consulting reference corpus and formulating response..."):
                assistant_msg = agent_service.query(user_query, history=prior_history)

        render_assistant_card(assistant_msg)

    # Save assistant message to persistent state and rerun cleanly
    st.session_state.messages.append(assistant_msg)
    st.rerun()

# Extra bottom space so the last message is never covered by the fixed bottom input bar
st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

