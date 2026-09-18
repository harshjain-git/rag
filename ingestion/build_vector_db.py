"""
Vector Database Build & Persistent Indexing Script.
Extracts, cleans, chunks, embeds, and stores all 7 PDF documents in ChromaDB.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ingestion.pdf_extractor import extract_all_documents
from ingestion.text_cleaner import clean_text
from ingestion.chunker import chunk_document_pages
from ingestion.vector_store import get_vector_store, add_chunks_to_vector_store


def build_and_verify_vector_db():
    print("=" * 60)
    print("PHASE 5 & 6: EMBEDDING GENERATION & CHROMADB PERSISTENCE")
    print("=" * 60)

    # 1. Extract raw pages from all PDFs
    print("\n[1/4] Extracting pages from data/raw/...")
    raw_pages = extract_all_documents(config.DATA_RAW_DIR)
    print(f"      Extracted {len(raw_pages)} pages.")

    # 2. Clean page texts
    print("\n[2/4] Cleaning text content...")
    cleaned_pages = []
    for page in raw_pages:
        p_copy = page.copy()
        p_copy["text"] = clean_text(page["text"])
        cleaned_pages.append(p_copy)

    # 3. Generate chunks
    print("\n[3/4] Generating chunks with LangChain text splitter...")
    chunks = chunk_document_pages(cleaned_pages)
    print(f"      Generated {len(chunks)} total chunks.")

    # 4. Embed and persist in ChromaDB
    print("\n[4/4] Embedding chunks with BAAI/bge-base-en-v1.5 & saving to ChromaDB...")
    vector_store = get_vector_store()
    add_chunks_to_vector_store(chunks, vector_store)
    print(f"      SUCCESS: Persisted {len(chunks)} chunks to '{config.CHROMA_DB_DIR}'!")

    # 5. Run test query verification (Phase 7 preview)
    print("\n" + "-" * 60)
    print("TEST SEMANTIC RETRIEVAL QUERY VERIFICATION:")
    test_query = "What are the ASVAB score qualification standards?"
    print(f"Query: {test_query!r}\n")

    results = vector_store.similarity_search(test_query, k=3)
    for idx, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        print(f"  Result #{idx} [{src} — Page {page}] (ID: {chunk_id}):")
        print(f"  Snippet: {doc.page_content[:150].strip()!r}...")
        print("-" * 50)

    print("=" * 60)


if __name__ == "__main__":
    build_and_verify_vector_db()
