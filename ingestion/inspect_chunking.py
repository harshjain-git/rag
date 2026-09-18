"""
Inspection Script for Phase 4 (Chunking Implementation & Verification).
Extracts, cleans, and chunks all 7 PDF documents, displaying full metrics and metadata structure.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ingestion.pdf_extractor import extract_all_documents, extract_pages_from_pdf
from ingestion.text_cleaner import clean_text
from ingestion.chunker import chunk_document_pages


def run_chunking_inspection():
    print("=" * 60)
    print("PHASE 4 INSPECTION: LANGCHAIN RECURSIVE CHUNKING")
    print("=" * 60)

    # 1. Extract raw pages
    raw_pages = extract_all_documents(config.DATA_RAW_DIR)

    # 2. Clean each page
    cleaned_pages = []
    for page in raw_pages:
        page_copy = page.copy()
        page_copy["text"] = clean_text(page["text"])
        cleaned_pages.append(page_copy)

    # 3. Generate chunks
    chunks = chunk_document_pages(cleaned_pages)

    total_chunks = len(chunks)
    avg_chunk_len = (
        sum(len(c["text"]) for c in chunks) // total_chunks if total_chunks > 0 else 0
    )

    print(f"Extraction & Cleaning Overview:")
    print(f"  • Source PDFs Processed: {len(list(config.DATA_RAW_DIR.glob('*.pdf')))}")
    print(f"  • Total Pages Extracted & Cleaned: {len(cleaned_pages)}")

    print(f"\nChunking Summary Statistics:")
    print(f"  • Target Chunk Size: {config.TARGET_CHUNK_SIZE} chars/tokens")
    print(f"  • Target Overlap: {config.TARGET_CHUNK_OVERLAP} chars/tokens")
    print(f"  • Total Chunks Generated: {total_chunks}")
    print(f"  • Average Chunk Length: {avg_chunk_len} characters")

    # Display 1 full sample chunk structure
    if chunks:
        sample = chunks[0]
        print("\n" + "-" * 50)
        print("SAMPLE CHUNK STRUCTURE & METADATA CHAIN:")
        print("-" * 50)
        print("Chunk Text:", repr(sample["text"]) + "...")
        print("Metadata:", sample["metadata"])

    print("=" * 60)

# target_pdf = Path("data/raw/apa2.pdf")
# pages = extract_pages_from_pdf(target_pdf)

# # Clean each page text
# for p in pages:
#     p["text"] = clean_text(p["text"])

# # Chunk all cleaned pages
# chunks = chunk_document_pages(pages)

# print("\n" + "-" * 50)
# print(f"SAMPLE CHUNK STRUCTURE & METADATA CHAIN ({target_pdf.name}):")
# print("-" * 50)
# print("Total Chunks Generated:", len(chunks))
# print("\nFirst Chunk Text:", repr(chunks[0]["text"][:200]) + "...")
# print("First Chunk Metadata:", chunks[0]["metadata"])
# for chunk in chunks :
#     print(chunk["text"])
#     print("\n" + "-" * 50)


# if __name__ == "__main__":
#     run_chunking_inspection()

