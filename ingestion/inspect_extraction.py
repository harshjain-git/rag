"""
Inspection Script for Phase 2 (PDF Extraction).
Runs extraction across all 7 raw PDF files and displays statistics and metadata verification.
"""

import sys
from pathlib import Path

# Add project root to sys.path so config can be imported directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ingestion.pdf_extractor import extract_all_documents, extract_pages_from_pdf


def run_inspection():
    print("=" * 60)
    print("PHASE 2 INSPECTION: PDF EXTRACTION WITH PyMuPDF")
    print("=" * 60)
    
    data_dir = config.DATA_RAW_DIR
    pdf_files = sorted(list(data_dir.glob("*.pdf")))
    
    print(f"Target Directory: {data_dir}")
    print(f"Found {len(pdf_files)} PDF files:\n")
    
    total_pages_all = 0
    low_text_pages_all = 0
    
    for pdf_file in pdf_files:
        pages = extract_pages_from_pdf(pdf_file)
        total_pages = len(pages)
        low_text_count = sum(1 for p in pages if p["metadata"]["is_low_text"])
        
        total_pages_all += total_pages
        low_text_pages_all += low_text_count
        
        print(f"  📄 Document: {pdf_file.name}")
        print(f"     • Total Pages: {total_pages}")
        print(f"     • Low-text / Image-heavy Pages: {low_text_count}")
        print(f"     • First Page Sample (length {len(pages[0]['text'])} chars): {pages[0]['text'][:100].strip()!r}...")
        print("-" * 50)
        
    print("\nExtraction Summary:")
    print(f"  • Total PDF Documents Parsed: {len(pdf_files)}")
    print(f"  • Total Pages Extracted: {total_pages_all}")
    print(f"  • Total Low-Text Pages Flagged: {low_text_pages_all}")
    
    # Inspect 1 full sample object
    all_pages = extract_all_documents(data_dir)
    if all_pages:
        sample = all_pages[0]
        print("\nMetadata Chain Sample Structure:")
        print("  text:", sample["text"][:120].strip() + "...")
        print("  metadata:", sample["metadata"])
        
    print("=" * 60)

# # Specify the PDF file you want to inspect
# target_pdf = Path("data/raw/apa1.pdf")
# pages = extract_pages_from_pdf(target_pdf)
# for p in pages:
#     print(f"=== {p['metadata']['source']} - PAGE {p['metadata']['page']} ===")
#     print(p["text"])


if __name__ == "__main__":
    run_inspection()
