"""
Inspection Script for Phase 3 (Text Cleaning & Metadata Preparation).
Applies clean_text() across all extracted document pages and presents a Before vs After comparison.
"""

import sys
from pathlib import Path

# Add project root to sys.path so config can be imported directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ingestion.pdf_extractor import extract_all_documents
from ingestion.text_cleaner import clean_text


def run_cleaning_inspection():
    print("=" * 60)
    print("PHASE 3 INSPECTION: TEXT CLEANING & METADATA PREPARATION")
    print("=" * 60)

    data_dir = config.DATA_RAW_DIR
    raw_pages = extract_all_documents(data_dir)

    print(f"Total Extracted Pages to Clean: {len(raw_pages)}\n")

    cleaned_pages = []
    total_raw_chars = 0
    total_cleaned_chars = 0

    for page in raw_pages:
        raw_txt = page["text"]
        clean_txt = clean_text(raw_txt)

        raw_len = len(raw_txt)
        clean_len = len(clean_txt)

        total_raw_chars += raw_len
        total_cleaned_chars += clean_len

        # Attach updated metadata
        updated_metadata = page["metadata"].copy()
        updated_metadata["raw_char_count"] = raw_len
        updated_metadata["cleaned_char_count"] = clean_len

        cleaned_pages.append({
            "text": clean_txt,
            "raw_text": raw_txt,
            "metadata": updated_metadata
        })

    print("Cleaning Summary Statistics:")
    print(f"  • Total Raw Characters: {total_raw_chars:,}")
    print(f"  • Total Cleaned Characters: {total_cleaned_chars:,}")
    reduction = total_raw_chars - total_cleaned_chars
    pct = (reduction / total_raw_chars * 100) if total_raw_chars > 0 else 0
    print(f"  • Noise Reduction: {reduction:,} chars ({pct:.2f}%)")

    # Display sample Before vs After for 1 page
    if cleaned_pages:
        sample = cleaned_pages[0]
        print("\n" + "-" * 50)
        print(f"SAMPLE BEFORE VS AFTER CLEANING ({sample['metadata']['source']} - Page {sample['metadata']['page']})")
        print("-" * 50)
        print("--- BEFORE (RAW TEXT) ---")
        print(repr(sample["raw_text"][:200]))
        print("\n--- AFTER (CLEANED TEXT) ---")
        print(repr(sample["text"][:200]))
        print("\n--- UPDATED METADATA CHAIN ---")
        print(sample["metadata"])

    print("=" * 60)

# from ingestion.pdf_extractor import extract_pages_from_pdf

# target_pdf = Path("data/raw/apa2.pdf")
# pages = extract_pages_from_pdf(target_pdf)
# page_1 = pages[3]  # Index 0 is Page 1

# raw_text = page_1["text"]
# cleaned_text = clean_text(raw_text)
# # 3. Print side-by-side comparison & character metrics
# print("=" * 60)
# print(f"INSPECTING PAGE 1 OF: {target_pdf.name}")
# print("=" * 60)
# print("\n--- 🔴 RAW TEXT (Page 1) ---")
# print(raw_text)
# print("\n--- 🟢 CLEANED TEXT (Page 1) ---")
# print(cleaned_text)
# print("\n" + "=" * 60)
# print("DIFFERENCE METRICS:")
# print(f"  • Raw Character Count: {len(raw_text)}")
# print(f"  • Cleaned Character Count: {len(cleaned_text)}")
# print(f"  • Characters Removed: {len(raw_text) - len(cleaned_text)}")
# print("=" * 60)
# for p in pages:
#     print(f"=== {p['metadata']['source']} - PAGE {p['metadata']['page']} ===")
#     print("--- RAW TEXT ---")
#     print(p["text"])
#     print("--- CLEANED TEXT ---")
#     clean = clean_text(p["text"])
#     print(clean)




if __name__ == "__main__":
    run_cleaning_inspection()
