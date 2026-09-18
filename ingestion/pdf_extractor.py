"""
PDF Extraction Module using PyMuPDF (fitz).
Extracts text page-by-page from raw PDF files while preserving metadata.
"""

from pathlib import Path
from typing import List, Dict, Any
import pymupdf


def extract_pages_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts text from a single PDF document page by page.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of dictionaries containing extracted text and page metadata.
    """
    pdf_path = Path(pdf_path)
    extracted_pages = []
    
    # Open document using PyMuPDF
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    filename = pdf_path.name

    for page_idx, page in enumerate(doc):

        # 1-indexed page number for clear citation referencing
        page_num = page_idx + 1
        raw_text = page.get_text("text") or ""

        
        
        # Check if page is low-text / potentially scanned image
        is_low_text = len(raw_text.strip()) < 50
        
        extracted_pages.append({
            "text": raw_text,
            "metadata": {
                "source": filename,
                "page": page_num,
                "total_pages": total_pages,
                "is_low_text": is_low_text
            }
        })
        

    doc.close()
    return extracted_pages



def extract_all_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Iterates through all PDF files in data_dir and extracts all pages.
    
    Args:
        data_dir: Directory containing raw PDF files.
        
    Returns:
        Flat list of all page objects across all PDFs.
    """
    data_dir = Path(data_dir)
    pdf_files = sorted(list(data_dir.glob("*.pdf")))
    
    all_extracted_pages = []
    for pdf_file in pdf_files:
        pages = extract_pages_from_pdf(pdf_file)
        all_extracted_pages.extend(pages)
        
    return all_extracted_pages
