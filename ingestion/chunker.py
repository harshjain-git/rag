from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.TARGET_CHUNK_SIZE,
    chunk_overlap=config.TARGET_CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)


def chunk_document_pages(pages):
    chunks = []
    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue
        for idx, chunk_text in enumerate(splitter.split_text(text)):
            meta = page["metadata"].copy()
            meta["chunk_id"] = f"{meta['source']}_p{meta['page']}_c{idx}"
            chunks.append({"text": chunk_text, "metadata": meta})
    return chunks
