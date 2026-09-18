# Cadet Readiness Advisor - RAG System

A Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **PyMuPDF**, **ChromaDB**, and **Gemini Flash Lite** (`gemini-2.5-flash-lite`) for Cadet Readiness Assessment and Guidance.

## Project Structure

```
cadet-readiness-advisor/
│
├── data/
│   └── raw/             # Raw PDF documents and data files
│
├── ingestion/           # Data loading, parsing, cleaning, and LangChain text chunking
├── retrieval/           # LangChain Vector store querying & context retrieval
├── evaluation/          # RAG assessment, ground truth benchmarking, and metrics
├── app/                 # Streamlit web application layer
│
├── config.py            # Central configuration & parameters
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation
```


## Setup Instructions

1. Place source PDF documents into `data/raw/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run ingestion and launch application modules.
