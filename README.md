# ⚡ NAVVA RAG - Enterprise AI Document Assistant & Chatbot

An enterprise-grade, highly modular **Retrieval-Augmented Generation (RAG)** system built from scratch in Python. The application enables users to upload multi-format documents (PDF, DOCX, TXT, MD), sanitize raw text streams, index embeddings into a local vector database, and perform context-aware Q&A with strict source citations using Google Gemini or OpenAI LLMs.

---

## 🌟 Key Features

- **📄 Multi-Format Document Ingestion**:
  - Full support for **PDF**, **Microsoft Word (DOCX)**, **Text (TXT)**, and **Markdown (MD)** documents.
  - Custom `PureTextSplitter` for intelligent recursive text chunking with metadata tracking (source file, page numbers, chunk IDs).

- **🧹 Automated Document Sanitizer**:
  - Built-in text cleaning engine (`clean_extracted_text()`) that automatically strips out raw digital signatures, PKCS#7 certificate streams, CID font artifacts (`(cid:...)`), and non-printable control characters.

- **⚡ Lightweight & Fast Vector Search Engine**:
  - `NumpyVectorStore` utilizing **TF-IDF & Cosine Similarity** written in pure Python and NumPy.
  - Ensures 100% CPU compatibility, fast execution, local disk persistence (`vector_index.pkl`), and zero OS binary DLL policy issues.

- **🤖 Multi-LLM Provider & Automatic Model Fallback Chain**:
  - Supports **Google Gemini API** (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-3.6-flash`, `gemini-1.5-pro`) and **OpenAI API** (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`).
  - Features an **Automated Model Fallback Chain** that dynamically recovers from API endpoint model deprecations.
  - Includes a **Local Extractive Fallback Mode** so you can query documents out-of-the-box even without an API key!

- **📚 Interactive Source Citations & Anti-Hallucination**:
  - Every response includes expandable citation cards displaying exact source document names, page numbers, chunk IDs, and text snippets.

- **🎨 Modern Glassmorphic Dashboard**:
  - Dark-themed interactive UI built with Streamlit and custom CSS styling.
  - Includes sidebar controls for API keys, chunk size, overlap, Top-K retrieval, temperature, and instant index statistics.

---

## 🏗️ Architecture Overview

```
 ┌───────────────────────────┐
 │   Uploaded Documents      │  (PDF, DOCX, TXT, MD)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │ Document Text Sanitizer   │  (Strips Signatures, CID codes, Control chars)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │    PureTextSplitter       │  (Recursive Text Chunking & Metadata Enrichment)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │     NumpyVectorStore      │  (TF-IDF Vector Indexing & Cosine Similarity Search)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │    RAG Query Engine       │  (Gemini / OpenAI Model Fallback Chain)
 └─────────────┬─────────────┘
               │
               ▼
 ┌───────────────────────────┐
 │  Interactive Web Dashboard│  (Streamlit UI with Citation Cards)
 └───────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── app.py                   # Streamlit Glassmorphic Web Dashboard
├── requirements.txt         # Project dependencies
├── .env.example             # Environment configuration template
├── README.md                # Project documentation
├── src/
│   ├── __init__.py
│   ├── config.py            # Hyperparameters, paths, and system prompt templates
│   ├── document_processor.py# Multi-format document parser, text sanitizer & splitter
│   ├── vector_store.py      # Numpy vector store & cosine similarity search engine
│   └── rag_engine.py        # RAG query engine & LLM generation pipeline
└── tests/
    ├── __init__.py
    └── test_rag.py          # Complete unit test suite
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure **Python 3.10+** is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables (Optional)
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

### 4. Launch the Web Application
Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

To run the automated test suite validating document parsing, vector indexing, similarity search, and RAG query execution:

```bash
python tests/test_rag.py
```

---

## 🎓 Technical Highlights for Interviews

- **Robust Preprocessing**: Solved raw PDF stream garbage by implementing regex sanitization for digital signature blocks, PKCS#7 certificate streams, and CID font mappings.
- **Enterprise OS Safety**: Engineered a pure-Python/NumPy vector search engine (`NumpyVectorStore`) ensuring zero binary DLL policy blocks on restricted enterprise Windows environments.
- **Fault-Tolerant API Integration**: Built a resilient fallback chain across Gemini models (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`) to prevent endpoint 404 errors.
- **Verifiable RAG Outputs**: Guaranteed anti-hallucination tracking by mapping every query answer back to source document pages and chunk snippets.
