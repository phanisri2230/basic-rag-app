<<<<<<< HEAD
# basic-rag-app
=======
# 📄 Production Document RAG Pipeline

A high-precision, production-grade **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **ChromaDB**, **PyMuPDF**, **RapidOCR**, and **Groq Cloud LLM** (`qwen/qwen3.8-27b`).

---

## 🌟 Key Features

- **Hybrid Multi-Engine PDF Processing**: Automatically extracts text using `PyMuPDF` with fallback to `pdfplumber` and `pypdf`.
- **Automatic Visual OCR Fallback**: Detects missing `/ToUnicode` font maps (e.g. `(cid:)` tags) or scanned pages and automatically applies **RapidOCR (ONNX Runtime)** to extract clean, visual text.
- **In-Memory Vector Search**: Uses **ChromaDB** with `all-MiniLM-L6-v2` embeddings for fast semantic chunk retrieval.
- **Strict Anti-Hallucination Prompting**: System instructions force the LLM to answer *strictly* using retrieved document context, preventing false assumptions.
- **Source Snippet Transparency**: Includes an expandable UI component showing the exact document snippets and page numbers used to generate each answer.
- **Isolated Sessions**: Dynamically generates unique vector database collections per PDF upload to prevent data bleed across sessions.

---

## 🛠️ Architecture Overview

```
[ PDF Document Upload ]
           │
           ▼
[ Hybrid PDF Engine (PyMuPDF) ] ── (If corrupt / (cid:) tags) ──► [ ONNX RapidOCR Engine ]
           │
           ▼
[ Clean Text & Page Metadata ]
           │
           ▼
[ Recursive Character Text Splitter ]
           │
           ▼
[ Sentence-Transformers Embeddings (all-MiniLM-L6-v2) ]
           │
           ▼
[ Chroma Vector Store (In-Memory) ]
           │
           ▼ (User Question)
[ Semantic Vector Similarity Retrieval ]
           │
           ▼
[ Strictly Grounded Context Prompt ]
           │
           ▼
[ Groq LLM API (Qwen 2.5 / GPT-OSS) ]
           │
           ▼
[ Grounded Answer & Source Attribution ]
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- A **Groq API Key** (Get your free key at [console.groq.com](https://console.groq.com/keys))

### 2. Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/basic-rag-app.git
   cd basic-rag-app
   ```

2. **Create & Activate a Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / Mac**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Setup

Create a `.env` file in the root directory (or copy `.env.example`):
```bash
cp .env.example .env
```

Add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 4. Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your web browser.

---

## 💻 Tech Stack

- **Frontend / UI**: Streamlit
- **LLM Provider**: Groq API (`ChatGroq`)
- **Primary LLM**: `qwen/qwen3.8-27b`
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector Database**: ChromaDB
- **PDF Extraction**: PyMuPDF (`pymupdf`), `pdfplumber`, `pypdf`
- **OCR Engine**: RapidOCR (`rapidocr-onnxruntime`)
- **Orchestration**: LangChain Core / Community

---

## 📜 License

MIT License.
>>>>>>> 3762168 (Initial commit: Production-ready Document RAG Pipeline)
