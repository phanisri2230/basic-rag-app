import os
import uuid
import re
import io
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Document QA System", layout="wide", page_icon="📄")
st.title("📄 Document RAG Pipeline (Groq & ChromaDB)")
st.caption("Upload a PDF document, ask questions, and get precise, grounded answers.")

# Verify API key
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ `GROQ_API_KEY` is missing! Please add it to your `.env` file and restart.")
    st.stop()

# Cache embedding model to load once across reruns
@st.cache_resource(show_spinner="Loading Embedding Model (all-MiniLM-L6-v2)...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cache ONNX OCR engine
@st.cache_resource(show_spinner="Initializing OCR engine (RapidOCR)...")
def get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception:
        return None

# Initialize session state tracking
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = None
if "latest_retrieved_docs" not in st.session_state:
    st.session_state.latest_retrieved_docs = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "engine_used" not in st.session_state:
    st.session_state.engine_used = None
if "is_corrupt" not in st.session_state:
    st.session_state.is_corrupt = False

def reset_system():
    """Wipes vector store collection and clears session state."""
    if st.session_state.vector_db is not None:
        try:
            st.session_state.vector_db.delete_collection()
        except Exception:
            pass
    st.session_state.vector_db = None
    st.session_state.processed_file = None
    st.session_state.latest_answer = None
    st.session_state.latest_retrieved_docs = []
    st.session_state.all_chunks = []
    st.session_state.collection_name = None
    st.session_state.engine_used = None
    st.session_state.is_corrupt = False

def clean_text(text: str) -> str:
    """Cleans extracted PDF text while preserving sentence boundaries."""
    if not text:
        return ""
    # Replace control chars / null bytes / form feeds
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_garbage_text(text: str) -> bool:
    """Checks if extracted text contains (cid:) tags or corrupt font symbols."""
    if not text or len(text.strip()) < 10:
        return True
    if "(cid:" in text:
        return True
    alphanumeric = sum(1 for c in text if c.isalnum())
    ratio = alphanumeric / float(len(text))
    return ratio < 0.35

def extract_pdf_pages(file_bytes: bytes, filename: str):
    """
    Multi-engine PDF text extractor with OCR:
    Uses PyMuPDF first. If (cid:) tags or garbled symbols are detected,
    renders page images and applies RapidOCR automatically.
    """
    page_docs = []
    ocr_pages_count = 0
    engine_used = "PyMuPDF"
    ocr_engine = get_ocr_engine()
    
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text")
            cleaned = clean_text(text)
            
            # If text contains (cid:) tags or is garbled, run OCR on rendered page
            if is_garbage_text(cleaned) and ocr_engine is not None:
                try:
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    ocr_res, _ = ocr_engine(img_bytes)
                    if ocr_res:
                        lines = [line[1] for line in ocr_res if line[1].strip()]
                        cleaned = clean_text("\n".join(lines))
                        ocr_pages_count += 1
                except Exception:
                    pass
            
            if cleaned and not is_garbage_text(cleaned):
                page_docs.append(Document(
                    page_content=cleaned,
                    metadata={"page": i + 1, "source": filename}
                ))
    except Exception:
        page_docs = []

    if ocr_pages_count > 0:
        engine_used = f"PyMuPDF + Auto OCR ({ocr_pages_count} pages OCR'd)"

    # Fallback 2: pdfplumber if PyMuPDF returned no valid docs
    if not page_docs:
        try:
            import pdfplumber
            page_docs = []
            engine_used = "pdfplumber"
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    cleaned = clean_text(text)
                    if cleaned and not is_garbage_text(cleaned):
                        page_docs.append(Document(
                            page_content=cleaned,
                            metadata={"page": i + 1, "source": filename}
                        ))
        except Exception:
            pass

    all_extracted_text = " ".join([d.page_content for d in page_docs])
    is_corrupt = is_garbage_text(all_extracted_text)

    return page_docs, engine_used, is_corrupt

# Fixed RAG Configuration (Hidden from UI)
selected_model = "qwen/qwen3.8-27b"
retrieval_mode = "Similarity Search"
retrieval_k = 6
chunk_size = 800
chunk_overlap = 150

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Actions")
    if st.button("🗑️ Reset Pipeline & Clear DB", use_container_width=True):
        reset_system()
        st.rerun()

# Step 1: File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # If a new or different file is uploaded, wipe previous data and re-index
    if st.session_state.processed_file != uploaded_file.name or st.session_state.vector_db is None:
        reset_system()
        
        with st.spinner(f"Extracting text from `{uploaded_file.name}` using PyMuPDF..."):
            file_bytes = uploaded_file.read()
            page_docs, engine_used, is_corrupt = extract_pdf_pages(file_bytes, uploaded_file.name)
            
            if not page_docs:
                st.warning("⚠️ No readable text found in this PDF (it may contain scanned images without OCR).")
            else:
                # 2. Chunk text using RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunked_docs = splitter.split_documents(page_docs)
                st.session_state.all_chunks = chunked_docs
                
                # 3. Embed and store in in-memory ChromaDB with unique collection name
                embeddings = get_embedding_model()
                col_name = f"pdf_{uuid.uuid4().hex[:10]}"
                
                vector_db = Chroma.from_documents(
                    documents=chunked_docs,
                    embedding=embeddings,
                    collection_name=col_name
                )
                
                st.session_state.vector_db = vector_db
                st.session_state.processed_file = uploaded_file.name
                st.session_state.collection_name = col_name
                st.session_state.engine_used = engine_used
                st.session_state.is_corrupt = is_corrupt
                
                st.success(f"✅ Successfully indexed {len(chunked_docs)} chunks across {len(page_docs)} pages using `{engine_used}`.")
                if is_corrupt:
                    st.error("⚠️ **Warning:** The text in this PDF appears garbled or contains unreadable font encodings. If answer quality is low, please upload a searchable PDF or run OCR.")

# Step 2: Query & Answer Generation
if st.session_state.vector_db is not None:
    st.markdown("---")
    query = st.text_input("💬 Ask a question based on your document:", placeholder="e.g. What are the key recommendations in Section 3?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_btn = st.button("🔍 Generate Answer", type="primary", use_container_width=True)
    
    if submit_btn and query.strip():
        with st.spinner("Retrieving context and generating answer..."):
            # Vector Retrieval
            if "MMR" in retrieval_mode:
                retriever = st.session_state.vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": retrieval_k, "fetch_k": retrieval_k * 3}
                )
            else:
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": retrieval_k}
                )
            
            relevant_docs = retriever.invoke(query)
            st.session_state.latest_retrieved_docs = relevant_docs
            
            # Format Context with Source Page Numbers
            context_blocks = []
            for idx, doc in enumerate(relevant_docs):
                page_num = doc.metadata.get("page", "Unknown")
                context_blocks.append(f"[Snippet {idx+1} | Page {page_num}]\n{doc.page_content}")
            
            context_str = "\n\n".join(context_blocks)
            
            # Context Cap Guard (max ~12,000 characters)
            max_chars = 12000
            if len(context_str) > max_chars:
                context_str = context_str[:max_chars] + "\n\n[Context truncated for prompt safety...]"
            
            # Strict Grounded Prompt
            prompt = f"""You are a precise AI document assistant. Answer the user's question ONLY using the factual context provided below.

RULES:
1. Base your answer STRICTLY on the facts present in the context. Do NOT extrapolate or introduce external facts.
2. If the context does NOT contain enough information to answer the question, state clearly: "I cannot find the answer to this question in the uploaded document."
3. Refer to specific sections or page numbers if mentioned in the context snippets.

CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER:"""

            # LLM Execution with fallback
            try:
                llm = ChatGroq(model_name=selected_model, temperature=0.0)
                response = llm.invoke(prompt)
                st.session_state.latest_answer = response.content
            except Exception as primary_err:
                try:
                    # Fallback model
                    fallback_model = "openai/gpt-oss-20b" if selected_model != "openai/gpt-oss-20b" else "qwen/qwen3.8-27b"
                    llm_fallback = ChatGroq(model_name=fallback_model, temperature=0.0)
                    response = llm_fallback.invoke(prompt)
                    st.session_state.latest_answer = response.content
                    st.warning(f"Primary model `{selected_model}` failed. Used fallback model `{fallback_model}`.")
                except Exception as err2:
                    st.error(f"❌ Error communicating with Groq API: {err2}")

# Step 3: Output Display
if st.session_state.latest_answer:
    st.markdown("### 💡 Answer:")
    st.markdown(st.session_state.latest_answer)
    
    # Transparency: Show Retrieved Chunks & Page Numbers
    if st.session_state.latest_retrieved_docs:
        with st.expander("🔍 View Retrieved Document Snippets (Context Used)"):
            for i, doc in enumerate(st.session_state.latest_retrieved_docs):
                page_num = doc.metadata.get("page", "N/A")
                st.markdown(f"**Snippet {i+1}** *(Page {page_num})*")
                st.info(doc.page_content)