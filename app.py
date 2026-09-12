import os
import tempfile
import streamlit as st

from src.config import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_TOP_K, 
    DEFAULT_TEMPERATURE,
    SUPPORTED_PROVIDERS
)
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="NAVVA RAG - AI Document Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern Custom Styling (Glassmorphism, Vibrant Dark Mode, Glowing Accents)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #111827 50%, #0F172A 100%);
        color: #F3F4F6;
    }

    /* Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Main Header Title */
    .main-header {
        background: linear-gradient(90deg, #60A5FA 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Cards & Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Citation Source Badge */
    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: #FFFFFF;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .page-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        color: #C084FC;
        border: 1px solid rgba(139, 92, 246, 0.4);
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
    }

    /* Buttons Styling */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }

    /* Streamlit Chat Inputs & Messages */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin-bottom: 0.8rem !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State Variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ⚡ NAVVA RAG AI")
    st.caption("Custom Document Assistant Engine")
    st.markdown("---")

    # LLM Settings
    st.subheader("🤖 Model Configuration")
    provider = st.selectbox("LLM Provider", SUPPORTED_PROVIDERS, index=0)
    
    api_key = ""
    if provider != "Local Fallback":
        api_key = st.text_input(
            f"{provider} API Key", 
            type="password", 
            help="Enter your API Key to enable conversational RAG synthesis."
        )
        if not api_key:
            st.warning("⚠️ No API key provided. Running in Local Fallback mode.")
    
    model_name = ""
    if provider == "Google Gemini":
        selected_model = st.selectbox("Gemini Model", [
            "gemini-2.5-flash", 
            "gemini-2.0-flash", 
            "gemini-1.5-flash", 
            "gemini-1.5-pro", 
            "Custom Model Name..."
        ])
        if selected_model == "Custom Model Name...":
            model_name = st.text_input("Enter Custom Model Identifier", value="gemini-3.6-flash")
        else:
            model_name = selected_model
    elif provider == "OpenAI":
        selected_model = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "Custom Model Name..."])
        if selected_model == "Custom Model Name...":
            model_name = st.text_input("Enter Custom Model Identifier", value="gpt-4o")
        else:
            model_name = selected_model

    st.markdown("---")

    # RAG Hyperparameters
    st.subheader("⚙️ Retrieval Settings")
    chunk_size = st.slider("Chunk Size (chars)", 200, 2000, DEFAULT_CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 500, DEFAULT_CHUNK_OVERLAP, 50)
    top_k = st.slider("Top-K Retrieved Chunks", 1, 10, DEFAULT_TOP_K, 1)
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)

    st.markdown("---")

    # File Upload Section
    st.subheader("📁 Document Indexer")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", use_container_width=True):
            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            total_chunks_added = 0
            processed_file_names = []

            with st.spinner("Processing & embedding documents..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        chunks, _ = processor.process_and_chunk(tmp_path)
                        for chunk in chunks:
                            chunk.metadata["source"] = uploaded_file.name
                        
                        added = st.session_state.vector_store.add_documents(chunks)
                        total_chunks_added += added
                        processed_file_names.append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            if total_chunks_added > 0:
                st.session_state.indexed_files.extend(processed_file_names)
                st.success(f"Successfully indexed {total_chunks_added} chunks from {len(processed_file_names)} files!")

    # Vector Store Metrics & Clear Button
    stats = st.session_state.vector_store.get_stats()
    st.markdown("---")
    st.markdown("### 📊 Index Statistics")
    col1, col2 = st.columns(2)
    col1.metric("Indexed Chunks", stats["total_chunks"])
    col2.metric("Status", stats["status"])

    if stats["indexed_sources"]:
        st.markdown("**Indexed Files:**")
        for src in stats["indexed_sources"]:
            st.caption(f"• {src}")

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.chat_history = []
        st.session_state.indexed_files = []
        st.rerun()

# --- MAIN CHAT APPLICATION ---
st.markdown('<div class="main-header">⚡ NAVVA RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions across your PDFs, Word documents, text files, and notes with instant vector retrieval & source verification.</div>', unsafe_allow_html=True)

# Preset Prompt Suggestions
st.markdown("**Quick Prompts:**")
p_cols = st.columns(3)
quick_prompt = ""
if p_cols[0].button("📄 Summarize uploaded documents"):
    quick_prompt = "Provide a high-level summary of the main points covered in the uploaded documents."
if p_cols[1].button("🔑 Key concepts & findings"):
    quick_prompt = "What are the most important key concepts, figures, or takeaways mentioned?"
if p_cols[2].button("❓ What questions can this answer?"):
    quick_prompt = "List 5 key questions that can be answered based on the document content."

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Document Sources & Citations", expanded=False):
                for idx, src in enumerate(message["sources"], start=1):
                    st.markdown(
                        f'<span class="source-badge">📄 {src["source"]}</span> '
                        f'<span class="page-badge">Page {src["page"]}</span> '
                        f'<span class="page-badge">Chunk #{src["chunk_id"]}</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(src["snippet"])
                    st.divider()

# Handle Chat Input
user_input = st.chat_input("Ask anything about your documents...") or quick_prompt

if user_input:
    # 1. Add User Message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. RAG Engine Query Execution
    rag_engine = RAGEngine(
        vector_store_manager=st.session_state.vector_store,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching document index & synthesizing answer..."):
            result = rag_engine.query(user_input, top_k=top_k)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("📚 View Document Sources & Citations", expanded=True):
                    for idx, src in enumerate(sources, start=1):
                        st.markdown(
                            f'<span class="source-badge">📄 {src["source"]}</span> '
                            f'<span class="page-badge">Page {src["page"]}</span> '
                            f'<span class="page-badge">Chunk #{src["chunk_id"]}</span>',
                            unsafe_allow_html=True
                        )
                        st.caption(src["snippet"])
                        if idx < len(sources):
                            st.divider()

    # 3. Store Assistant Response in History
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
