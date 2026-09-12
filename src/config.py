import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(BASE_DIR / "vectorstore"))
TEMP_UPLOAD_DIR = str(BASE_DIR / "temp_uploads")

# Default Parameters
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_TEMPERATURE = 0.2

# Supported LLM Providers
SUPPORTED_PROVIDERS = ["Google Gemini", "OpenAI", "Local Fallback"]

# Supported File Extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]

SYSTEM_PROMPT_TEMPLATE = """You are an intelligent, precise AI Document Assistant. 
Answer the user's question based strictly on the provided document contexts.

Instructions:
1. Provide clear, direct, readable, and structured answers using markdown formatting and bullet points where appropriate.
2. Rely ONLY on the information provided in the context. Ignore any raw digital signatures, font encoding codes, certificate blobs, or unreadable symbols present in the context.
3. NEVER output raw binary signatures, hex dumps, CID codes, or garbage character strings in your answer.
4. If the context does not contain enough legible information to answer the question, state clearly: "I could not find sufficient information in the uploaded documents to answer this question."

Context:
{context}

Question: {question}

Answer:"""

def get_api_key(provider: str) -> str:
    """Retrieve API key from environment or return empty string."""
    if provider == "Google Gemini":
        return os.getenv("GEMINI_API_KEY", "")
    elif provider == "OpenAI":
        return os.getenv("OPENAI_API_KEY", "")
    return ""
