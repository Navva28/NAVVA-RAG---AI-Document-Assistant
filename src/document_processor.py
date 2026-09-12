import os
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Document parsing libraries
import pypdf
import docx
from langchain_core.documents import Document

def clean_extracted_text(text: str) -> str:
    """
    Sanitize raw extracted document text:
    - Removes non-printable control characters and CID font artifacts.
    - Removes digital signatures, certificate blocks, and raw hex streams.
    - Filters out lines composed mostly of non-alphanumeric noise.
    - Normalizes whitespace.
    """
    if not text:
        return ""

    # 1. Remove CID font artifacts like (cid:123)
    text = re.sub(r'\(cid:\d+\)', '', text)

    # 2. Remove digital signature streams, certificate blocks, and long hex strings
    text = re.sub(r'-----BEGIN [A-Z ]+-----[A-Za-z0-9+/=\s]+-----END [A-Z ]+-----', '', text)
    text = re.sub(r'/ByteRange\s*\[[^\]]+\]', '', text)
    text = re.sub(r'/Contents\s*<[A-Fa-f0-9\s]+>', '', text)
    text = re.sub(r'\b[0-9A-Fa-f]{64,}\b', '', text)

    # 3. Filter non-printable ASCII control characters (keep \n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 4. Filter unreadable noise lines
    cleaned_lines = []
    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        
        # Calculate ratio of readable alphanumeric/space characters
        alnum_count = sum(1 for c in line_str if c.isalnum() or c.isspace() or c in '.,!?-:;()[]"\'')
        ratio = alnum_count / len(line_str)
        
        # Omit lines dominated by random symbols/hex noise (unless very short heading/bullet)
        if ratio < 0.45 and len(line_str) > 8:
            continue
            
        cleaned_lines.append(line_str)

    text = "\n".join(cleaned_lines)

    # 5. Normalize multiple spaces & excessive blank lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


class PureTextSplitter:
    """
    Pure Python Recursive Text Splitter.
    Splits text on natural boundaries (paragraphs, newlines, spaces) within target chunk_size
    and chunk_overlap with zero binary DLL dependencies.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Look backwards for natural sentence/paragraph separator
                found_sep = False
                for sep in self.separators:
                    if not sep:
                        continue
                    sep_pos = text.rfind(sep, start + self.chunk_overlap, end)
                    if sep_pos != -1 and sep_pos > start:
                        end = sep_pos + len(sep)
                        found_sep = True
                        break

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            if end >= text_len:
                break

            # Slide start index respecting overlap
            step = end - self.chunk_overlap
            start = max(step, start + 1)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result_docs = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for idx, chunk_text in enumerate(text_chunks, start=1):
                meta = doc.metadata.copy()
                meta["chunk_index"] = idx
                result_docs.append(Document(page_content=chunk_text, metadata=meta))
        return result_docs


class DocumentProcessor:
    """Handles loading, parsing, and chunking of documents in multiple formats."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = PureTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Extract text from a PDF file page by page and sanitize output."""
        documents = []
        filename = Path(file_path).name
        try:
            reader = pypdf.PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                clean_text = clean_extracted_text(raw_text)
                if clean_text:
                    doc = Document(
                        page_content=clean_text,
                        metadata={
                            "source": filename,
                            "file_path": file_path,
                            "page": page_num,
                            "total_pages": len(reader.pages),
                            "file_type": "pdf"
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Error reading PDF {filename}: {str(e)}")
        return documents

    def load_docx(self, file_path: str) -> List[Document]:
        """Extract text from a Word DOCX document."""
        documents = []
        filename = Path(file_path).name
        try:
            doc_file = docx.Document(file_path)
            full_text = []
            for para in doc_file.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            
            raw_content = "\n\n".join(full_text)
            clean_content = clean_extracted_text(raw_content)
            if clean_content:
                doc = Document(
                    page_content=clean_content,
                    metadata={
                        "source": filename,
                        "file_path": file_path,
                        "page": 1,
                        "total_pages": 1,
                        "file_type": "docx"
                    }
                )
                documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Error reading DOCX {filename}: {str(e)}")
        return documents

    def load_txt(self, file_path: str) -> List[Document]:
        """Extract text from a plain TXT or MD document."""
        filename = Path(file_path).name
        ext = Path(file_path).suffix.lower()
        raw_content = ""
        
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    raw_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        clean_content = clean_extracted_text(raw_content)
        if not clean_content:
            return []

        doc = Document(
            page_content=clean_content,
            metadata={
                "source": filename,
                "file_path": file_path,
                "page": 1,
                "total_pages": 1,
                "file_type": "markdown" if ext == ".md" else "text"
            }
        )
        return [doc]

    def process_file(self, file_path: str) -> List[Document]:
        """Parse file based on extension and return raw documents."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext == ".docx":
            return self.load_docx(file_path)
        elif ext in [".txt", ".md"]:
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def chunk_documents(self, raw_documents: List[Document]) -> List[Document]:
        """Split raw documents into smaller chunk documents with updated chunk metadata."""
        if not raw_documents:
            return []

        chunks = self.text_splitter.split_documents(raw_documents)
        
        # Enrich metadata with chunk identifiers
        for idx, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def process_and_chunk(self, file_path: str) -> Tuple[List[Document], int]:
        """Process a single file and return its chunked documents along with raw doc count."""
        raw_docs = self.process_file(file_path)
        chunks = self.chunk_documents(raw_docs)
        return chunks, len(raw_docs)
