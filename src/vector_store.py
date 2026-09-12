import os
import shutil
import re
import pickle
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from src.config import VECTOR_STORE_DIR

class NumpyVectorStore:
    """
    Pure-Python & Numpy high-performance Vector Indexing & Cosine Similarity Search Engine.
    Bypasses third-party OS DLL policy restrictions on Windows while providing fast, accurate RAG retrieval.
    """

    def __init__(self, vocabulary_size: int = 10000):
        self.vocabulary_size = vocabulary_size
        self.documents: List[Document] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.doc_vectors: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase tokens and bigrams."""
        words = re.findall(r'\b\w+\b', text.lower())
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams

    def fit_and_index(self, documents: List[Document]):
        """Build vocabulary, calculate IDF, and encode document vectors."""
        self.documents = documents
        if not documents:
            self.vocab = {}
            self.idf = None
            self.doc_vectors = None
            return

        # 1. Build Vocabulary
        doc_tokens_list = [self._tokenize(doc.page_content) for doc in documents]
        token_counts: Dict[str, int] = {}
        for tokens in doc_tokens_list:
            for token in set(tokens):
                token_counts[token] = token_counts.get(token, 0) + 1

        # Keep top terms sorted by frequency
        sorted_terms = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocabulary_size]
        self.vocab = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        
        num_docs = len(documents)
        vocab_len = len(self.vocab)
        
        if vocab_len == 0:
            return

        # 2. Calculate IDF vector
        self.idf = np.zeros(vocab_len, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = token_counts.get(term, 1)
            self.idf[idx] = math.log((1 + num_docs) / (1 + df)) + 1.0

        # 3. Calculate Normalized Document Vectors (TF-IDF)
        matrix = np.zeros((num_docs, vocab_len), dtype=np.float32)
        for doc_idx, tokens in enumerate(doc_tokens_list):
            for token in tokens:
                if token in self.vocab:
                    term_idx = self.vocab[token]
                    matrix[doc_idx, term_idx] += 1.0

        # Sublinear TF scaling: 1 + log(tf)
        tf_matrix = np.where(matrix > 0, 1.0 + np.log(np.maximum(matrix, 1.0)), 0.0)
        tfidf_matrix = tf_matrix * self.idf

        # L2 Normalize Document Vectors
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = tfidf_matrix / norms

    def query(self, query_text: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """Query index and return top-k documents sorted by cosine similarity score."""
        if not self.documents or self.doc_vectors is None or len(self.vocab) == 0:
            return []

        tokens = self._tokenize(query_text)
        query_vec = np.zeros(len(self.vocab), dtype=np.float32)
        
        for token in tokens:
            if token in self.vocab:
                term_idx = self.vocab[token]
                query_vec[term_idx] += 1.0

        if np.max(query_vec) == 0:
            # Fallback: return initial documents if no vocabulary match
            return [(doc, 0.0) for doc in self.documents[:top_k]]

        tf_query = np.where(query_vec > 0, 1.0 + np.log(np.maximum(query_vec, 1.0)), 0.0)
        tfidf_query = tf_query * self.idf

        norm = np.linalg.norm(tfidf_query)
        if norm > 0:
            tfidf_query = tfidf_query / norm

        # Cosine Similarity Dot Product
        scores = np.dot(self.doc_vectors, tfidf_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]


class VectorStoreManager:
    """Manages creation, indexing, persistence, and similarity search of document vector embeddings."""

    def __init__(self, persist_dir: str = VECTOR_STORE_DIR, api_key: str = "", provider: str = "Local"):
        self.persist_dir = persist_dir
        self.provider = provider
        self.api_key = api_key
        self.all_documents: List[Document] = []
        self.index = NumpyVectorStore()
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing vector store from disk if available."""
        index_file = os.path.join(self.persist_dir, "vector_index.pkl")
        if os.path.exists(index_file):
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                    self.all_documents = data.get("documents", [])
                    self.index.fit_and_index(self.all_documents)
            except Exception as e:
                print(f"Warning: Failed to load vector index from {self.persist_dir}: {e}")

    def add_documents(self, documents: List[Document]) -> int:
        """Add document chunks to the vector database and save to disk."""
        if not documents:
            return 0

        self.all_documents.extend(documents)
        self.index.fit_and_index(self.all_documents)
        self.save()
        return len(documents)

    def save(self):
        """Save vector store persistence data to disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        index_file = os.path.join(self.persist_dir, "vector_index.pkl")
        try:
            with open(index_file, "wb") as f:
                pickle.dump({"documents": self.all_documents}, f)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def search_similarity(self, query: str, top_k: int = 4) -> List[Document]:
        """Perform similarity search for top-k matching document chunks."""
        results = self.index.query(query, top_k=top_k)
        return [doc for doc, _ in results]

    def search_with_score(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """Perform similarity search returning documents with relevance scores."""
        return self.index.query(query, top_k=top_k)

    def clear(self):
        """Delete vector store persistence and reset state."""
        self.all_documents = []
        self.index = NumpyVectorStore()
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
            except Exception as e:
                print(f"Error clearing vector store directory: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return vector database metrics."""
        total_chunks = len(self.all_documents)
        sources = set()
        for doc in self.all_documents:
            if isinstance(doc, Document) and "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        return {
            "total_chunks": total_chunks,
            "status": "Active" if total_chunks > 0 else "Empty",
            "persist_dir": self.persist_dir,
            "indexed_sources": list(sources)
        }
