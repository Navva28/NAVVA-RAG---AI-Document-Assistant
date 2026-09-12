import os
import sys
import unittest
import tempfile
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary text file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, "sample_ai_report.txt")
        
        self.sample_text = """
        Artificial Intelligence (AI) and Retrieval-Augmented Generation (RAG)
        RAG is a technique for enhancing LLM responses by fetching relevant facts from external knowledge bases.
        FAISS is an open-source library developed by Meta for efficient similarity search and clustering of dense vectors.
        NAVVA RAG AI Assistant enables fast, private, multi-document question answering across PDFs and text files.
        """
        
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write(self.sample_text)

        self.vector_store_dir = os.path.join(self.temp_dir.name, "test_vectorstore")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_document_processor(self):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks, raw_count = processor.process_and_chunk(self.test_file_path)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata["source"], "sample_ai_report.txt")

    def test_vector_store_and_search(self):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks, _ = processor.process_and_chunk(self.test_file_path)

        store = VectorStoreManager(persist_dir=self.vector_store_dir)
        store.add_documents(chunks)

        stats = store.get_stats()
        self.assertEqual(stats["status"], "Active")
        self.assertGreater(stats["total_chunks"], 0)

        results = store.search_similarity("What is FAISS?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("FAISS", results[0].page_content)

    def test_rag_engine_fallback(self):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks, _ = processor.process_and_chunk(self.test_file_path)

        store = VectorStoreManager(persist_dir=self.vector_store_dir)
        store.add_documents(chunks)

        engine = RAGEngine(vector_store_manager=store, provider="Local Fallback")
        response = engine.query("What is RAG?", top_k=2)

        self.assertIn("answer", response)
        self.assertIn("sources", response)
        self.assertGreater(len(response["sources"]), 0)
        self.assertEqual(response["sources"][0]["source"], "sample_ai_report.txt")

if __name__ == "__main__":
    unittest.main()
