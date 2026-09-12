import os
from typing import List, Dict, Any, Generator, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from src.config import SYSTEM_PROMPT_TEMPLATE, DEFAULT_TEMPERATURE
from src.vector_store import VectorStoreManager

class RAGEngine:
    """RAG Query Engine combining similarity retrieval, context formatting, and LLM answer generation."""

    def __init__(
        self, 
        vector_store_manager: VectorStoreManager, 
        provider: str = "Google Gemini", 
        api_key: str = "",
        model_name: str = "gemini-2.5-flash",
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.vector_store = vector_store_manager
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name or "gemini-2.5-flash"
        self.temperature = temperature

    def format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format document chunks into rich citation metadata cards."""
        sources = []
        for doc in docs:
            metadata = doc.metadata
            sources.append({
                "source": metadata.get("source", "Unknown Document"),
                "page": metadata.get("page", 1),
                "chunk_id": metadata.get("chunk_id", 0),
                "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                "full_content": doc.page_content
            })
        return sources

    def _fallback_extractive_answer(self, query: str, docs: List[Document]) -> str:
        """Local fallback synthesis when no LLM API key is provided or API is unreachable."""
        if not docs:
            return "No relevant documents found. Please upload document files first."

        relevant_snippets = []
        for idx, doc in enumerate(docs, start=1):
            source_info = f"[Source {idx}: {doc.metadata.get('source', 'Doc')} - Page {doc.metadata.get('page', 1)}]"
            snippet = doc.page_content.strip()
            relevant_snippets.append(f"{source_info}\n{snippet}")

        header = "💡 *(Running in Local Fallback Mode. Add a valid Gemini or OpenAI API Key in the sidebar for conversational AI generation.)*\n\n"
        context_summary = "\n\n---\n\n".join(relevant_snippets)
        return f"{header}**Relevant Document Extracts matching your query:**\n\n{context_summary}"

    def _generate_gemini(self, formatted_prompt: str) -> str:
        """Generate response using official google.genai SDK with model fallback."""
        # Candidate model queue starting with user requested model_name
        candidate_models = [
            self.model_name,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b"
        ]
        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

        # 1. Try official google.genai SDK
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            last_exception = None

            for model_id in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=formatted_prompt
                    )
                    if hasattr(response, 'text') and response.text:
                        return response.text
                except Exception as e:
                    last_exception = e
                    err_msg = str(e).lower()
                    # If model not found (404), try next model candidate
                    if "404" in err_msg or "not_found" in err_msg or "not found" in err_msg:
                        continue
                    else:
                        raise e # Other error like invalid API key

            if last_exception:
                raise last_exception

        except ImportError:
            pass

        # 2. Try LangChain Google GenAI fallback
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            last_exception = None

            for model_id in models_to_try:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_id,
                        google_api_key=self.api_key,
                        temperature=self.temperature
                    )
                    res = llm.invoke(formatted_prompt)
                    return res.content if hasattr(res, 'content') else str(res)
                except Exception as e:
                    last_exception = e
                    err_msg = str(e).lower()
                    if "404" in err_msg or "not_found" in err_msg or "not found" in err_msg:
                        continue
                    else:
                        raise e

            if last_exception:
                raise last_exception
        except Exception as e:
            raise e

        raise RuntimeError("Failed to generate response with Gemini API.")

    def _generate_openai(self, formatted_prompt: str) -> str:
        """Generate response using OpenAI API."""
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=self.model_name or "gpt-3.5-turbo",
            api_key=self.api_key,
            temperature=self.temperature
        )
        res = llm.invoke(formatted_prompt)
        return res.content if hasattr(res, 'content') else str(res)

    def query(self, user_query: str, top_k: int = 4) -> Dict[str, Any]:
        """Execute complete RAG pipeline: retrieve context, invoke LLM, and return answer with sources."""
        # 1. Retrieve relevant document chunks
        retrieved_docs = self.vector_store.search_similarity(user_query, top_k=top_k)
        sources = self.format_sources(retrieved_docs)

        if not retrieved_docs:
            return {
                "answer": "No documents are currently indexed. Please upload files in the sidebar.",
                "sources": []
            }

        # 2. Format Context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source_tag = f"[Doc: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}]"
            context_parts.append(f"--- Document Snippet {i} {source_tag} ---\n{doc.page_content}")
        
        formatted_context = "\n\n".join(context_parts)

        # 3. Format Prompt
        prompt_template = PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        formatted_prompt = prompt_template.format(context=formatted_context, question=user_query)

        # 4. Generate LLM Response or Fallback
        answer = ""
        if self.provider == "Google Gemini" and self.api_key:
            try:
                answer = self._generate_gemini(formatted_prompt)
            except Exception as e:
                answer = f"⚠️ Gemini API Error: {str(e)}\n\n" + self._fallback_extractive_answer(user_query, retrieved_docs)
        elif self.provider == "OpenAI" and self.api_key:
            try:
                answer = self._generate_openai(formatted_prompt)
            except Exception as e:
                answer = f"⚠️ OpenAI API Error: {str(e)}\n\n" + self._fallback_extractive_answer(user_query, retrieved_docs)
        else:
            answer = self._fallback_extractive_answer(user_query, retrieved_docs)

        return {
            "answer": answer,
            "sources": sources
        }
