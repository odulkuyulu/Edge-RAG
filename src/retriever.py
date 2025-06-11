"""
Retriever Module

This module implements a sophisticated document retrieval and response generation system that:
1. Detects the language of incoming queries using Azure Language Service
2. Generates embeddings for semantic search
3. Retrieves relevant documents from a vector database
4. Generates contextual responses using language-specific LLM models

The system supports multiple languages, with special handling for Arabic queries.
"""

import os
import ollama
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
import logging
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from src.embeddings import TextEmbeddingModel
from src.vector_db import VectorDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Retriever:
    """
    A class that orchestrates the entire retrieval and response generation process.
    
    This class combines multiple services:
    - Azure Language Service for language detection
    - Text embedding generation for semantic search
    - Vector database for document retrieval
    - LLM for response generation
    """
    
    def __init__(self):
        """
        Initialize the Retriever with all necessary services and clients.
        Sets up language detection, embedding generation, and vector database connections.
        """
        # Initialize Azure Language Service for language detection
        self.azure_language_endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.azure_language_key = os.getenv("AZURE_LANGUAGE_KEY")
        self.text_analytics_client = None

        if self.azure_language_endpoint and self.azure_language_key:
            self.text_analytics_client = TextAnalyticsClient(
                endpoint=self.azure_language_endpoint,
                credential=AzureKeyCredential(self.azure_language_key)
            )
            logger.info("Azure Language client initialized for Retriever.")
        else:
            logger.warning("Azure Language credentials not found. Language detection will be skipped in Retriever.")

        # Initialize embedding generator for semantic search
        self.embedding_generator = TextEmbeddingModel()
        logger.info("Text embedding model initialized.")

        # Initialize vector database for document storage and retrieval
        self.vector_store = VectorDBClient()
        logger.info("Vector database client initialized.")

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text using Azure Language Service.
        
        Args:
            text (str): The input text to detect language for
            
        Returns:
            str: ISO 639-1 language code (e.g., 'en' for English, 'ar' for Arabic)
                 Defaults to 'en' if detection fails
        """
        if not self.text_analytics_client:
            logger.warning("Language detection service not available, defaulting to English.")
            return "en"

        try:
            documents = [text]
            response = self.text_analytics_client.detect_language(documents, country_hint="us")
            
            if response and response[0].primary_language:
                detected_lang = response[0].primary_language.iso6391_name
                logger.info(f"Detected language: {detected_lang}")
                return detected_lang
            else:
                logger.warning("Language detection failed, defaulting to English.")
                return "en"
        except Exception as e:
            logger.error(f"Error detecting language with Azure Language Service: {e}")
            return "en"

    def _generate_llm_response(self, query: str, context: str, model_name: str, detected_language: str) -> str:
        """
        Generate a response using the specified LLM model.
        
        Args:
            query (str): The user's question
            context (str): Retrieved context from vector database
            model_name (str): Name of the LLM model to use
            detected_language (str): Detected language code
            
        Returns:
            str: Generated response from the LLM
            
        Raises:
            Exception: If communication with LLM service fails
        """
        base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        try:
            # Construct language-specific prompt
            if detected_language == "ar":
                prompt = "Your response MUST be in Arabic only.\n\n" \
                         "You are a helpful AI assistant. Use the following context to answer the question.\n" \
                         "If you cannot find the answer in the context, say so.\n\n" \
                         f"Context: {context}\n\n" \
                         f"Question: {query}\n\n" \
                         "Answer:"
            else:
                prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say so.

Context: {context}

Question: {query}

Answer:"""
            
            logger.info(f"Sending request to Ollama API for model: {model_name}")
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with Ollama API: {e}")
            raise Exception(f"Failed to communicate with LLM service: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error generating LLM response: {e}")
            raise Exception(f"Failed to generate LLM response: {str(e)}")

    def retrieve_and_generate_response(self, query: str) -> Dict[str, Any]:
        """
        Main method that orchestrates the entire retrieval and response generation process.
        
        Args:
            query (str): The user's question
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - response: The generated response
                - sources: List of source documents with metadata
                - detected_language: The detected language code
                - llm_model_used: The LLM model used for generation
                
        Raises:
            Exception: If any step in the process fails
        """
        try:
            # 1. Detect query language
            detected_language = self._detect_language(query)
            logger.info(f"Processing query in language: {detected_language}")

            # 2. Select appropriate LLM model based on language
            if detected_language == "ar":
                llm_model_name = "phi4-mini:latest"  # Model optimized for Arabic
            else:
                llm_model_name = "gemma3:1b"  # Default model for other languages
            logger.info(f"Selected LLM model: {llm_model_name}")
            
            # 3. Generate query embedding for semantic search
            logger.info("Generating query embedding...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # 4. Search for relevant documents in vector database
            logger.info("Searching for relevant documents...")
            results = self.vector_store.search(query_embedding, limit=5)
            if not results:
                logger.warning("No relevant documents found for query.")
                return {
                    "response": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "detected_language": detected_language,
                    "llm_model_used": llm_model_name
                }
            
            # 5. Generate response using LLM with retrieved context
            logger.info("Generating response using LLM...")
            context = "\n".join([doc.text for doc in results])
            response_text = self._generate_llm_response(query, context, llm_model_name, detected_language)
            
            # 6. Return comprehensive response with metadata
            return {
                "response": response_text,
                "sources": [
                    {
                        "text": doc.text,
                        "source": doc.metadata.get("source", "Unknown"),
                        "score": doc.score
                    }
                    for doc in results
                ],
                "detected_language": detected_language,
                "llm_model_used": llm_model_name
            }
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate_response: {e}")
            raise Exception(f"Failed to process query: {str(e)}") 