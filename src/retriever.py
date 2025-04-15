import os
import re
import requests
import ollama
import numpy as np
import nltk
import string
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging
from functools import lru_cache
from dataclasses import dataclass
from pydantic import BaseModel
import asyncio
import aiohttp
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data
nltk.download("punkt")

# ✅ Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ Define embedding sizes for models
EMBEDDING_SIZES = {
    "english": 1024,  # bge-m3 (Optimized for retrieval)
    "arabic": 1024,   # bge-m3 embeddings
}

# ✅ Azure AI Language Service Configuration
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

if not AZURE_LANGUAGE_ENDPOINT or not AZURE_LANGUAGE_KEY:
    raise ValueError("Azure Language Service configuration missing. Please set AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY environment variables.")

# Add proper logging instead of print statements
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrieverConfig:
    qdrant_url: str
    qdrant_api_key: str
    azure_language_endpoint: str
    azure_language_key: str
    embedding_size: int = 1024
    max_results: int = 10
    score_threshold: float = 0.0

class SearchResult(BaseModel):
    text: str
    score: float
    vector_score: float
    entity_score: float
    source: str
    chunk_id: int
    total_chunks: int
    language: str
    matched_entities: Dict[str, List[str]]

# -----------------------------------------------
# 🔹 Function: Detect Query Language
# -----------------------------------------------

def detect_language(text: str) -> str:
    """
    Detects the language of a given text using Azure Language Service.
    Improved to handle mixed language content.
    
    Args:
        text: The text to detect language for
        
    Returns:
        str: "arabic" or "english" based on detection
    """
    try:
        # Remove trailing slash if present and add the correct path
        base_endpoint = AZURE_LANGUAGE_ENDPOINT.rstrip('/')
        endpoint = f"{base_endpoint}/text/analytics/v3.1/languages"
        
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
            "Content-Type": "application/json"
        }
        
        # Split text into chunks to handle mixed content
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        detected_languages = set()
        
        for chunk in chunks:
            payload = {
                "documents": [{
                    "id": "1",
                    "text": chunk
                }]
            }
            
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "documents" in result and result["documents"]:
                detected_lang = result["documents"][0]["detectedLanguage"]["iso6391Name"]
                confidence = result["documents"][0]["detectedLanguage"]["confidenceScore"]
                if confidence > 0.7:  # Only consider high confidence detections
                    detected_languages.add(detected_lang)
        
        # If we detected both languages, prioritize Arabic for mixed content
        if "ar" in detected_languages:
            return "arabic"
        elif "en" in detected_languages:
            return "english"
        
        # Default to Arabic if no clear detection
        return "arabic"
            
    except Exception as e:
        print(f"Error detecting language: {e}")
        # Default to Arabic for safety
        return "arabic"

# -----------------------------------------------
# 🔹 Function: Generate Query Embeddings
# -----------------------------------------------

@lru_cache(maxsize=1000)
def generate_embedding(text, language):
    """Generates embeddings using different models for Arabic & English queries."""
    model_name = "bge-m3" if language == "arabic" else "bge-m3"
    
    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    # ✅ Fix: Ensure embedding size matches Qdrant expectations
    expected_size = EMBEDDING_SIZES[language]

    # 🔹 Ensure embedding has the correct size
    if len(embedding) < expected_size:
        embedding = np.pad(embedding, (0, expected_size - len(embedding)), 'constant')
    elif len(embedding) > expected_size:
        embedding = embedding[:expected_size]  # Truncate if larger

    return list(embedding)  # ✅ FIXED: Returning as list directly

# -----------------------------------------------
# 🔹 Function: Tokenize Text for BM25
# -----------------------------------------------

def tokenize_text(text, language):
    """Tokenizes input text for BM25 retrieval, handling Arabic separately."""
    if language == "arabic":
        return word_tokenize(text)  # Arabic tokenization (better for BM25)
    return [word.lower() for word in word_tokenize(text) if word not in string.punctuation]

# -----------------------------------------------
# 🔹 Function: Extract Entities
# -----------------------------------------------

def extract_entities(text: str, language: str = "en") -> List[Dict[str, str]]:
    """Extract named entities from text using Azure Language Service."""
    try:
        base_endpoint = AZURE_LANGUAGE_ENDPOINT.rstrip('/')
        endpoint = f"{base_endpoint}/text/analytics/v3.1/entities/recognition/general"
        
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "documents": [{
                "id": "1",
                "text": text,
                "language": "ar" if language == "arabic" else "en"
            }]
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if "documents" in result and result["documents"]:
            return [{"text": entity["text"], "category": entity["category"]} 
                   for entity in result["documents"][0]["entities"]]
            
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
    
    return []

# -----------------------------------------------
# 🔹 Function: Calculate Entity Score
# -----------------------------------------------

def calculate_entity_score(query_entities: List[Dict[str, str]], doc_entities: Dict[str, List[str]]) -> float:
    """Calculate similarity score based on matching entities."""
    if not query_entities or not doc_entities:
        return 0.0
    
    score = 0.0
    for query_entity in query_entities:
        query_text = query_entity["text"].lower()
        query_category = query_entity["category"]
        
        # Check if the entity exists in the same category
        if query_category in doc_entities:
            doc_entities_in_category = [e.lower() for e in doc_entities[query_category]]
            if query_text in doc_entities_in_category:
                score += 1.0  # Direct match in same category
            else:
                # Check for partial matches
                for doc_entity in doc_entities_in_category:
                    if query_text in doc_entity or doc_entity in query_text:
                        score += 0.5  # Partial match
    
    return score / len(query_entities)  # Normalize score

# -----------------------------------------------
# 🔹 Function: Search Documents with Hybrid Retrieval
# -----------------------------------------------

def search_documents(query: str, language: str) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using vector similarity.
    Improved to handle mixed language queries and multiple sources.
    
    Args:
        query: The search query
        language: The detected language of the query
        
    Returns:
        List[Dict[str, Any]]: List of relevant documents with scores
    """
    try:
        # Clean and normalize the query
        query = query.strip()
        
        # Extract entities from the query
        query_entities = extract_entities(query, language)
        
        # Generate embedding for the query
        embedding = generate_embedding(query, language)
        if embedding is None:
            return []
        
        # Search in both collections if mixed language is detected
        results = []
        
        # Search in Arabic collection
        arabic_results = client.search(
            collection_name="rag_docs_ar",
            query_vector=embedding,
            limit=5
        )
        for result in arabic_results:
            payload = result.payload
            metadata = payload.get("metadata", {})
            doc_entities = metadata.get("entities", {})
            
            # Calculate entity matching score
            entity_score = calculate_entity_score(query_entities, doc_entities)
            
            # Add source type score (prefer text files over PDFs for general queries)
            source_score = 1.0 if not metadata.get("source", "").lower().endswith('.pdf') else 0.5
            
            results.append({
                "text": payload["text"],
                "score": result.score * source_score,  # Adjust score based on source type
                "vector_score": result.score,
                "source": metadata.get("source", "Unknown"),
                "language": "arabic",
                "chunk_id": metadata.get("chunk_id", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "matched_entities": doc_entities,
                "entity_score": entity_score
            })
        
        # Search in English collection
        english_results = client.search(
            collection_name="rag_docs_en",
            query_vector=embedding,
            limit=5
        )
        for result in english_results:
            payload = result.payload
            metadata = payload.get("metadata", {})
            doc_entities = metadata.get("entities", {})
            
            # Calculate entity matching score
            entity_score = calculate_entity_score(query_entities, doc_entities)
            
            # Add source type score (prefer text files over PDFs for general queries)
            source_score = 1.0 if not metadata.get("source", "").lower().endswith('.pdf') else 0.5
            
            results.append({
                "text": payload["text"],
                "score": result.score * source_score,  # Adjust score based on source type
                "vector_score": result.score,
                "source": metadata.get("source", "Unknown"),
                "language": "english",
                "chunk_id": metadata.get("chunk_id", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "matched_entities": doc_entities,
                "entity_score": entity_score
            })
        
        # Sort results by combined score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:10]  # Return top 10 results
        
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return []

# -----------------------------------------------
# 🔹 Function: Clean AI Response & Apply Arabic Formatting
# -----------------------------------------------

def clean_ai_response(text, language):
    """Cleans AI-generated responses."""
    return text  # Simply return the text without any formatting

# -----------------------------------------------
# 🔹 Function: Generate AI Response
# -----------------------------------------------

def generate_response(query, results, max_length=512, temperature=0.9, top_k=40, repetition_penalty=1.0):
    """Generates a response using the appropriate LLM model based on detected language and retrieved documents."""
    language = detect_language(query)
    model_name = "gemma3:1b" if language == "arabic" else "phi4-mini:3.8b"

    # Prepare context from retrieved documents - improved formatting
    context_parts = []
    for i, doc in enumerate(results[:3]):
        # Format the source with complete text and metadata
        source_text = f"Source {i+1} (Score: {doc.get('score', 0):.2f}, Document: {doc.get('source', 'Unknown')}):\n{doc.get('text', '')}"
        
        # Handle matched entities safely
        matched_entities = doc.get('matched_entities', {})
        if matched_entities and isinstance(matched_entities, dict):
            entities_text = []
            for category, entities in matched_entities.items():
                if isinstance(entities, list):
                    entities_text.append(f"{category}: {', '.join(entities)}")
                else:
                    entities_text.append(f"{category}: {entities}")
            if entities_text:
                source_text += f"\nRelevant entities: {'; '.join(entities_text)}"
        
        context_parts.append(source_text)
    
    context = "\n\n".join(context_parts)

    prompt = f"""
    You are a precise and helpful AI assistant. Answer the question based on the provided sources.

    Question: {query}

    Available Sources:
    {context}

    Important Rules:
    1. Use ONLY information from the provided sources
    2. Give a direct, concise answer without mentioning sources
    3. Do not include any references or citations
    4. Do not mention limitations of the sources
    5. If information is insufficient, simply state what is known
    6. Keep the tone professional and informative
    7. Focus on facts and key information
    8. Use clear, simple language
    9. Avoid phrases like "based on the sources" or "according to"
    10. Do not recommend checking other sources

    Response Guidelines:
    - Start with the main facts
    - Be direct and clear
    - Keep it concise
    - No source references
    - No disclaimers
    """

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "top_k": top_k,
                "max_length": max_length,
                "repetition_penalty": repetition_penalty,
                "stop": ["\n\n", "user:", "assistant:"]
            }
        )

        response_text = response["message"]["content"]
        
        # Log the raw response for debugging
        logger.info(f"Raw LLM response: {response_text}")
        
        # Clean up any potential formatting issues
        response_text = response_text.replace('\n\n\n', '\n\n')  # Remove excessive newlines
        response_text = ' '.join(response_text.split())  # Fix spacing issues
        response_text = response_text.replace(' .', '.').replace(' ,', ',')  # Fix punctuation spacing
        
        # Verify response against sources
        if not verify_response(response_text, context):
            if language == "arabic":
                return "عذراً، لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال"
            return "I apologize, but the available sources don't contain sufficient information to provide a complete answer to your question."

        return clean_ai_response(response_text, language)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if language == "arabic":
            return "عذراً، حدث خطأ أثناء معالجة الإجابة"
        return "I apologize, but there was an error processing your request."

def verify_response(response: str, context: str) -> bool:
    """Verify that the response is based on the provided context."""
    # Convert to lowercase for comparison
    response_lower = response.lower()
    context_lower = context.lower()
    
    # Check if response contains key terms from context
    key_terms = set(context_lower.split())
    response_terms = set(response_lower.split())
    
    # Calculate overlap
    overlap = key_terms.intersection(response_terms)
    
    # If less than 20% of key terms are used, response might be generic
    if len(overlap) / len(key_terms) < 0.2:
        return False
    
    # Check for specific numbers and names (case-insensitive)
    numbers = re.findall(r'\d+', context)
    names = re.findall(r'[A-Z][a-z]+', context)
    
    # Response should contain at least some of the specific details
    # But don't require ALL numbers and names
    if numbers and not any(num in response for num in numbers):
        # Only require numbers if they are critical to the answer
        if any(num in context_lower for num in ['$', 'billion', 'million']):
            return False
    if names and not any(name.lower() in response_lower for name in names):
        # Only require names if they are key entities
        if any(name in context_lower for name in ['microsoft', 'amazon', 'google', 'apple']):
            return False
    
    return True

# End of file - Remove any UI-related code that was here