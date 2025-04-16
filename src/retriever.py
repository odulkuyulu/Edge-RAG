"""
Edge RAG Retriever System
=======================

This module implements a hybrid document retrieval system using Azure AI Language services
and vector similarity search. It combines multiple retrieval strategies to find the most
relevant documents for a given query.

Key Features:
- Language detection and handling (Arabic/English)
- Vector similarity search using embeddings
- BM25 text matching
- Entity recognition and matching
- Key phrase extraction and matching
- Sentiment analysis and matching

Azure AI Language Services:
- Language Detection
- Named Entity Recognition (NER)
- Key Phrase Extraction
- Sentiment Analysis
"""

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
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import json

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download("punkt")

# =============================================================================
# Configuration and Constants
# =============================================================================

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Embedding Model Configuration
EMBEDDING_SIZES = {
    "english": 1024,  # bge-m3 (Optimized for retrieval)
    "arabic": 1024,   # bge-m3 embeddings
}

# Azure AI Language Service Configuration
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

if not AZURE_LANGUAGE_ENDPOINT or not AZURE_LANGUAGE_KEY:
    raise ValueError("Azure Language Service configuration missing. Please set AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class RetrieverConfig:
    """Configuration for the retriever system."""
    qdrant_url: str
    qdrant_api_key: str
    azure_language_endpoint: str
    azure_language_key: str
    embedding_size: int = 1024
    max_results: int = 10
    score_threshold: float = 0.0

class SearchResult(BaseModel):
    """Model for search results with scoring information."""
    text: str
    score: float
    vector_score: float
    entity_score: float
    source: str
    chunk_id: int
    total_chunks: int
    language: str
    matched_entities: Dict[str, List[str]]

# =============================================================================
# Azure AI Language Service Integration
# =============================================================================

def init_azure_client() -> TextAnalyticsClient:
    """
    Initialize Azure Text Analytics client.
    
    Returns:
        TextAnalyticsClient: Initialized Azure client or None if initialization fails
    """
    try:
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        key = os.getenv("AZURE_LANGUAGE_KEY")
        credential = AzureKeyCredential(key)
        client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
        return client
    except Exception as e:
        logger.error(f"Error initializing Azure client: {e}")
        return None

def detect_language_azure(text: str) -> Dict[str, Any]:
    """
    Detect language using Azure Text Analytics.
    
    Args:
        text: Input text to detect language for
        
    Returns:
        Dict containing detected language and confidence score
    """
    try:
        client = init_azure_client()
        if not client:
            return {"language": "unknown", "confidence": 0.0}
            
        # Split text into chunks if too long (Azure has a limit)
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        detected_languages = []
        
        for chunk in chunks:
            result = client.detect_language([chunk])[0]
            if not result.is_error:
                detected_languages.append({
                    "language": result.primary_language.iso6391_name,
                    "confidence": result.primary_language.confidence_score
                })
        
        if not detected_languages:
            return {"language": "unknown", "confidence": 0.0}
            
        # Get the most confident detection
        best_detection = max(detected_languages, key=lambda x: x["confidence"])
        
        # Map Azure language codes to our system
        language_map = {
            "ar": "arabic",
            "en": "english"
        }
        
        return {
            "language": language_map.get(best_detection["language"], "unknown"),
            "confidence": best_detection["confidence"]
        }
        
    except Exception as e:
        logger.error(f"Error detecting language with Azure: {e}")
        return {"language": "unknown", "confidence": 0.0}

def extract_entities_azure(text: str, language: str) -> Dict[str, List[str]]:
    """
    Extract entities using Azure Text Analytics.
    
    Args:
        text: Input text to extract entities from
        language: Language code (ar/en)
        
    Returns:
        Dict of entity categories and their values
    """
    try:
        client = init_azure_client()
        if not client:
            logger.error("Failed to initialize Azure client")
            return {}
            
        # Split text into chunks if too long
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        all_entities = {}
        
        for chunk in chunks:
            result = client.recognize_entities(
                [chunk],
                language=language
            )[0]
            
            if not result.is_error:
                for entity in result.entities:
                    category = entity.category
                    if category not in all_entities:
                        all_entities[category] = []
                    if entity.text not in all_entities[category]:
                        all_entities[category].append(entity.text)
        
        logger.info(f"Extracted {sum(len(v) for v in all_entities.values())} entities")
        return all_entities
        
    except Exception as e:
        logger.error(f"Error extracting entities with Azure: {e}")
        return {}

def extract_key_phrases_azure(text: str, language: str) -> List[str]:
    """
    Extract key phrases using Azure Text Analytics.
    
    Args:
        text: Input text to extract phrases from
        language: Language code (ar/en)
        
    Returns:
        List of extracted key phrases
    """
    try:
        client = init_azure_client()
        if not client:
            logger.error("Failed to initialize Azure client")
            return []
            
        # Split text into chunks if too long
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        all_phrases = []
        
        for chunk in chunks:
            result = client.extract_key_phrases(
                [chunk],
                language=language
            )[0]
            
            if not result.is_error:
                all_phrases.extend(result.key_phrases)
        
        logger.info(f"Extracted {len(all_phrases)} key phrases")
        return all_phrases
        
    except Exception as e:
        logger.error(f"Error extracting key phrases with Azure: {e}")
        return []

def analyze_sentiment_azure(text: str, language: str) -> Dict[str, float]:
    """
    Analyze sentiment using Azure Text Analytics.
    
    Args:
        text: Input text to analyze
        language: Language code (ar/en)
        
    Returns:
        Dict with sentiment scores (positive, neutral, negative)
    """
    try:
        client = init_azure_client()
        if not client:
            logger.error("Failed to initialize Azure client")
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            
        # Split text into chunks if too long
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        total_sentiment = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        chunk_count = 0
        
        for chunk in chunks:
            result = client.analyze_sentiment(
                [chunk],
                language=language
            )[0]
            
            if not result.is_error:
                total_sentiment["positive"] += result.confidence_scores.positive
                total_sentiment["neutral"] += result.confidence_scores.neutral
                total_sentiment["negative"] += result.confidence_scores.negative
                chunk_count += 1
        
        if chunk_count > 0:
            # Average the sentiment scores
            total_sentiment = {k: v/chunk_count for k, v in total_sentiment.items()}
        
        logger.info(f"Sentiment analysis: {total_sentiment}")
        return total_sentiment
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment with Azure: {e}")
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

# =============================================================================
# Scoring Functions
# =============================================================================

def calculate_entity_score(query_entities: Dict[str, List[str]], doc_entities: Dict[str, List[str]]) -> float:
    """
    Calculate similarity score based on matching entities.
    
    Args:
        query_entities: Entities from the query
        doc_entities: Entities from the document
        
    Returns:
        Float score between 0 and 1
    """
    try:
        if not query_entities or not doc_entities:
            return 0.0
            
        # Convert doc_entities to the correct format if it's a string
        if isinstance(doc_entities, str):
            try:
                doc_entities = json.loads(doc_entities)
            except:
                return 0.0
            
        # Count matching entities
        matches = 0
        total = sum(len(v) for v in query_entities.values())
        
        if total == 0:
            return 0.0
            
        for category, entities in query_entities.items():
            if category in doc_entities:
                doc_entity_texts = [e.lower() for e in doc_entities[category]]
                for entity in entities:
                    if entity.lower() in doc_entity_texts:
                        matches += 1
                        
        # Calculate score
        score = matches / total if total > 0 else 0.0
        logger.info(f"Entity score: {score:.2f} (matches: {matches}, total: {total})")
        return score
        
    except Exception as e:
        logger.error(f"Error calculating entity score: {e}")
        return 0.0

def calculate_key_phrase_score(query_phrases: List[str], doc_phrases: List[str]) -> float:
    """
    Calculate similarity score based on matching key phrases.
    
    Args:
        query_phrases: Key phrases from the query
        doc_phrases: Key phrases from the document
        
    Returns:
        Float score between 0 and 1
    """
    try:
        if not query_phrases or not doc_phrases:
            return 0.0
            
        # Convert doc_phrases to list if it's a string
        if isinstance(doc_phrases, str):
            try:
                doc_phrases = json.loads(doc_phrases)
            except:
                return 0.0
            
        # Count matching phrases
        matches = 0
        total = len(query_phrases)
        
        if total == 0:
            return 0.0
            
        query_phrases_lower = [p.lower() for p in query_phrases]
        doc_phrases_lower = [p.lower() for p in doc_phrases]
        
        for phrase in query_phrases_lower:
            if phrase in doc_phrases_lower:
                matches += 1
                
        # Calculate score
        score = matches / total if total > 0 else 0.0
        logger.info(f"Key phrase score: {score:.2f} (matches: {matches}, total: {total})")
        return score
        
    except Exception as e:
        logger.error(f"Error calculating key phrase score: {e}")
        return 0.0

def calculate_sentiment_score(query_sentiment: Dict[str, float], doc_sentiment: Dict[str, float]) -> float:
    """
    Calculate similarity score based on sentiment matching.
    
    Args:
        query_sentiment: Sentiment scores from the query
        doc_sentiment: Sentiment scores from the document
        
    Returns:
        Float score between 0 and 1
    """
    try:
        if not query_sentiment or not doc_sentiment:
            return 0.0
            
        # Convert doc_sentiment to dict if it's a string
        if isinstance(doc_sentiment, str):
            try:
                doc_sentiment = json.loads(doc_sentiment)
            except:
                return 0.0
            
        # Calculate sentiment similarity
        # We'll use a simple weighted average of the differences
        weights = {"positive": 0.4, "neutral": 0.2, "negative": 0.4}
        total_diff = 0.0
        total_weight = 0.0
        
        for sentiment, weight in weights.items():
            if sentiment in query_sentiment and sentiment in doc_sentiment:
                diff = 1.0 - abs(query_sentiment[sentiment] - doc_sentiment[sentiment])
                total_diff += diff * weight
                total_weight += weight
                
        score = total_diff / total_weight if total_weight > 0 else 0.0
        logger.info(f"Sentiment score: {score:.2f}")
        return score
        
    except Exception as e:
        logger.error(f"Error calculating sentiment score: {e}")
        return 0.0

def calculate_bm25_score(query: str, text: str, language: str) -> float:
    """
    Calculate BM25 score between query and text.
    
    Args:
        query: Search query
        text: Document text
        language: Language code (ar/en)
        
    Returns:
        Float score between 0 and 1
    """
    try:
        # Tokenize the text and query
        text_tokens = word_tokenize(text.lower())
        query_tokens = word_tokenize(query.lower())
        
        # Create BM25 index
        bm25 = BM25Okapi([text_tokens])
        
        # Calculate score
        score = bm25.get_scores(query_tokens)[0]
        
        # Normalize score to 0-1 range
        normalized_score = min(1.0, max(0.0, score))
        
        return normalized_score
        
    except Exception as e:
        logger.error(f"Error calculating BM25 score: {e}")
        return 0.0

# =============================================================================
# Core Retrieval Functions
# =============================================================================

@lru_cache(maxsize=1000)
def generate_embedding(text: str, language: str) -> List[float]:
    """
    Generate embeddings using different models for Arabic & English queries.
    
    Args:
        text: Input text
        language: Language code (ar/en)
        
    Returns:
        List of embedding values
    """
    model_name = "bge-m3" if language == "arabic" else "bge-m3"
    
    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    # Ensure embedding size matches Qdrant expectations
    expected_size = EMBEDDING_SIZES[language]

    # Ensure embedding has the correct size
    if len(embedding) < expected_size:
        embedding = np.pad(embedding, (0, expected_size - len(embedding)), 'constant')
    elif len(embedding) > expected_size:
        embedding = embedding[:expected_size]  # Truncate if larger

    return list(embedding)

def search_documents(query: str, language: str = None) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using hybrid retrieval.
    
    This function implements a multi-stage retrieval process:
    1. Language detection
    2. Entity and key phrase extraction
    3. Sentiment analysis
    4. Vector similarity search
    5. BM25 text matching
    6. Combined scoring and ranking
    
    Args:
        query: Search query
        language: Optional language code (ar/en)
        
    Returns:
        List of relevant documents with scores and metadata
    """
    try:
        logger.info(f"Searching for query: {query}")
        
        # Detect language if not provided
        if not language:
            detection = detect_language_azure(query)
            language = detection["language"]
            logger.info(f"Detected language: {language} (confidence: {detection['confidence']:.2f})")
        else:
            logger.info(f"Using provided language: {language}")
        
        # Extract entities, key phrases, and sentiment from query
        query_entities = extract_entities_azure(query, "ar" if language == "arabic" else "en")
        logger.info(f"Query entities: {query_entities}")
        
        query_phrases = extract_key_phrases_azure(query, "ar" if language == "arabic" else "en")
        logger.info(f"Query key phrases: {query_phrases}")
        
        query_sentiment = analyze_sentiment_azure(query, "ar" if language == "arabic" else "en")
        logger.info(f"Query sentiment: {query_sentiment}")
        
        # Generate query embedding
        query_embedding = generate_embedding(query, language)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []

        # Search in appropriate collection
        results = []
        collection_name = f"rag_docs_{'ar' if language == 'arabic' else 'en'}"
        logger.info(f"Searching in collection: {collection_name}")
        
        # First do vector search
        vector_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,  # Get more candidates
            score_threshold=0.1  # Lower threshold to get more candidates
        )
        
        logger.info(f"Found {len(vector_results)} vector matches")
        
        # Process and score results
        for result in vector_results:
            payload = result.payload
            if payload and "text" in payload:
                # Calculate BM25 score
                text = payload["text"]
                bm25_score = calculate_bm25_score(query, text, language)
                
                # Calculate entity match score
                doc_entities = payload.get("metadata", {}).get("entities", {})
                entity_score = calculate_entity_score(query_entities, doc_entities)
                
                # Calculate key phrase match score
                doc_phrases = payload.get("metadata", {}).get("key_phrases", [])
                logger.info(f"Document phrases: {doc_phrases}")
                key_phrase_score = calculate_key_phrase_score(query_phrases, doc_phrases)
                
                # Calculate sentiment match score
                doc_sentiment = payload.get("metadata", {}).get("sentiment", {})
                sentiment_score = calculate_sentiment_score(query_sentiment, doc_sentiment)
                
                # Combine scores with weights
                combined_score = (
                    0.3 * result.score +      # Vector similarity
                    0.3 * bm25_score +        # Text matching
                    0.2 * entity_score +      # Entity matching
                    0.1 * key_phrase_score +  # Key phrase matching
                    0.1 * sentiment_score     # Sentiment matching
                )
                
                if combined_score >= 0.1:  # Lower threshold to get more results
                    result_data = {
                        "text": text,
                        "score": combined_score,
                        "vector_score": result.score,
                        "bm25_score": bm25_score,
                        "entity_score": entity_score,
                        "key_phrase_score": key_phrase_score,
                        "sentiment_score": sentiment_score,
                        "source": payload.get("metadata", {}).get("source", "unknown"),
                        "language": language,
                        "matched_entities": doc_entities,
                        "matched_phrases": doc_phrases,
                        "sentiment": doc_sentiment,
                        "chunk_index": payload.get("metadata", {}).get("chunk_index", 0),
                        "total_chunks": payload.get("metadata", {}).get("total_chunks", 0)
                    }
                    results.append(result_data)
                    logger.info(f"Added result with combined score: {combined_score:.2f}")
                    logger.info(f"Vector score: {result.score:.2f}")
                    logger.info(f"BM25 score: {bm25_score:.2f}")
                    logger.info(f"Entity score: {entity_score:.2f}")
                    logger.info(f"Key phrase score: {key_phrase_score:.2f}")
                    logger.info(f"Sentiment score: {sentiment_score:.2f}")

        # Sort by combined score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Returning {len(results)} results")
        return results[:3]  # Return only top 3 most relevant results

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

# =============================================================================
# Response Generation
# =============================================================================

def generate_response(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Generate a response using the retrieved documents.
    
    Args:
        query: Original search query
        results: List of retrieved documents
        
    Returns:
        Generated response text
    """
    try:
        logger.info(f"Generating response for query: {query}")
        logger.info(f"Number of results to process: {len(results)}")
        
        if not results:
            logger.warning("No results found to generate response")
            return "لم أتمكن من العثور على معلومات محددة حول هذا الموضوع في الوثائق المتاحة" if any('\u0600' <= char <= '\u06FF' for char in query) else "I couldn't find specific information about this in the available documents"
        
        # Prepare context from top 3 relevant sources
        context_parts = []
        for i, result in enumerate(results[:3], 1):
            source_info = f"Source {i} (Score: {result.get('score', 0):.2f}, Document: {result.get('source', 'unknown')})"
            context_parts.append(f"{source_info}:\n{result.get('text', '')}\n")
        
        context = "\n".join(context_parts)
        logger.info("Context prepared for response generation")

        # Use gemma3:1b model for both languages
        model = "gemma3:1b"
        logger.info(f"Using model: {model}")

        # Determine if the query is in Arabic
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in query)

        # Generate response using Ollama
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "system",
                "content": """أنت مساعد ذكي دقيق. مهمتك هي تقديم إجابة واضحة ومباشرة بناءً على المصادر المقدمة. يجب أن ترد بنفس لغة السؤال ولا تضيف أي معلومات غير موجودة في المصادر. لا تكرر السؤال في الإجابة. استخدم لغة عربية واضحة وسهلة الفهم. إذا كانت المعلومات موجودة في أكثر من مصدر، قم بدمجها بشكل منطقي في إجابة واحدة. ركز على المعلومات المحددة والأرقام والحقائق المذكورة في المصادر. إذا لم تجد معلومات كافية، قل ذلك بوضوح."""
            } if is_arabic else {
                "role": "system",
                "content": """You are a precise fact-checking assistant. Your task is to provide a single, clear answer based on the provided sources. Respond in the same language as the question. Do not add any information not present in the sources. Do not repeat the question in your answer. If information exists across multiple sources, combine it logically into a single answer. Focus on specific information, numbers, and facts mentioned in the sources. If you don't find sufficient information, clearly state that."""
            },
            {
                "role": "user",
                "content": f"""Based on the following sources, provide a direct answer to the question. Use only the information provided in these sources and respond in the same language as the question.

Question: {query}

Sources:
{context}

Answer format:
- Answer must be a single, clear sentence
- Use only the information from these sources
- Focus on specific information, numbers, and facts
- Respond in the same language as the question
- Do not repeat the question in your answer
- Do not add any information not in the sources
- Do not mention other sources or information
- If information exists across multiple sources, combine it logically
- If you don't find sufficient information, say "I couldn't find specific information about this in the available documents"

Answer:"""
            }]
        )

        # Verify response is based on context
        response_text = response['message']['content'].strip()
        if not verify_response(response_text, context):
            return "لم أتمكن من العثور على معلومات كافية في المصادر للإجابة على سؤالك بدقة." if is_arabic else "I couldn't find enough information in the sources to answer your question accurately."

        logger.info("Response generated successfully")
        return response_text

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "عذراً، حدث خطأ أثناء معالجة طلبك" if is_arabic else "I apologize, but there was an error processing your request."

def verify_response(response: str, context: str) -> bool:
    """
    Verify that the response is based on the provided context.
    
    Args:
        response: Generated response text
        context: Source context used for generation
        
    Returns:
        Boolean indicating if response is valid
    """
    if not response or not context:
        return False
        
    # Convert to lowercase for comparison
    response_lower = response.lower()
    context_lower = context.lower()
    
    # Extract key phrases (2-3 word combinations)
    def get_phrases(text):
        words = text.split()
        phrases = []
        for i in range(len(words)-1):
            phrases.append(" ".join(words[i:i+2]))
        for i in range(len(words)-2):
            phrases.append(" ".join(words[i:i+3]))
        return phrases
    
    context_phrases = get_phrases(context_lower)
    response_phrases = get_phrases(response_lower)
    
    # Check for phrase overlap
    matching_phrases = set(context_phrases) & set(response_phrases)
    if len(matching_phrases) > 0:
        return True
    
    # Fallback to key terms if no phrases match
    key_terms = set(context_lower.split())
    response_terms = set(response_lower.split())
    
    # Calculate overlap
    overlap = key_terms.intersection(response_terms)
    
    # Very lenient threshold
    return len(overlap) > 2  # Only require 3 matching terms