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
        # Convert language code to Azure format
        lang_code = "ar" if language == "arabic" else "en"
        
        # Prepare request
        url = f"{AZURE_LANGUAGE_ENDPOINT}/language/:analyze-text?api-version=2023-04-01"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY
        }
        data = {
            "kind": "EntityRecognition",
            "analysisInput": {
                "documents": [
                    {
                        "id": "1",
                        "text": text,
                        "language": lang_code
                    }
                ]
            }
        }
        
        # Make API call
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Entity extraction failed: {response.text}")
            return {}
            
        result = response.json()
        
        # Process and organize entities by category
        entities = {}
        if 'results' in result and 'documents' in result['results']:
            for doc in result['results']['documents']:
                for entity in doc.get('entities', []):
                    category = entity['category']
                    if category not in entities:
                        entities[category] = []
                    if entity['text'] not in entities[category]:
                        entities[category].append(entity['text'])
        
        logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
        return entities
        
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
        # Convert language code to Azure format
        lang_code = "ar" if language == "arabic" else "en"
        
        # Prepare request
        url = f"{AZURE_LANGUAGE_ENDPOINT}/language/:analyze-text?api-version=2023-04-01"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY
        }
        data = {
            "kind": "KeyPhraseExtraction",
            "analysisInput": {
                "documents": [
                    {
                        "id": "1",
                        "text": text,
                        "language": lang_code
                    }
                ]
            }
        }
        
        # Make API call
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Key phrase extraction failed: {response.text}")
            return []
            
        result = response.json()
        
        # Extract key phrases
        phrases = []
        if 'results' in result and 'documents' in result['results']:
            for doc in result['results']['documents']:
                phrases.extend(doc.get('keyPhrases', []))
        
        logger.info(f"Extracted {len(phrases)} key phrases")
        return phrases
        
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
        # Convert language code to Azure format
        lang_code = "ar" if language == "arabic" else "en"
        
        # Prepare request
        url = f"{AZURE_LANGUAGE_ENDPOINT}/language/:analyze-text?api-version=2023-04-01"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY
        }
        data = {
            "kind": "SentimentAnalysis",
            "analysisInput": {
                "documents": [
                    {
                        "id": "1",
                        "text": text,
                        "language": lang_code
                    }
                ]
            }
        }
        
        # Make API call
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Sentiment analysis failed: {response.text}")
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            
        result = response.json()
        
        # Extract sentiment scores
        sentiment = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        if 'results' in result and 'documents' in result['results']:
            for doc in result['results']['documents']:
                if 'confidenceScores' in doc:
                    sentiment = {
                        "positive": doc['confidenceScores']['positive'],
                        "neutral": doc['confidenceScores']['neutral'],
                        "negative": doc['confidenceScores']['negative']
                    }
                    break
        
        logger.info(f"Sentiment analysis: {sentiment}")
        return sentiment
        
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
    Search for relevant documents using multiple retrieval strategies.
    
    Args:
        query: Search query
        language: Optional language code (ar/en)
        
    Returns:
        List of relevant documents with scores
    """
    try:
        # Detect language if not provided
        if not language:
            lang_result = detect_language_azure(query)
            language = lang_result["language"] if isinstance(lang_result, dict) else "unknown"
            logger.info(f"Detected language: {language}")
        
        # Get collection name based on language
        collection_name = f"rag_docs_{'ar' if language == 'arabic' else 'en'}"
        
        # Extract entities and key phrases from query
        query_entities = extract_entities_azure(query, language)
        query_phrases = extract_key_phrases_azure(query, language)
        query_sentiment = analyze_sentiment_azure(query, language)
        
        logger.info(f"Searching for query: {query}")
        logger.info(f"Using provided language: {language}")
        logger.info(f"Query entities: {query_entities}")
        logger.info(f"Query key phrases: {query_phrases}")
        logger.info(f"Query sentiment: {query_sentiment}")
        
        # Generate query embedding
        query_embedding = generate_embedding(query, language)
        
        # Search in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20
        )
        
        # Process and score results
        scored_results = []
        for result in search_results:
            doc = result.payload
            if not doc:
                continue
                
            # Calculate vector similarity score
            vector_score = result.score
            
            # Calculate entity matching score
            doc_entities = doc.get("entities", {})
            if isinstance(doc_entities, str):
                try:
                    doc_entities = json.loads(doc_entities)
                except:
                    doc_entities = {}
            entity_score = calculate_entity_score(query_entities, doc_entities)
            
            # Calculate key phrase matching score
            doc_phrases = doc.get("key_phrases", [])
            if isinstance(doc_phrases, str):
                try:
                    doc_phrases = json.loads(doc_phrases)
                except:
                    doc_phrases = []
            phrase_score = calculate_key_phrase_score(query_phrases, doc_phrases)
            
            # Calculate sentiment matching score
            doc_sentiment = doc.get("sentiment", {})
            if isinstance(doc_sentiment, str):
                try:
                    doc_sentiment = json.loads(doc_sentiment)
                except:
                    doc_sentiment = {}
            sentiment_score = calculate_sentiment_score(query_sentiment, doc_sentiment)
            
            # Calculate BM25 score
            bm25_score = calculate_bm25_score(query, doc["text"], language)
            
            # Combine scores with weights
            final_score = (
                0.4 * vector_score +
                0.2 * entity_score +
                0.2 * phrase_score +
                0.1 * sentiment_score +
                0.1 * bm25_score
            )
            
            # Create result object with matched entities
            matched_entities = {}
            if isinstance(doc_entities, dict) and isinstance(query_entities, dict):
                for category, entities in doc_entities.items():
                    if category in query_entities:
                        matched_entities[category] = [
                            entity for entity in entities 
                            if entity.lower() in [e.lower() for e in query_entities[category]]
                        ]
            
            scored_result = {
                "text": doc["text"],
                "score": final_score,
                "vector_score": vector_score,
                "entity_score": entity_score,
                "source": doc.get("source", "unknown"),
                "chunk_id": doc.get("chunk_id", 0),
                "total_chunks": doc.get("total_chunks", 1),
                "language": language,
                "matched_entities": matched_entities
            }
            
            scored_results.append(scored_result)
        
        # Sort by final score and return top results
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:10]

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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query>")
        sys.exit(1)
        
    query = sys.argv[1]
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    # Search for documents
    results = search_documents(query)
    
    if not results:
        print("No results found.")
        sys.exit(0)
        
    # Print search results
    print("\nSearch Results:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Vector Score: {result['vector_score']:.4f}")
        print(f"Entity Score: {result['entity_score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Language: {result['language']}")
        if result['matched_entities']:
            print("Matched Entities:")
            for category, entities in result['matched_entities'].items():
                print(f"  {category}: {', '.join(entities)}")
        print("-" * 30)
    
    # Generate and print response
    print("\nGenerated Response:")
    print("-" * 50)
    response = generate_response(query, results)
    print(response)