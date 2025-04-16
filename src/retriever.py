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

def calculate_entity_score(query_entities: Dict[str, List[str]], doc_entities: Dict[str, List[str]]) -> float:
    """Calculate similarity score based on matching entities."""
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
        print(f"Entity score: {score:.2f} (matches: {matches}, total: {total})")
        return score
        
    except Exception as e:
        print(f"Error calculating entity score: {e}")
        return 0.0

# -----------------------------------------------
# 🔹 Function: Search Documents with Hybrid Retrieval
# -----------------------------------------------

def extract_entities_azure(text: str, language: str) -> Dict[str, List[str]]:
    """Extract entities using Azure Text Analytics."""
    try:
        client = init_azure_client()
        if not client:
            print("Failed to initialize Azure client")
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
        
        print(f"Extracted {sum(len(v) for v in all_entities.values())} entities")
        return all_entities
        
    except Exception as e:
        print(f"Error extracting entities with Azure: {e}")
        return {}

def search_documents(query: str, language: str = None) -> List[Dict[str, Any]]:
    """Search for relevant documents in both collections."""
    try:
        print(f"Searching for query: {query}")
        
        # Detect language if not provided
        if not language:
            detection = detect_language_azure(query)
            language = detection["language"]
            print(f"Detected language: {language} (confidence: {detection['confidence']:.2f})")
        else:
            print(f"Using provided language: {language}")
        
        # Extract entities from query
        query_entities = extract_entities_azure(query, "ar" if language == "arabic" else "en")
        print(f"Query entities: {query_entities}")
        
        # Generate query embedding
        query_embedding = generate_embedding(query, language)
        if query_embedding is None:
            print("Failed to generate query embedding")
            return []

        # Search in both collections with increased limit
        results = []
        collection_name = f"rag_docs_{'ar' if language == 'arabic' else 'en'}"
        print(f"Searching in collection: {collection_name}")
        
        # First do vector search
        vector_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,  # Get more candidates
            score_threshold=0.3  # Lower threshold to get more candidates
        )
        
        print(f"Found {len(vector_results)} vector matches")
        
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
                
                # Combine scores with weights
                combined_score = (
                    0.4 * result.score +  # Vector similarity
                    0.3 * bm25_score +    # Text matching
                    0.3 * entity_score     # Entity matching
                )
                
                if combined_score >= 0.3:  # Lower threshold to get more results
                    result_data = {
                        "text": text,
                        "score": combined_score,
                        "vector_score": result.score,
                        "bm25_score": bm25_score,
                        "entity_score": entity_score,
                        "source": payload.get("metadata", {}).get("source", "unknown"),
                        "language": language,
                        "matched_entities": doc_entities,
                        "chunk_index": payload.get("metadata", {}).get("chunk_index", 0),
                        "total_chunks": payload.get("metadata", {}).get("total_chunks", 0)
                    }
                    results.append(result_data)
                    print(f"Added result with combined score: {combined_score:.2f}")

        # Sort by combined score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        print(f"Returning {len(results)} results")
        return results[:3]  # Return only top 3 most relevant results

    except Exception as e:
        print(f"Error searching documents: {e}")
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

def generate_response(query: str, results: List[Dict[str, Any]]) -> str:
    """Generate a response using the retrieved documents."""
    try:
        print(f"Generating response for query: {query}")
        print(f"Number of results to process: {len(results)}")
        
        if not results:
            print("No results found to generate response")
            return "لم أتمكن من العثور على معلومات محددة حول هذا الموضوع في الوثائق المتاحة" if any('\u0600' <= char <= '\u06FF' for char in query) else "I couldn't find specific information about this in the available documents"
        
        # Prepare context from top 3 relevant sources
        context_parts = []
        for i, result in enumerate(results[:3], 1):
            source_info = f"Source {i} (Score: {result.get('score', 0):.2f}, Document: {result.get('source', 'unknown')})"
            context_parts.append(f"{source_info}:\n{result.get('text', '')}\n")
        
        context = "\n".join(context_parts)
        print("Context prepared for response generation")

        # Use gemma3:1b model for both languages
        model = "gemma3:1b"
        print(f"Using model: {model}")

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
                "content": f"""بناءً على المصادر التالية، قدم إجابة مباشرة على السؤال. استخدم فقط المعلومات الواردة في هذه المصادر ورد بنفس لغة السؤال.

السؤال: {query}

المصادر:
{context}

تنسيق الإجابة:
- يجب أن تكون الإجابة جملة واحدة واضحة
- استخدم فقط المعلومات من هذه المصادر
- ركز على المعلومات المحددة والأرقام والحقائق
- رد بنفس لغة السؤال
- لا تكرر السؤال في الإجابة
- لا تضيف أي معلومات غير موجودة في المصادر
- لا تذكر مصادر أو معلومات أخرى
- استخدم لغة عربية واضحة وسهلة الفهم
- إذا كانت المعلومات موجودة في أكثر من مصدر، قم بدمجها بشكل منطقي
- إذا لم تجد معلومات كافية، قل "لم أتمكن من العثور على معلومات محددة حول هذا الموضوع في الوثائق المتاحة"

مثال على الإجابة الصحيحة:
السؤال: ما هي قيمة شراكة مايكروسوفت وG42 في الإمارات؟
الإجابة: أعلنت مايكروسوفت عن استثمار بقيمة 1.5 مليار دولار في شراكة مع G42 لتطوير الذكاء الاصطناعي في الإمارات.

الإجابة:""" if is_arabic else f"""Based on the following sources, provide a direct answer to the question. Use only the information provided in these sources and respond in the same language as the question.

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

Example of a good answer:
Question: What is the value of Microsoft's partnership with G42 in the UAE?
Answer: Microsoft announced a $1.5 billion investment in a partnership with G42 to develop AI in the UAE.

Answer:"""
            }]
        )

        # Verify response is based on context
        response_text = response['message']['content'].strip()
        if not verify_response(response_text, context):
            return "لم أتمكن من العثور على معلومات كافية في المصادر للإجابة على سؤالك بدقة." if is_arabic else "I couldn't find enough information in the sources to answer your question accurately."

        print("Response generated successfully")
        return response_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return "عذراً، حدث خطأ أثناء معالجة طلبك" if is_arabic else "I apologize, but there was an error processing your request."

def verify_response(response: str, context: str) -> bool:
    """Verify that the response is based on the provided context."""
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

ENGLISH_PROMPT_TEMPLATE = """You are a helpful AI assistant. Based on the following context, provide a detailed and specific answer to the question. Include relevant facts, figures, and specific initiatives when available.

Context:
{context}

Question: {question}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Includes specific details and examples
3. Cites relevant facts and figures
4. Explains the significance of the initiatives
5. Uses clear and professional language

Answer:"""

ARABIC_PROMPT_TEMPLATE = """أنت مساعد ذكي مفيد. بناءً على السياق التالي، قدم إجابة مفصلة ومحددة للسؤال. أدرج الحقائق والأرقام والمبادرات المحددة عند توفرها.

السياق:
{context}

السؤال: {question}

يرجى تقديم إجابة شاملة:
1. تجيب مباشرة على السؤال
2. تتضمن تفاصيل وأمثلة محددة
3. تستشهد بالحقائق والأرقام ذات الصلة
4. تشرح أهمية المبادرات
5. تستخدم لغة واضحة ومهنية

الإجابة:"""

def calculate_bm25_score(query: str, text: str, language: str) -> float:
    """Calculate BM25 score between query and text."""
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
        print(f"Error calculating BM25 score: {e}")
        return 0.0

def init_azure_client():
    """Initialize Azure Text Analytics client."""
    try:
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        key = os.getenv("AZURE_LANGUAGE_KEY")
        credential = AzureKeyCredential(key)
        client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
        return client
    except Exception as e:
        print(f"Error initializing Azure client: {e}")
        return None

def detect_language_azure(text: str) -> Dict[str, Any]:
    """Detect language using Azure Text Analytics."""
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
        print(f"Error detecting language with Azure: {e}")
        return {"language": "unknown", "confidence": 0.0}

# End of file - Remove any UI-related code that was here