"""
Test suite for the document retriever.

This module contains tests for:
- Language detection
- Query processing
- Document retrieval
- Scoring mechanisms
- Response generation
"""

import os
import pytest
from src.retriever import (
    detect_language_azure,
    generate_embedding,
    search_documents,
    generate_response,
    calculate_entity_score,
    calculate_key_phrase_score,
    calculate_sentiment_score,
    calculate_bm25_score
)
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test data
TEST_QUERIES = [
    {
        "text": "What is Microsoft's investment in G42?",
        "language": "english",
        "expected_entities": ["Microsoft", "G42"],
        "expected_phrases": ["Microsoft investment", "G42", "cloud computing"],
        "expected_content": ["1.5 billion", "UAE"]
    },
    {
        "text": "ما هي استثمارات جوجل في الإمارات؟",
        "language": "arabic",
        "expected_entities": ["جوجل", "الإمارات"],
        "expected_phrases": ["استثمارات جوجل", "الذكاء الاصطناعي"],
        "expected_content": ["1.2 مليار", "دبي"]
    }
]

@pytest.fixture
def qdrant_client():
    """Initialize Qdrant client for testing."""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

def test_language_detection():
    """Test language detection for different queries."""
    for query in TEST_QUERIES:
        detection = detect_language_azure(query["text"])
        assert detection["language"] == query["language"]
        assert detection["confidence"] > 0.7

def test_embedding_generation():
    """Test embedding generation for queries."""
    for query in TEST_QUERIES:
        embedding = generate_embedding(query["text"], query["language"])
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

def test_document_retrieval(qdrant_client):
    """Test document retrieval with different queries."""
    for query in TEST_QUERIES:
        results = search_documents(query["text"], query["language"])
        assert isinstance(results, list)
        if results:  # If we have indexed documents
            assert len(results) > 0
            for result in results:
                assert "text" in result
                assert "score" in result
                assert "vector_score" in result
                assert "entity_score" in result
                assert "key_phrase_score" in result
                assert "sentiment_score" in result

def test_scoring_functions():
    """Test individual scoring functions."""
    # Test entity scoring
    query_entities = {"Organization": ["Microsoft", "G42"]}
    doc_entities = {"Organization": ["Microsoft", "G42", "UAE"]}
    entity_score = calculate_entity_score(query_entities, doc_entities)
    assert 0 <= entity_score <= 1
    
    # Test key phrase scoring
    query_phrases = ["Microsoft investment", "G42"]
    doc_phrases = ["Microsoft investment", "G42", "UAE"]
    phrase_score = calculate_key_phrase_score(query_phrases, doc_phrases)
    assert 0 <= phrase_score <= 1
    
    # Test sentiment scoring
    query_sentiment = {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
    doc_sentiment = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
    sentiment_score = calculate_sentiment_score(query_sentiment, doc_sentiment)
    assert 0 <= sentiment_score <= 1
    
    # Test BM25 scoring
    query = "Microsoft investment G42"
    text = "Microsoft announced a $1.5 billion investment in G42 to develop AI in the UAE."
    bm25_score = calculate_bm25_score(query, text, "english")
    assert 0 <= bm25_score <= 1

def test_response_generation():
    """Test response generation with retrieved documents."""
    for query in TEST_QUERIES:
        # First get some results
        results = search_documents(query["text"], query["language"])
        
        # Generate response
        response = generate_response(query["text"], results)
        
        # Basic response validation
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check if response contains expected content
        if results:  # If we have results
            # Check for expected entities
            found_entities = 0
            for entity in query["expected_entities"]:
                if entity.lower() in response.lower():
                    found_entities += 1
            assert found_entities > 0, f"No expected entities found in response for query: {query['text']}"
            
            # Check for expected content
            found_content = 0
            for content in query["expected_content"]:
                if content.lower() in response.lower():
                    found_content += 1
            assert found_content > 0, f"No expected content found in response for query: {query['text']}"

def test_combined_scoring():
    """Test the combined scoring mechanism."""
    for query in TEST_QUERIES:
        results = search_documents(query["text"], query["language"])
        if results:  # If we have results
            for result in results:
                # Check all score components
                assert 0 <= result["score"] <= 1
                assert 0 <= result["vector_score"] <= 1
                assert 0 <= result["entity_score"] <= 1
                assert 0 <= result["key_phrase_score"] <= 1
                assert 0 <= result["sentiment_score"] <= 1
                
                # Verify combined score is weighted average
                expected_score = (
                    0.3 * result["vector_score"] +
                    0.3 * result["bm25_score"] +
                    0.2 * result["entity_score"] +
                    0.1 * result["key_phrase_score"] +
                    0.1 * result["sentiment_score"]
                )
                assert abs(result["score"] - expected_score) < 0.01 