"""
Test suite for the document indexer.

This module contains tests for:
- Document processing
- Language detection
- Entity extraction
- Embedding generation
- Document indexing in Qdrant
"""

import os
import pytest
from src.indexer import (
    process_document,
    detect_language,
    extract_entities,
    generate_embedding,
    index_document,
    process_with_document_intelligence
)
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test data matching actual content from data files
SAMPLE_DOCUMENTS = [
    {
        "text": "G42, a leading UAE-based artificial intelligence and cloud computing company, has entered into a strategic partnership with Microsoft to accelerate digital transformation in the UAE and the broader region. The partnership includes a $1.5 billion investment from Microsoft to support G42's AI initiatives and cloud infrastructure development.",
        "source": "ai_en.txt",
        "language": "english",
        "expected_entities": ["G42", "Microsoft", "UAE"],
        "expected_phrases": ["strategic partnership", "digital transformation", "cloud computing"],
        "expected_content": ["$1.5 billion", "artificial intelligence"]
    },
    {
        "text": "أعلنت شركة جوجل عن استثمارات جديدة في الإمارات العربية المتحدة بقيمة 1.2 مليار دولار لدعم تطوير الذكاء الاصطناعي والتحول الرقمي في المنطقة. هذه الاستثمارات تأتي في إطار رؤية الإمارات لتصبح مركزاً عالمياً للابتكار التكنولوجي.",
        "source": "ai_ar.txt",
        "language": "arabic",
        "expected_entities": ["جوجل", "الإمارات العربية المتحدة"],
        "expected_phrases": ["الذكاء الاصطناعي", "التحول الرقمي"],
        "expected_content": ["1.2 مليار دولار", "مركزاً عالمياً"]
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
    """Test language detection for different texts."""
    for doc in SAMPLE_DOCUMENTS:
        detected_lang = detect_language(doc["text"])
        assert detected_lang == doc["language"]

def test_entity_extraction():
    """Test entity extraction for different languages."""
    for doc in SAMPLE_DOCUMENTS:
        entities = extract_entities(doc["text"], doc["language"])
        assert isinstance(entities, list)
        
        # Check for expected entities
        found_entities = 0
        for expected_entity in doc["expected_entities"]:
            if any(entity["text"] == expected_entity for entity in entities):
                found_entities += 1
        
        # At least 50% of expected entities should be found
        min_expected = len(doc["expected_entities"]) // 2
        assert found_entities >= min_expected, f"Found only {found_entities} entities out of {len(doc['expected_entities'])} expected"

def test_embedding_generation():
    """Test embedding generation for different languages."""
    for doc in SAMPLE_DOCUMENTS:
        embedding = generate_embedding(doc["text"], doc["language"])
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

def test_document_processing():
    """Test document processing with different languages."""
    for doc in SAMPLE_DOCUMENTS:
        processed = process_document(doc["text"], doc["source"])
        assert isinstance(processed, list)
        assert len(processed) > 0
        for chunk in processed:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "embedding" in chunk

def test_document_indexing(qdrant_client):
    """Test document indexing in Qdrant."""
    for doc in SAMPLE_DOCUMENTS:
        # Process document
        processed = process_document(doc["text"], doc["source"])
        
        # Index document
        collection_name = f"rag_docs_{'ar' if doc['language'] == 'arabic' else 'en'}"
        success = index_document(doc["source"], doc["text"])
        assert success
        
        # Verify document was indexed using query_points
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=generate_embedding(doc["text"], doc["language"]),
            limit=1
        )
        # Check that we have results and the text matches
        assert hasattr(search_results, 'points'), "No results found in search response"
        assert len(search_results.points) > 0, "No points found in search results"
        assert search_results.points[0].payload["text"] in doc["text"], f"Expected text not found in result: {search_results.points[0].payload['text']}"

def test_document_intelligence():
    """Test document intelligence processing."""
    # Create a temporary test file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(SAMPLE_DOCUMENTS[0]["text"].encode('utf-8'))
        temp_file_path = temp_file.name
    
    try:
        result = process_with_document_intelligence(temp_file_path)
        assert result is not None
        assert "text" in result
        assert "tables" in result
        assert "key_value_pairs" in result
        assert "entities" in result
        assert "metadata" in result
    finally:
        os.unlink(temp_file_path) 