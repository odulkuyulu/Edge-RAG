import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.retriever import (
    detect_language,
    extract_entities,
    search_documents,
    generate_response
)

def test_retriever():
    """Comprehensive test of the retriever functionality."""
    # Load environment variables
    load_dotenv()
    
    print("\n🚀 Starting Retriever Tests...")
    
    # Test 1: Language Detection
    print("\n📝 Test 1: Language Detection")
    test_texts = [
        ("What is artificial intelligence?", "english"),
        ("ما هو الذكاء الاصطناعي؟", "arabic"),
        ("AI is transforming industries worldwide", "english"),
        ("الذكاء الاصطناعي يغير العالم", "arabic")
    ]
    
    for text, expected in test_texts:
        detected = detect_language(text)
        print(f"Text: {text[:30]}...")
        print(f"Expected: {expected}, Detected: {detected}")
        print(f"Result: {'✅' if detected == expected else '❌'}")
    
    # Test 2: Entity Extraction
    print("\n🔍 Test 2: Entity Extraction")
    test_queries = [
        ("Microsoft and OpenAI announced new AI features", "english"),
        ("أعلنت مايكروسوفت وأوبن إيه آي عن ميزات جديدة", "arabic")
    ]
    
    for query, lang in test_queries:
        entities = extract_entities(query, lang)
        print(f"\nQuery: {query}")
        print("Entities found:")
        for entity in entities:
            print(f"- {entity['text']} ({entity['category']})")
    
    # Test 3: Document Search
    print("\n🔎 Test 3: Document Search")
    test_searches = [
        ("What is AI?", "english"),
        ("ما هو الذكاء الاصطناعي؟", "arabic"),
        ("Latest developments in AI", "english"),
        ("أحدث تطورات الذكاء الاصطناعي", "arabic")
    ]
    
    for query, lang in test_searches:
        print(f"\nSearching for: {query}")
        results = search_documents(query, lang)
        print(f"Found {len(results)} results")
        if results:
            print("\nTop result:")
            print(f"Text: {results[0]['text'][:100]}...")
            print(f"Score: {results[0]['score']:.2f}")
            print(f"Source: {results[0]['source']}")
    
    # Test 4: Response Generation
    print("\n💬 Test 4: Response Generation")
    test_questions = [
        ("What is artificial intelligence?", "english"),
        ("ما هو الذكاء الاصطناعي؟", "arabic")
    ]
    
    for question, lang in test_questions:
        print(f"\nQuestion: {question}")
        results = search_documents(question, lang)
        response = generate_response(question, results)
        print("\nResponse:")
        print(response)
    
    print("\n✨ Retriever Tests Completed!")

if __name__ == "__main__":
    test_retriever() 