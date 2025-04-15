import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.indexer import load_documents, index_document, process_document
from dotenv import load_dotenv

def test_indexer():
    """Test the document indexing functionality."""
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting indexer test...")
    
    # Test 1: Load documents from data directory
    print("\n📂 Test 1: Loading documents from data directory...")
    documents = load_documents()
    print(f"✅ Found {len(documents)} documents to process")
    
    # Test 2: Process and index each document
    print("\n📄 Test 2: Processing and indexing documents...")
    for doc in documents:
        print(f"\nProcessing document: {doc.get('filename', 'Unknown')}")
        
        # Process the document
        chunks = process_document(doc["text"], doc.get("filename"))
        print(f"✅ Document split into {len(chunks)} chunks")
        
        # Index the document
        index_document(doc["text"], doc.get("filename"))
        print("✅ Document indexed successfully")
        
        # Print some metadata from the first chunk
        if chunks:
            first_chunk = chunks[0]
            print("\nFirst chunk metadata:")
            print(f"Language: {first_chunk['metadata'].get('language', 'unknown')}")
            print(f"Source: {first_chunk['metadata'].get('source', 'unknown')}")
            if 'tables' in first_chunk['metadata']:
                print(f"Tables found: {len(first_chunk['metadata']['tables'])}")
            if 'key_value_pairs' in first_chunk['metadata']:
                print(f"Key-value pairs found: {len(first_chunk['metadata']['key_value_pairs'])}")
            if 'entities' in first_chunk['metadata']:
                print(f"Entities found: {len(first_chunk['metadata']['entities'])}")
    
    print("\n✨ Indexer test completed successfully!")

if __name__ == "__main__":
    test_indexer() 