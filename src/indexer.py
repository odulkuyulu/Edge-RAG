"""
Document Indexer Module

This module handles document processing and indexing for the RAG system.
It uses Azure Document Intelligence for document processing and Qdrant for vector storage.

Key Features:
- Document processing with Azure Document Intelligence
- Language detection and entity extraction
- Document chunking and embedding generation
- Vector storage in Qdrant
- Support for multiple document formats (txt, json, csv)
- Bilingual support (Arabic and English)
"""

import os
import uuid
import json
import csv
import textwrap
import requests
import ollama
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import tempfile

# ================================
# 🔹 CONFIGURATION & INITIAL SETUP
# ================================

# Load environment variables
load_dotenv()

# Azure AI Services Configuration
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
AZURE_DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTEL_KEY")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "1024"))

# Initialize Azure AI Services clients
try:
    document_analysis_client = DocumentAnalysisClient(
        endpoint=AZURE_DOC_INTEL_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
    )
    AZURE_DOC_INTEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: Azure Document Intelligence not available: {e}")
    AZURE_DOC_INTEL_AVAILABLE = False
    document_analysis_client = None

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ================================
# 🔹 QDRANT COLLECTION SETUP
# ================================

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int = None) -> None:
    """
    Creates a Qdrant collection if it doesn't exist.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
        vector_size: Size of the vectors to store (defaults to EMBEDDING_SIZE from env)
    """
    if vector_size is None:
        vector_size = EMBEDDING_SIZE

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )
        print(f"Created new collection '{collection_name}'")

# Create collections for both languages
for lang in ["en", "ar"]:
    collection_name = f"rag_docs_{lang}"
    create_collection_if_not_exists(client, collection_name)

print("✅ Qdrant collections are now correctly set up!")

# ================================
# 🔹 LANGUAGE DETECTION FUNCTION
# ================================

def detect_language(text: str) -> str:
    """
    Detects the language of a given text using Azure Language Service.
    
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
        
        payload = {
            "documents": [{
                "id": "1",
                "text": text
            }]
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if "documents" in result and result["documents"]:
            detected_lang = result["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "arabic" if detected_lang == "ar" else "english"
            
    except Exception as e:
        print(f"Error detecting language: {e}")
    
    return "english"  # Default to English if detection fails

# ================================
# 🔹 EMBEDDING GENERATION FUNCTION
# ================================

def generate_embedding(text: str, language: str) -> List[float]:
    """
    Generates embeddings using bge-m3 for both Arabic & English.
    
    Args:
        text: The text to generate embedding for
        language: The language of the text ("arabic" or "english")
        
    Returns:
        List[float]: The generated embedding vector
    """
    try:
        response = ollama.embeddings(model="bge-m3", prompt=text)
        embedding = response["embedding"]

        # Ensure embedding size matches Qdrant vector size
        if len(embedding) < EMBEDDING_SIZE:
            embedding = np.pad(embedding, (0, EMBEDDING_SIZE - len(embedding)), 'constant', constant_values=0)
        elif len(embedding) > EMBEDDING_SIZE:
            embedding = embedding[:EMBEDDING_SIZE]  # Truncate if too large

        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# ================================
# 🔹 DOCUMENT PROCESSING FUNCTION
# ================================

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
        print(f"Error extracting entities: {e}")
    
    return []

# ================================
# 🔹 DOCUMENT INTELLIGENCE PROCESSING
# ================================

def process_with_document_intelligence(file_path: str) -> Dict[str, Any]:
    """
    Process document using Azure Document Intelligence.
    Extracts structured content including tables, key-value pairs, and entities.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dict[str, Any]: Structured content and metadata
    """
    if not AZURE_DOC_INTEL_AVAILABLE:
        print("Azure Document Intelligence not available, using basic text extraction")
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            return {
                "text": content.decode('utf-8', errors='ignore'),
                "tables": [],
                "key_value_pairs": [],
                "entities": [],
                "metadata": {
                    "page_count": 1,
                    "languages": [],
                    "styles": []
                }
            }
        except Exception as e:
            print(f"Error in basic text extraction: {e}")
            return None

    try:
        with open(file_path, "rb") as f:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-document", f
            )
        result = poller.result()

        # Extract structured content
        content = {
            "text": result.content,
            "tables": [],
            "key_value_pairs": [],
            "entities": [],
            "metadata": {
                "page_count": len(result.pages),
                "languages": [],
                "styles": []
            }
        }

        # Process pages and extract metadata
        for page in result.pages:
            # Extract page-level metadata
            page_metadata = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "angle": page.angle
            }
            content["metadata"]["styles"].append(page_metadata)

        # Process tables
        for table in result.tables:
            table_data = {
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": []
            }
            
            for cell in table.cells:
                table_data["cells"].append({
                    "text": cell.content,
                    "row_index": cell.row_index,
                    "column_index": cell.column_index
                })
            content["tables"].append(table_data)

        # Process key-value pairs
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                content["key_value_pairs"].append({
                    "key": kv_pair.key.content,
                    "value": kv_pair.value.content,
                    "confidence": kv_pair.confidence
                })

        # Process entities
        for entity in result.entities:
            content["entities"].append({
                "text": entity.text,
                "category": entity.category,
                "confidence": entity.confidence
            })

        return content
        
    except Exception as e:
        print(f"Error processing document with Azure Document Intelligence: {e}")
        # Fallback to basic text extraction
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            return {
                "text": content.decode('utf-8', errors='ignore'),
                "tables": [],
                "key_value_pairs": [],
                "entities": [],
                "metadata": {
                    "page_count": 1,
                    "languages": [],
                    "styles": []
                }
            }
        except Exception as e:
            print(f"Error in fallback text extraction: {e}")
            return None

# ================================
# 🔹 ENHANCED DOCUMENT PROCESSING
# ================================

def process_document(content: Union[str, bytes], filename: str = None) -> List[Dict[str, Any]]:
    """
    Process a document and prepare it for indexing.
    Handles both plain text and structured documents.
    
    Args:
        content: The document content (text or binary)
        filename: Optional path to the document file
        
    Returns:
        List[Dict[str, Any]]: List of processed document chunks with metadata
    """
    # Initialize variables
    text = ""
    additional_metadata = {}
    
    # First try to process with Document Intelligence if it's a file
    if filename and os.path.exists(filename):
        doc_intelligence_result = process_with_document_intelligence(filename)
        if doc_intelligence_result:
            # Use the structured content from Document Intelligence
            text = doc_intelligence_result["text"]
            additional_metadata = {
                "tables": doc_intelligence_result["tables"],
                "key_value_pairs": doc_intelligence_result["key_value_pairs"],
                "entities": doc_intelligence_result["entities"],
                "metadata": doc_intelligence_result["metadata"]
            }
    else:
        # If no filename or file doesn't exist, use the content directly
        if isinstance(content, bytes):
            # For binary content, we need to process it with Document Intelligence
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] if filename else '.pdf') as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                doc_intelligence_result = process_with_document_intelligence(temp_path)
                if doc_intelligence_result:
                    text = doc_intelligence_result["text"]
                    additional_metadata = {
                        "tables": doc_intelligence_result["tables"],
                        "key_value_pairs": doc_intelligence_result["key_value_pairs"],
                        "entities": doc_intelligence_result["entities"],
                        "metadata": doc_intelligence_result["metadata"]
                    }
            finally:
                os.unlink(temp_path)
        else:
            text = content
    
    # If we still don't have text, try to decode the content
    if not text and isinstance(content, bytes):
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
    
    # Split text into chunks
    chunks = textwrap.wrap(text, CHUNK_SIZE) if text.strip() else []
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Detect language for each chunk
        language = detect_language(chunk)
        
        # Extract entities from the chunk
        entities = extract_entities(chunk, language)
        
        # Group entities by category
        entities_by_category = {}
        for entity in entities:
            category = entity["category"]
            if category not in entities_by_category:
                entities_by_category[category] = []
            entities_by_category[category].append(entity["text"])
        
        # Combine Document Intelligence metadata with chunk metadata
        chunk_metadata = {
            "chunk_id": i,
            "total_chunks": len(chunks),
            "source": filename or "unknown",
            "language": language,
            "entities": entities_by_category,
            "metadata": additional_metadata.get("metadata", {})
        }
        
        # Add Document Intelligence metadata if available
        if additional_metadata:
            chunk_metadata.update(additional_metadata)
        
        processed_chunks.append({
            "text": chunk,
            "metadata": chunk_metadata
        })
    
    return processed_chunks

# ================================
# 🔹 DOCUMENT INDEXING FUNCTION
# ================================

def index_document(file_path: str, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> bool:
    """Indexes a document in Qdrant after processing it."""
    try:
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        
        # Set source in metadata if not present
        if "source" not in metadata:
            metadata["source"] = os.path.basename(file_path)
        
        print(f"Processing document: {metadata['source']}")
        
        # Process document with Document Intelligence if it's a PDF
        if file_path.lower().endswith('.pdf'):
            print("PDF detected, using Document Intelligence...")
            doc_intelligence_result = process_with_document_intelligence(file_path)
            if doc_intelligence_result:
                print("Document Intelligence processing successful")
                text = doc_intelligence_result["text"]
                metadata.update({
                    "tables": doc_intelligence_result["tables"],
                    "key_value_pairs": doc_intelligence_result["key_value_pairs"],
                    "entities": doc_intelligence_result["entities"],
                    "metadata": doc_intelligence_result["metadata"]
                })
            else:
                print("Document Intelligence processing failed, falling back to basic text extraction")
                # Fallback to basic text extraction
                if isinstance(content, bytes):
                    try:
                        text = content.decode('utf-8')
                    except UnicodeDecodeError:
                        text = content.decode('latin-1')
                else:
                    text = content
        else:
            # For non-PDF files, use the content directly
            if isinstance(content, bytes):
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')
            else:
                text = content
        
        # Split text into chunks
        chunks = textwrap.wrap(text, CHUNK_SIZE) if text.strip() else []
        print(f"Split document into {len(chunks)} chunks")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Detect language for the chunk
            language = detect_language(chunk)
            print(f"Detected language: {language}")
            
            # Extract entities from the chunk
            entities = extract_entities(chunk, language)
            print(f"Extracted {len(entities)} entities")
            
            # Generate embedding for the chunk
            embedding = generate_embedding(chunk, language)
            if embedding is None:
                print("Failed to generate embedding, skipping chunk")
                continue
            
            # Prepare metadata for the chunk
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "entities": entities,
                "language": language
            }
            
            # Create point for Qdrant
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "metadata": chunk_metadata
                }
            )
            
            # Add to appropriate collection
            collection_name = f"rag_docs_{'ar' if language == 'arabic' else 'en'}"
            client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            print(f"Indexed chunk in collection: {collection_name}")
        
        print("Document indexing completed successfully")
        return True
        
    except Exception as e:
        print(f"Error indexing document {file_path}: {e}")
        return False

# ================================
# 🔹 DOCUMENT LOADING FUNCTION
# ================================

def load_documents() -> List[Dict[str, Any]]:
    """
    Loads text, JSON, and CSV documents from the `data/` folder.
    
    Returns:
        List[Dict[str, Any]]: List of documents with their content and filenames
    """
    documents = []

    for file in os.listdir("data"):
        file_path = os.path.join("data", file)

        # Load text files
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    documents.append({"text": text, "filename": file})

        # Load JSON files
        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for doc in json_data:
                    if "text" in doc and doc["text"].strip():
                        documents.append({"text": doc["text"], "filename": file})

        # Load CSV files (assuming columns: text, lang)
        elif file.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"].strip():
                        documents.append({"text": row["text"], "filename": file})

    return documents

# ================================
# 🔹 MAIN EXECUTION
# ================================

if __name__ == "__main__":
    documents = load_documents()

    if not documents:
        print("⚠️ No documents to index. Exiting...")
        exit()

    # Process each document
    total_docs = len(documents)
    for i, doc in enumerate(documents, start=1):
        print(f"📄 Processing document {i}/{total_docs}...")
        index_document(doc["filename"], doc["text"])

    print(f"✅ Successfully indexed {total_docs} documents from `data/` folder!")