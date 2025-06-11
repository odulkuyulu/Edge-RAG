"""
Vector Store Module

This module provides a client for interacting with Qdrant vector database. It handles:
1. Document storage with vector embeddings
2. Similarity search using cosine distance
3. Collection management
4. Metadata tracking for documents

The module uses Qdrant as the underlying vector database, which provides efficient
similarity search capabilities for high-dimensional vectors.
"""

import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    """
    Data class representing a search result from the vector database.
    
    Attributes:
        text (str): The text content of the matched document
        metadata (Dict[str, Any]): Additional information about the document
        score (float): Similarity score between query and document
    """
    text: str
    metadata: Dict[str, Any]
    score: float

class VectorDBClient:
    """
    A client for interacting with Qdrant vector database.
    
    This class provides methods for:
    - Creating and managing collections
    - Adding documents with embeddings
    - Performing similarity search
    - Managing document metadata
    """
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector database client.
        
        Args:
            collection_name (str): Name of the collection to use
        """
        self.collection_name = collection_name
        self.client = QdrantClient("localhost", port=6333)
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """
        Create the collection if it doesn't exist.
        
        The collection is configured with:
        - 1024-dimensional vectors (matching the embedding model)
        - Cosine distance for similarity measurement
        """
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
    
    def add_document(self, text: str, embedding: List[float], metadata: Dict[str, Any]):
        """
        Add a document to the vector store.
        
        Args:
            text (str): The document text content
            embedding (List[float]): Vector embedding of the document
            metadata (Dict[str, Any]): Additional document metadata
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID for the document
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata
                    }
                )
            ]
        )
    
    def search(self, query_embedding: List[float], limit: int = 5) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding (List[float]): Vector embedding of the query
            limit (int): Maximum number of results to return
            
        Returns:
            List[SearchResult]: List of similar documents with their scores
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            SearchResult(
                text=hit.payload["text"],
                metadata=hit.payload["metadata"],
                score=hit.score
            )
            for hit in results
        ]
    
    def delete_collection(self):
        """Delete the entire collection from the vector database."""
        self.client.delete_collection(collection_name=self.collection_name)

    def get_unique_sources(self) -> List[str]:
        """
        Get a list of unique source files in the vector store.
        
        Returns:
            List[str]: Sorted list of unique source file names
        """
        try:
            # Get all points from the collection
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust this number based on your needs
            )[0]
            
            # Extract unique source files from metadata
            sources = set()
            for point in points:
                if "metadata" in point.payload and "source" in point.payload["metadata"]:
                    sources.add(point.payload["metadata"]["source"])
            
            return sorted(list(sources))
        except Exception as e:
            print(f"Error getting unique sources: {e}")
            return []

    def store_embeddings(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store multiple document chunks and their embeddings in Qdrant.
        
        Args:
            doc_id (str): Unique identifier for the document
            chunks (List[Dict[str, Any]]): List of text chunks with metadata
            embeddings (List[List[float]]): List of embeddings corresponding to chunks
        """
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "metadata": {
                        **chunk["metadata"],
                        "doc_id": doc_id
                    }
                }
            )
            points.append(point)
            
        # Upsert points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def search_by_text(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text similarity.
        
        Note: This method assumes the query text can be converted to a vector
        by splitting on spaces. For proper text search, use the embedding model
        to generate embeddings first.
        
        Args:
            query_text (str): Query text to search for
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with text, metadata, and scores
        """
        results = self.search(query_embedding=[float(x) for x in query_text.split()], limit=top_k)
        return [
            {
                "text": result.text,
                "metadata": result.metadata,
                "score": result.score
            }
            for result in results
        ] 